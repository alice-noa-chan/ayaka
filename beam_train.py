"""beam.cloud GPU training deployment for Electra/AYAKA.

Run from the repo root — beam syncs the workspace (see .beamignore)
into the remote container and executes the decorated function on GPU.
Invoke by calling the wrapper remotely under the beam SDK env:

    python -c "import beam_train; beam_train.smoke.remote()"  # env check
    python -c "import beam_train; beam_train.train.remote()"  # real run

The file must live at the repo root: the SDK derives the remote
handler name from the module's relative path, so a nested file
produces an unimportable handler on Windows.

Runtime configuration comes from env vars (pass via --env KEY=VAL or
the function's env mapping below):

    AYAKA_MODEL_SIZE   electra-small|electra-base|electra-large|tiny
    AYAKA_STEPS        optimizer steps (default 1000)
    AYAKA_SAMPLES_PER_STEP  mixture samples drawn per step (default 64)
    AYAKA_TOKEN_BUDGET packed-batch token budget (default 65536)
    AYAKA_LIMIT_PER_SPEC    max rows loaded per dataset spec
    AYAKA_SPECS        comma-separated DATASET_SPECS keys
    AYAKA_TOKENIZER    path to a tokenizers JSON on the volume (""
                       -> HashTokenizer fallback)
    AYAKA_RUN_NAME     run dir under /artifacts (default timestamped)
    AYAKA_COMPILE      "1" -> torch.compile(inductor)
    HF_TOKEN           (secret) optional, for gated datasets

Checkpoints, metrics history, run config, and the dataset manifest are
persisted on the `ayaka-artifacts` volume mounted at /artifacts.
"""

from __future__ import annotations

import os
import sys
import time

from beam import GpuType, Image, Volume, function

# ------------------------------------------------------------ remote image
#
# torch ships in the image build; flash-attn compiles against it
# (--no-build-isolation). Attention falls back to SDPA when flash-attn
# is unavailable, so the `|| true` keeps the image build resilient.

IMAGE = Image(
    python_version="python3.11",
    python_packages=[
        "torch>=2.2",
        "tokenizers>=0.19",
        "datasets>=2.18",
        "numpy>=1.26",
    ],
).add_commands(
    [
        "pip install flash-attn --no-build-isolation || true",
    ]
)

VOLUME = Volume(name="ayaka-artifacts", mount_path="/artifacts")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _run(smoke: bool, **overrides) -> dict:
    """Executed inside the remote container. The synced repo is the
    working directory; make it importable, then drive run_training.
    Keyword args override the env-derived RunConfig fields."""
    sys.path.insert(0, os.getcwd())
    import torch

    from ayaka.training.run import RunConfig, run_training, synthetic_pools

    print(f"[beam] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[beam] gpu: {torch.cuda.get_device_name(0)}", flush=True)

    specs = [s for s in _env("AYAKA_SPECS", "").split(",") if s] or None
    cfg = RunConfig(
        model_size=_env("AYAKA_MODEL_SIZE", "tiny" if smoke else "electra-small"),
        steps=int(_env("AYAKA_STEPS", "3" if smoke else "1000")),
        samples_per_step=int(_env("AYAKA_SAMPLES_PER_STEP", "8" if smoke else "64")),
        token_budget=int(_env("AYAKA_TOKEN_BUDGET", "4096" if smoke else "65536")),
        lr=float(_env("AYAKA_LR", "3e-5")),
        limit_per_spec=int(_env("AYAKA_LIMIT_PER_SPEC", "20000")),
        tokenizer_path=_env("AYAKA_TOKENIZER", ""),
        artifacts_dir="/artifacts",
        run_name=_env("AYAKA_RUN_NAME", time.strftime("%Y%m%d-%H%M%S")),
        compile=_env("AYAKA_COMPILE", "0") == "1",
    )
    if specs:
        cfg.specs = specs
    if "specs" in overrides and isinstance(overrides["specs"], str):
        overrides["specs"] = [s for s in overrides["specs"].split(",") if s]
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise TypeError(f"unknown RunConfig field: {k}")
        setattr(cfg, k, v)

    # Smoke mode skips dataset downloads: synthetic pools validate the
    # whole path (forward/backward, packing, checkpoint, volume writes).
    result = run_training(cfg, pools=synthetic_pools(n_per_cell=8)) if smoke else run_training(cfg)
    print(f"[beam] done: {result}", flush=True)
    return result


@function(
    name="ayaka-train",
    cpu=8,
    memory="48Gi",
    gpu=GpuType.A10G,  # serverless; also RTX4090 / L40S / A100-40 / H100
    gpu_count=1,
    image=IMAGE,
    volumes=[VOLUME],
    timeout=-1,  # long-running: no container cap
    retries=0,
    headless=True,  # keep training after the client disconnects
    secrets=["HF_TOKEN"],
    env={
        "TOKENIZERS_PARALLELISM": "false",
        "HF_DATASETS_TRUST_REMOTE_CODE": "0",
    },
)
def train(**overrides) -> dict:
    """Full fine-tune run on GPU (sec 49.1 stage-1, token-budget).

    Per-invocation config: `train.remote(steps=300, limit_per_spec=3000)`
    — kwargs are RunConfig fields; `specs` also accepts a comma string.
    """
    return _run(smoke=False, **overrides)


@function(
    name="ayaka-smoke",
    cpu=4,
    memory="16Gi",
    gpu=GpuType.A10G,
    gpu_count=1,
    image=IMAGE,
    volumes=[VOLUME],
    timeout=1800,
    retries=0,
    headless=False,
)
def smoke() -> dict:
    """Tiny synthetic run: verifies image, GPU, imports, and volume
    writes before spending on a real run."""
    return _run(smoke=True)
