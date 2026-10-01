"""CPU preparation followed by one bounded H100 worker, with no retries.

    modal run modal_v2.py

The maximum attached-GPU function lifetime is eight hours. CPU preparation
downloads pinned weights into the persistent volume before GPU allocation.
"""

import json
from pathlib import Path

import modal

app = modal.App("ayaka-v2-exploration")
volume = modal.Volume.from_name("ayaka-v2-exploration", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "transformers==5.17.0",
        "peft==0.21.0",
        "accelerate==1.15.0",
        "safetensors==0.8.0",
        "huggingface_hub==1.33.0",
        "tokenizers==0.23.2",
        "numpy==2.1.2",
        "flash-linear-attention==0.5.0",
        "datasets==5.0.1",
    )
    .env(
        {
            "HF_HOME": "/runs/hf-cache",
            "PYTHONPATH": "/root",
            "TOKENIZERS_PARALLELISM": "false",
            "TRITON_CACHE_DIR": "/runs/kernel-cache",
        }
    )
    .add_local_dir("ayaka", remote_path="/root/ayaka")
    .add_local_file("docs/experiments/v2_candidates.json", remote_path="/root/candidates.json")
)
COMMON = {
    "image": image,
    "volumes": {"/runs": volume},
    "cpu": 4,
    "memory": 32768,
    "max_containers": 1,
    "retries": 0,
}


@app.function(timeout=4 * 3600, **COMMON)
def prepare_cpu():
    from ayaka.experiments.v2 import prepare

    try:
        return prepare("/root/candidates.json", "/runs/exploration", download_weights=True)
    finally:
        volume.commit()


@app.function(gpu="H100", timeout=8 * 3600, **COMMON)
def explore_h100(
    budget_scale: float = 1.0,
    recover_screen: bool = False,
    refresh_curriculum: bool = False,
    resume_paired_screen: bool = False,
    screen_candidates: str = "",
):
    from ayaka.experiments.v2 import bounded_run

    volume.reload()
    try:
        options = {}
        if recover_screen:
            # Use the one-hour recovery allocation to repeat the failed screen;
            # retain its original two-hour reservation and omit reproduction.
            options = {
                "stages": ("screen", "heads", "sft", "evaluate"),
                "stage_limits": {"screen": 3600},
            }
        if refresh_curriculum:
            options = {
                "stages": ("screen", "heads", "sft", "evaluate"),
                "stage_limits": {"screen": 3000, "heads": 600},
            }
        if resume_paired_screen:
            options = {"stages": ("recover_screen", "heads", "sft", "evaluate")}
        return bounded_run(
            "/root/candidates.json",
            "/runs/exploration",
            scale=budget_scale,
            screen_candidates=tuple(screen_candidates.split(",")) if screen_candidates else (),
            **options,
        )
    finally:
        volume.commit()


@app.function(timeout=600, **COMMON)
def budget_cpu(observations=None):
    from ayaka.experiments.budget import reconcile_closed_windows

    volume.reload()
    try:
        if observations:
            return reconcile_closed_windows("/runs/exploration", observations)
        path = Path("/runs/exploration/budget.json")
        return json.loads(path.read_text()) if path.exists() else {"elapsed_s": 0}
    finally:
        volume.commit()


@app.local_entrypoint()
def main(
    prepare_only: bool = False,
    budget_scale: float = 1.0,
    recover_screen: bool = False,
    refresh_curriculum: bool = False,
    resume_paired_screen: bool = False,
    closed_windows: str = "",
    screen_candidates: str = "",
):
    if (not recover_screen and not resume_paired_screen) or refresh_curriculum:
        preparation = prepare_cpu.remote()
        print(json.dumps(preparation, indent=2))
    if not prepare_only:
        observations = json.loads(Path(closed_windows).read_text()) if closed_windows else None
        ledger = budget_cpu.remote(observations)
        remaining = int(8 * 3600 - ledger["elapsed_s"] - 120)
        if remaining <= 0:
            print(json.dumps({"status": "budget_exhausted", "ledger": ledger}, indent=2))
            return
        print(
            json.dumps(
                explore_h100.with_options(timeout=remaining).remote(
                    budget_scale,
                    recover_screen,
                    refresh_curriculum,
                    resume_paired_screen,
                    screen_candidates,
                ),
                indent=2,
            )
        )
