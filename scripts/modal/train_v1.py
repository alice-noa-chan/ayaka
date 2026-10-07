"""Modal entry point: run any ``ayaka.pipeline`` command on A100-80GB or H100.

    modal run scripts/modal/train_v1.py --cmd "train --model electra-small --run small-v1"
    modal run scripts/modal/train_v1.py --cmd "train --model electra-large --run large-v1 --set steps=3000" --gpu h100
    modal run scripts/modal/train_v1.py --cmd "teacher --ckpt /runs/large-v1/checkpoint --out /runs/large-v1/teacher.jsonl"
    modal run scripts/modal/train_v1.py --cmd "export --ckpt /runs/small-distill/checkpoint --name electra-small"
    modal volume get ayaka-runs exports ./exports        # download exported models

Artifacts and the Hugging Face cache live on the ``ayaka-runs`` volume, so
weights download once. Modal bills per second while the function runs.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import modal

# Local paths resolve from the repository root, so `modal run` works from any directory.
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")

app = modal.App("ayaka-electra")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6",
        "transformers>=5.6",
        "peft>=0.19",
        "accelerate>=1.0",
        "datasets>=3.0",
        "huggingface_hub>=0.30",
        "safetensors",
        "numpy>=1.26",
        "liger-kernel",
    )
    .env({"HF_HOME": "/runs/hf-cache", "AYAKA_ARTIFACTS": "/runs", "PYTHONPATH": "/root"})
    .add_local_dir(ROOT / "ayaka", remote_path="/root/ayaka")  # includes eval data files
)

runs = modal.Volume.from_name("ayaka-runs", create_if_missing=True)
COMMON = {
    "image": image,
    "volumes": {"/runs": runs},
    "timeout": 24 * 3600,
    "cpu": 4,
    "memory": 32768,
}


def _run(argv: list[str]):
    from ayaka.pipeline import main

    try:
        return main(argv)
    finally:
        runs.commit()


@app.function(gpu="A100-80GB", **COMMON)
def pipeline_a100(argv: list[str]):
    return _run(argv)


@app.function(gpu="H100", **COMMON)
def pipeline_h100(argv: list[str]):
    return _run(argv)


@app.local_entrypoint()
def main(cmd: str, gpu: str = "a100"):
    fn = {"a100": pipeline_a100, "h100": pipeline_h100}[gpu.lower()]
    result = fn.remote(shlex.split(cmd))
    print(str(result)[:4000])
