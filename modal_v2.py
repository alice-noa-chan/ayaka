"""CPU preparation followed by one bounded H100 worker, with no retries.

    modal run modal_v2.py

The maximum attached-GPU function lifetime is eight hours. CPU preparation
downloads pinned weights into the persistent volume before GPU allocation.
"""

import json

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
    .env({"HF_HOME": "/runs/hf-cache", "PYTHONPATH": "/root", "TOKENIZERS_PARALLELISM": "false"})
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
def explore_h100(budget_scale: float = 1.0):
    from ayaka.experiments.v2 import bounded_run

    volume.reload()
    try:
        return bounded_run("/root/candidates.json", "/runs/exploration", scale=budget_scale)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(prepare_only: bool = False, budget_scale: float = 1.0):
    preparation = prepare_cpu.remote()
    print(json.dumps(preparation, indent=2))
    if not prepare_only:
        print(json.dumps(explore_h100.remote(budget_scale), indent=2))
