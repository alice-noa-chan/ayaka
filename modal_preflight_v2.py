"""One bounded zero-update H100 profile inside the original exploration ceiling.

CPU preparation and pinned-weight audit precede GPU allocation. This entry point
has no optimizer-training mode. Each invocation reserves 1200 conservative GPU
seconds, uses H100! (no H200 upgrade), and performs no automatic retries.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("ayaka-v2-pretraining-profile")
volume = modal.Volume.from_name("ayaka-v2-exploration", create_if_missing=False)
BUNDLE = (
    Path(os.environ.get("AYAKA_PROFILE_BUNDLE", "runs/v2-pretraining-20261002-ready"))
    if modal.is_local()
    else Path("/root/pretraining-bundle")
)
profile_image = (
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
    .pip_install("torchvision==0.23.0", "pillow==11.3.0")
    .env(
        {
            "HF_HOME": "/runs/hf-cache",
            "PYTHONPATH": "/root",
            "TOKENIZERS_PARALLELISM": "false",
            "TRITON_CACHE_DIR": "/runs/kernel-cache",
        }
    )
    .add_local_dir("ayaka", remote_path="/root/ayaka")
    .add_local_dir(str(BUNDLE), remote_path="/root/pretraining-bundle")
)
OPTIONS = {
    "image": profile_image,
    "volumes": {"/runs": volume},
    "cpu": 4,
    "memory": 32768,
    "max_containers": 1,
    "retries": 0,
}
RESERVATION = 1200


@app.function(timeout=1200, **OPTIONS)
def prepare_cpu(job):
    from ayaka.training.cache_v2 import prepare_weights
    from ayaka.training.prepare_v2 import canonical, validate_bundle
    from ayaka.training.run_v2 import source_matches

    volume.reload()
    manifest, _ = validate_bundle("/root/pretraining-bundle")
    if not source_matches(manifest):
        raise ValueError("uploaded package sources do not match the immutable bundle")
    root = Path("/runs/pretraining-profiles") / job
    if root.exists():
        raise ValueError("profile job identity must be new")
    root.mkdir(parents=True)
    prepare_weights("/root/pretraining-bundle", root / "weight_cache.json")
    old = json.loads(Path("/runs/exploration/budget.json").read_text())
    ledger_path = Path("/runs/pretraining-profile-reservations.json")
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"jobs": []}
    if (
        old["elapsed_s"] + sum(row["reserved_gpu_seconds"] for row in ledger["jobs"]) + RESERVATION
        > 28800
    ):
        raise ValueError("original cumulative eight-GPU-hour ceiling would be exceeded")
    ledger["jobs"].append(
        {"job": job, "reserved_gpu_seconds": RESERVATION, "optimizer_updates_authorized": 0}
    )
    ledger_path.write_bytes(canonical(ledger) + b"\n")
    volume.commit()
    return {
        "status": "cpu_ready_gpu_not_started",
        "previous_gpu_bound_seconds": old["elapsed_s"],
        "reserved_gpu_seconds": RESERVATION,
        "cumulative_gpu_bound_seconds": old["elapsed_s"]
        + sum(row["reserved_gpu_seconds"] for row in ledger["jobs"]),
    }


@app.function(gpu="H100!", timeout=900, scaledown_window=2, **OPTIONS)
def profile_h100(job):
    import subprocess
    import sys
    import time

    import torch

    volume.reload()
    start = time.monotonic()
    root = Path("/runs/pretraining-profiles") / job
    ledger = json.loads(Path("/runs/pretraining-profile-reservations.json").read_text())
    if not any(
        row["job"] == job and row["optimizer_updates_authorized"] == 0 for row in ledger["jobs"]
    ):
        raise ValueError("CPU-only preparation/reservation is required")
    gpu = torch.cuda.get_device_properties(0)
    if "H100" not in gpu.name:
        raise ValueError("benchmark requires a real H100, not an automatic hardware upgrade")
    command = [
        sys.executable,
        "-m",
        "ayaka.training.run_v2",
        "--bundle",
        "/root/pretraining-bundle",
        "--profile-only",
        "--planned-steps",
        "1200",
        "--device",
        "cuda",
        "--max-train-seconds",
        "780",
        "--out",
        str(root / "measurement"),
    ]
    try:
        completed = subprocess.run(command, timeout=820, check=False)
        status = "complete" if completed.returncode == 0 else "failed_or_deadline_incomplete"
        result = {"status": status, "exit_code": completed.returncode}
    except subprocess.TimeoutExpired:
        result = {"status": "deadline_incomplete", "exit_code": 124}
    finally:
        volume.commit()
    result.update(
        gpu=gpu.name,
        gpu_memory_bytes=gpu.total_memory,
        measured_function_seconds=time.monotonic() - start,
        optimizer_steps=0,
    )
    files = {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*.json")
        if "untrained_io_probe" not in str(path)
    }
    return {"summary": result, "files": files}


@app.local_entrypoint()
def main(out: str):
    target = Path(out)
    if target.exists() or not (BUNDLE / "manifest.json").is_file():
        raise ValueError("use a prepared bundle and a new local output directory")
    job = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    preparation = prepare_cpu.remote(job)
    print(json.dumps(preparation, indent=2))
    result = profile_h100.remote(job)
    target.mkdir(parents=True)
    for name, raw in result["files"].items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (target / "summary.json").write_text(
        json.dumps({**result["summary"], "job": job, "preparation": preparation}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2))
    if result["summary"]["status"] != "complete":
        raise RuntimeError(
            "zero-update profiling incomplete; partial reports saved, no training admitted"
        )
