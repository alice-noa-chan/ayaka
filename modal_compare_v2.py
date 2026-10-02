"""Three matched zero-update GPU probes within the original exploration budget."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("ayaka-v2-gpu-comparison")
volume = modal.Volume.from_name("ayaka-v2-exploration", create_if_missing=False)
BUNDLE = (
    Path(os.environ.get("AYAKA_PROFILE_BUNDLE", "runs/v2-pretraining-20261002-ready"))
    if modal.is_local()
    else Path("/root/pretraining-bundle")
)
GPUS = {"h100": "H100!", "a100": "A100-80GB", "rtxpro6000": "RTX-PRO-6000"}
RESERVATION = 1300
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
    .add_local_file("benchmark_gpu_v2.py", remote_path="/root/benchmark_gpu_v2.py")
)
OPTIONS = {
    "image": image,
    "volumes": {"/runs": volume},
    "cpu": 4,
    "memory": 32768,
    "max_containers": 1,
    "retries": 0,
}


@app.function(timeout=1200, **OPTIONS)
def prepare_cpu(job):
    from ayaka.training.cache_v2 import prepare_weights
    from ayaka.training.prepare_v2 import canonical, validate_bundle
    from ayaka.training.run_v2 import source_matches

    volume.reload()
    manifest, _ = validate_bundle("/root/pretraining-bundle")
    if not source_matches(manifest):
        raise ValueError("immutable native package/bundle mismatch")
    root = Path("/runs/gpu-comparisons") / job
    root.mkdir(parents=True, exist_ok=False)
    prepare_weights("/root/pretraining-bundle", root / "weight_cache.json")
    original = json.loads(Path("/runs/exploration/budget.json").read_text())
    ledger_path = Path("/runs/pretraining-profile-reservations.json")
    ledger = json.loads(ledger_path.read_text())
    previous = original["elapsed_s"] + sum(row["reserved_gpu_seconds"] for row in ledger["jobs"])
    total = previous + RESERVATION * len(GPUS)
    if total > 28800:
        raise ValueError("three GPU reservations would exceed original eight-GPU-hour ceiling")
    for name, gpu in GPUS.items():
        ledger["jobs"].append(
            {
                "job": f"{job}-{name}",
                "gpu": gpu,
                "reserved_gpu_seconds": RESERVATION,
                "optimizer_updates_authorized": 0,
            }
        )
    ledger_path.write_bytes(canonical(ledger) + b"\n")
    volume.commit()
    return {
        "previous_cumulative_bound_seconds": previous,
        "comparison_reserved_gpu_seconds": RESERVATION * len(GPUS),
        "cumulative_gpu_bound_seconds": total,
    }


def run_probe(job, name):
    import subprocess
    import sys
    import time

    import torch

    volume.reload()
    root = Path("/runs/gpu-comparisons") / job / name
    ledger = json.loads(Path("/runs/pretraining-profile-reservations.json").read_text())
    if not any(
        row["job"] == f"{job}-{name}" and row["optimizer_updates_authorized"] == 0
        for row in ledger["jobs"]
    ):
        raise ValueError("CPU preparation and zero-update reservation required")
    gpu = torch.cuda.get_device_properties(0)
    expected = {"h100": ("H100", 75), "a100": ("A100", 75), "rtxpro6000": ("RTX PRO 6000", 90)}[
        name
    ]
    if expected[0] not in gpu.name or gpu.total_memory < expected[1] * 1024**3:
        raise ValueError("allocated GPU differs from requested benchmark hardware")
    start = time.monotonic()
    command = [
        sys.executable,
        "/root/benchmark_gpu_v2.py",
        "--bundle",
        "/root/pretraining-bundle",
        "--deadline",
        "970",
        "--out",
        str(root),
    ]
    try:
        result = subprocess.run(command, timeout=1000, check=False)
        summary = {
            "status": "complete" if result.returncode == 0 else "incomplete",
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        summary = {"status": "deadline_incomplete", "exit_code": 124}
    except Exception as exc:
        summary = {"status": "failed", "error": str(exc)}
    finally:
        volume.commit()
    summary.update(
        gpu=gpu.name,
        gpu_memory_bytes=gpu.total_memory,
        function_seconds=time.monotonic() - start,
        optimizer_steps=0,
    )
    reports = {
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob("*.json")
        if "io_probe" not in str(p)
    }
    return {"summary": summary, "reports": reports}


@app.function(gpu="H100!", timeout=1100, scaledown_window=2, **OPTIONS)
def h100(job):
    return run_probe(job, "h100")


@app.function(gpu="A100-80GB", timeout=1100, scaledown_window=2, **OPTIONS)
def a100(job):
    return run_probe(job, "a100")


@app.function(gpu="RTX-PRO-6000", timeout=1100, scaledown_window=2, **OPTIONS)
def rtxpro6000(job):
    return run_probe(job, "rtxpro6000")


@app.local_entrypoint()
def main(out: str):
    root = Path(out)
    if root.exists():
        raise ValueError("comparison needs a new local output directory")
    job = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    preparation = prepare_cpu.remote(job)
    root.mkdir(parents=True)
    (root / "preparation.json").write_text(
        json.dumps({**preparation, "job": job}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(preparation, indent=2))
    failed = []
    for name, function in (("h100", h100), ("a100", a100), ("rtxpro6000", rtxpro6000)):
        result = function.remote(job)
        target = root / name
        target.mkdir()
        for relative, raw in result["reports"].items():
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (target / "summary.json").write_text(
            json.dumps(result["summary"], indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"backend": name, **result["summary"]}, indent=2))
        if result["summary"]["status"] != "complete":
            failed.append(name)
    if failed:
        raise RuntimeError(f"Incomplete GPU comparisons preserved: {failed}")
