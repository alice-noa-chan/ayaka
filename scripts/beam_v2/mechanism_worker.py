"""One bounded, offline, frozen-weight diagnostic. No training entry point."""

import json
import math
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

try:
    from . import worker
except ImportError:  # Beam syncs this directory, not its parent package.
    import worker

RUN_NAME = "ayaka-mechanism-20261003-v1"
SECONDS = 3400


def overlay_allowed(name):
    p = PurePosixPath(name)
    return (
        not p.is_absolute()
        and ".." not in p.parts
        and "\\" not in name
        and (
            name
            in {
                "mechanism-plan.json",
                "mechanism-overlay-manifest.json",
                "ayaka/eval/mechanism_v2.py",
                "ayaka/eval/v2.py",
            }
            or name
            in {
                "diagnostic-pilot/" + n
                for n in ("ayaka_config.json", "electra_config.json", "head.safetensors")
            }
            or (
                len(p.parts) == 3
                and p.parts[:2] == ("diagnostic-pilot", "adapter")
                and p.suffix in {".json", ".safetensors"}
            )
        )
    )


def apply_overlay(archive, kit, expected):
    import zstandard

    archive, kit = Path(archive), Path(kit)
    if worker.digest(archive) != expected:
        raise ValueError("overlay checksum mismatch")
    names = set()
    with (
        archive.open("rb") as raw,
        zstandard.ZstdDecompressor().stream_reader(raw) as decoded,
        tarfile.open(fileobj=decoded, mode="r|") as tar,
    ):
        for member in tar:
            if not member.isfile() or not overlay_allowed(member.name) or member.name in names:
                raise ValueError("unexpected or duplicate overlay member")
            names.add(member.name)
            tar.extract(member, kit, filter="data")
    manifest = json.loads((kit / "mechanism-overlay-manifest.json").read_text())
    if names != set(manifest["files"]) | {"mechanism-overlay-manifest.json"}:
        raise ValueError("overlay inventory mismatch")
    for name, expected_file in manifest["files"].items():
        path = kit / name
        if (
            not overlay_allowed(name)
            or path.stat().st_size != expected_file["bytes"]
            or worker.digest(path) != expected_file["sha256"]
        ):
            raise ValueError("overlay file checksum mismatch")
    return manifest


def runtime_environment(kit):
    kit = Path(kit)
    return {
        "HF_HOME": str(kit / "hf-cache"),
        "HF_HUB_CACHE": str(kit / "hf-cache/hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(kit),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    }


def validate_admission(admission):
    if (
        admission.get("gpu") != "RTX5090"
        or admission.get("task_timeout_seconds") != SECONDS
        or admission.get("cpu_cores") != 2
        or admission.get("memory_gib") != 32
        or admission.get("automatic_retry") is not False
        or admission.get("full_training") is not False
    ):
        raise ValueError("resource admission does not match the fixed diagnostic")
    if float(admission["planned_ceiling_usd"]) > min(
        float(admission["credit_usd"]), float(admission["remaining_spend_cap_usd"])
    ):
        raise ValueError("diagnostic exceeds available credit or allowance")
    if not all(
        math.isfinite(float(admission[k]))
        for k in ("planned_ceiling_usd", "credit_usd", "remaining_spend_cap_usd")
    ):
        raise ValueError("invalid resource budget")
    if (
        admission.get("compute_ceiling_usd") != "2.35"
        or admission.get("planned_ceiling_usd") != "2.45"
    ):
        raise ValueError("require the frozen conservative cost envelope")


def execute(volume, overlay_sha, plan_sha, admission):
    validate_admission(admission)
    volume = Path(volume)
    if (volume / (RUN_NAME + ".receipt.json")).exists():
        raise ValueError("single admitted run already has a receipt; refuse paid retry")
    started = time.monotonic()
    root = Path(tempfile.mkdtemp(prefix="ayaka-mechanism-"))
    output = root / "result"
    output.mkdir()
    status, error = "failed", None

    def progress(phase):
        worker.write_json(volume / "progress.json", {"phase": phase, "at": time.time()})

    try:
        if shutil.disk_usage(root).free < 60 * 1024**3:
            raise ValueError("offline diagnostic requires 60GiB local scratch")
        progress("stage_immutable_kit")
        archive = worker.stage_archive(volume / worker.ARCHIVE, root / worker.ARCHIVE)
        progress("verify_and_extract_immutable_kit")
        kit = worker.extract_kit(archive, root)
        overlay = root / "mechanism-overlay-v1.tar.zst"
        shutil.copyfile(volume / overlay.name, overlay)
        manifest = apply_overlay(overlay, kit, overlay_sha)
        if worker.digest(kit / "mechanism-plan.json") != plan_sha:
            raise ValueError("plan checksum differs from the admitted plan")
        if json.loads((kit / "mechanism-plan.json").read_text())["run_name"] != RUN_NAME:
            raise ValueError("plan does not identify this separately admitted run")
        worker.write_json(output / "overlay.json", manifest)
        worker.write_json(output / "admission.json", admission)
        shutil.copyfile(kit / "mechanism-plan.json", output / "plan.json")
        python = kit / "runtime/bin/python3.11"
        env_args = ["env", *[f"{k}={v}" for k, v in runtime_environment(kit).items()]]
        probe = "import json,torch; p=torch.cuda.get_device_properties(0); print(json.dumps(dict(gpu=p.name,bytes=p.total_memory,count=torch.cuda.device_count(),bf16=torch.cuda.is_bf16_supported(),torch=torch.__version__)))"
        hardware = subprocess.run(
            [*env_args, str(python), "-c", probe], capture_output=True, text=True, timeout=120
        )
        (output / "hardware.stderr.log").write_text(hardware.stderr)
        hardware.check_returncode()
        observed = json.loads(hardware.stdout)
        worker.validate_hardware("RTX5090", observed)
        worker.write_json(output / "hardware.json", observed)
        remaining = SECONDS - (time.monotonic() - started) - 180
        if remaining < 2300:
            raise TimeoutError(
                "insufficient complete-cohort envelope after staging; no inference started"
            )
        progress("frozen_inference")
        worker.run_logged(
            [
                *env_args,
                str(python),
                "-m",
                "ayaka.eval.mechanism_v2",
                "--plan",
                str(kit / "mechanism-plan.json"),
                "--parent",
                str(kit / "v1-checkpoint"),
                "--pilot",
                str(kit / "diagnostic-pilot"),
                "--out",
                str(output / "mechanism"),
                "--seconds",
                str(remaining - 30),
            ],
            output / "execution.log",
            remaining,
            progress,
        )
        complete = json.loads((output / "mechanism/complete.json").read_text())
        if complete != {
            "complete": True,
            "optimizer_steps": 0,
            "full_training_started": False,
            "test_evaluated": False,
            "checkpoints_evaluated": ["parent"],
        }:
            raise ValueError("diagnostic completion receipt is invalid")
        status = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(error, flush=True)
    worker.write_json(
        output / "operational-outcome.json",
        {
            "status": status,
            "error": error,
            "elapsed_seconds": time.monotonic() - started,
            "optimizer_steps": 0,
            "automatic_retry": False,
        },
    )
    archive = root / (RUN_NAME + ".tar.zst")
    progress("package_and_deliver")
    receipt = worker.package_results(output, archive)
    destination = volume / archive.name
    shutil.copyfile(archive, destination)
    if worker.digest(destination) != receipt["sha256"]:
        raise ValueError("durable result copy checksum mismatch")
    receipt.update(status=status, error=error)
    worker.write_json(volume / (RUN_NAME + ".receipt.json"), receipt)
    progress("durable_receipt")
    return receipt
