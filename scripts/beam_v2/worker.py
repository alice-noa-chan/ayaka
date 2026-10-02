"""Run the verified offline kit on a bounded Beam reservation."""

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread

ARCHIVE = "ayaka-v2-clean-ready-20261002.tar.zst"
ARCHIVE_SHA = "a25b45abf551dc239d52509c90e52e0524bfc6643ed08543234106a651d3ab5e"
RECIPE_SHA = "7fa9ef8895051e5956adeba9e7084ddbfd7f3c9afc5f499a5957aa95cba82ab9"
TRAIN_SECONDS = 5400
EVAL_SECONDS = 3000
JOB_SECONDS = 9500


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def stage_archive(source, destination, *, workers=8, part_bytes=32 * 1024**2):
    """Read independent volume ranges concurrently, then verify locally."""
    source, destination = Path(source), Path(destination)
    size = source.stat().st_size
    with destination.open("xb") as stream:
        stream.truncate(size)

    def copy_range(start):
        stop = min(size, start + part_bytes)
        with source.open("rb") as reader, destination.open("r+b") as writer:
            reader.seek(start)
            writer.seek(start)
            while start < stop:
                data = reader.read(min(8 * 1024**2, stop - start))
                if not data:
                    raise ValueError("offline archive range was truncated")
                writer.write(data)
                start += len(data)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(copy_range, range(0, size, part_bytes)))
    if digest(destination) != ARCHIVE_SHA:
        raise ValueError("staged archive checksum mismatch")
    return destination


def extract_kit(archive, destination):
    import zstandard

    if digest(archive) != ARCHIVE_SHA:
        raise ValueError("offline archive checksum mismatch")
    destination = Path(destination)
    with (
        Path(archive).open("rb") as raw,
        zstandard.ZstdDecompressor().stream_reader(raw) as decoded,
        tarfile.open(fileobj=decoded, mode="r|", bufsize=8 * 1024**2) as tar,
    ):
        for member in tar:
            if member.name.split("/")[0] != "ayaka-v2":
                raise ValueError("archive escaped its expected root")
            tar.extract(member, destination, filter="data")
    kit = destination / "ayaka-v2"
    manifest = json.loads((kit / "clean-kit-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = kit / name
        if not path.resolve().is_relative_to(kit.resolve()):
            raise ValueError("manifest escaped the offline kit")
        if "symlink" in expected:
            if not path.is_symlink() or os.readlink(path) != expected["symlink"]:
                raise ValueError("runtime symlink mismatch")
        elif path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("extracted offline file checksum mismatch: " + name)
    return kit


def apply_training_allowance(kit):
    """Change only operational time; retain every dataset byte and model setting."""
    kit = Path(kit)
    recipe_path = kit / "bundle/training_config.json"
    manifest_path = kit / "bundle/manifest.json"
    if digest(recipe_path) != RECIPE_SHA:
        raise ValueError("refuse a different prepared training recipe")
    recipe = json.loads(recipe_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    before_manifest = digest(manifest_path)
    if recipe["completion_target_seconds"] != 3600:
        raise ValueError("unexpected original completion allowance")
    recipe["completion_target_seconds"] = TRAIN_SECONDS
    write_json(recipe_path, recipe)
    manifest["files"]["training_config.json"] = digest(recipe_path)
    write_json(manifest_path, manifest)
    overlay = {
        "original_archive_sha256": ARCHIVE_SHA,
        "original_recipe_sha256": RECIPE_SHA,
        "effective_recipe_sha256": digest(recipe_path),
        "original_manifest_sha256": before_manifest,
        "effective_manifest_sha256": digest(manifest_path),
        "changed_recipe_fields": {"completion_target_seconds": [3600, TRAIN_SECONDS]},
        "data_sha256": {k: v for k, v in manifest["files"].items() if k.endswith(".jsonl")},
        "optimizer_steps": 0,
    }
    write_json(kit / "beam-allowance.json", overlay)
    return overlay


def _prepare(volume):
    volume = Path(volume)
    with tempfile.TemporaryDirectory(prefix="ayaka-beam-cpu-") as temp:
        if shutil.disk_usage(temp).free < 60 * 1024**3:
            raise ValueError("CPU preparation needs 60GiB local disk before GPU reservation")
        write_json(volume / "prepare-progress.json", {"phase": "local_archive_staging"})
        archive = stage_archive(volume / ARCHIVE, Path(temp) / ARCHIVE)
        write_json(volume / "prepare-progress.json", {"phase": "local_extract_and_verify"})
        kit = extract_kit(archive, temp)
        overlay = apply_training_allowance(kit)
        write_json(volume / "prepare-progress.json", {"phase": "offline_cpu_plan"})
        subprocess.run(
            [
                "bash",
                str(kit / "scripts/runpod_v2/start_clean.sh"),
                "plan",
                "--out",
                str(kit / "outputs/beam-plan.json"),
            ],
            check=True,
            timeout=600,
        )
        receipt = {"ready": True, "archive_sha256": ARCHIVE_SHA, "overlay": overlay}
        write_json(volume / "ready.json", receipt)
        shutil.copyfile(kit / "outputs/beam-plan.json", volume / "plan.json")
        write_json(volume / "prepare-progress.json", {"phase": "complete"})
        return receipt


def prepare(volume):
    try:
        return _prepare(volume)
    except Exception as exc:
        write_json(
            Path(volume) / "ready.json",
            {"ready": False, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise


def run_logged(command, log_path, timeout, progress=None):
    """Stream progress and kill the whole subprocess tree on a timeout."""
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=os.name == "posix",
        )

        def forward():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                if progress:
                    progress(line[-500:])

        reader = Thread(target=forward, daemon=True)
        reader.start()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            raise
        finally:
            reader.join(timeout=10)
            process.stdout.close()
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)


def package_results(directory, archive):
    import zstandard

    directory, archive = Path(directory), Path(archive)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("result archive must not contain symlinks")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
    write_json(directory / "receipt-manifest.json", {"files": files})
    with (
        archive.open("xb") as raw,
        zstandard.ZstdCompressor(level=3, threads=4, write_checksum=True).stream_writer(
            raw
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as tar,
    ):
        tar.add(directory, arcname="result", recursive=True)
    return {"sha256": digest(archive), "bytes": archive.stat().st_size, "files": len(files)}


def execute(volume, run_name, expires_at):
    volume = Path(volume)
    if not math.isfinite(expires_at) or not time.time() < expires_at <= time.time() + 10800:
        raise ValueError("require the real backend expiry within three hours")
    ready = json.loads((volume / "ready.json").read_text())
    if not ready.get("ready") or ready["archive_sha256"] != ARCHIVE_SHA:
        raise ValueError("CPU preparation must pass before GPU execution")
    if run_name != "clean-pilot-20261002" or (volume / (run_name + ".receipt.json")).exists():
        raise ValueError("require the one explicitly admitted, unused pilot run")
    started = time.monotonic()
    root = Path(tempfile.mkdtemp(prefix="ayaka-beam-gpu-"))
    output = root / "result"
    output.mkdir()
    status, error = "failed", None

    def progress(phase):
        write_json(volume / (run_name + ".progress.json"), {"phase": phase, "at": time.time()})

    try:
        if shutil.disk_usage(root).free < 60 * 1024**3:
            raise ValueError("GPU workspace needs at least 60GiB free disk")
        print("Verifying and extracting the offline GPU kit", flush=True)
        progress("local_archive_staging")
        archive = stage_archive(volume / ARCHIVE, root / ARCHIVE)
        progress("local_extract_and_verify")
        kit = extract_kit(archive, root)
        overlay = apply_training_allowance(kit)
        if overlay != ready["overlay"]:
            raise ValueError("GPU kit differs from CPU-admitted preparation")
        write_json(output / "allowance.json", overlay)
        python = kit / "runtime/bin/python3.11"
        probe = (
            "import json,torch; p=torch.cuda.get_device_properties(0); "
            "assert torch.cuda.device_count()==1 and 'A100' in p.name "
            "and p.total_memory>75*1024**3 and torch.cuda.is_bf16_supported(); "
            "print(json.dumps(dict(gpu=p.name,bytes=p.total_memory,torch=torch.__version__,"
            "cuda=torch.version.cuda)))"
        )
        result = subprocess.run(
            [str(python), "-c", probe], capture_output=True, text=True, timeout=120
        )
        (output / "hardware.json").write_text(result.stdout)
        (output / "hardware.stderr.log").write_text(result.stderr)
        result.check_returncode()
        remaining = min(
            JOB_SECONDS - (time.monotonic() - started) - 300,
            expires_at - time.time() - 600,
        )
        if remaining < TRAIN_SECONDS + EVAL_SECONDS:
            raise TimeoutError("insufficient whole-job envelope before optimizer updates")
        run_logged(
            [
                "bash",
                str(kit / "scripts/runpod_v2/start_clean.sh"),
                "execute",
                "--out",
                str(output / "recovery"),
                "--training-window-seconds",
                str(TRAIN_SECONDS),
                "--evaluation-window-seconds",
                str(EVAL_SECONDS),
            ],
            output / "execution.log",
            remaining,
            progress,
        )
        complete = json.loads((output / "recovery/complete.json").read_text())
        if not complete["complete"] or complete["training_steps"] != 200:
            raise ValueError("fixed training/evaluation did not complete")
        status = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(error, flush=True)
    write_json(
        output / "operational-outcome.json",
        {
            "status": status,
            "error": error,
            "elapsed_seconds": time.monotonic() - started,
            "automatic_retry": False,
        },
    )
    archive = root / (run_name + ".tar.zst")
    progress("packaging_results")
    receipt = package_results(output, archive)
    destination = volume / archive.name
    shutil.copyfile(archive, destination)
    if digest(destination) != receipt["sha256"]:
        raise ValueError("durable result copy checksum mismatch")
    receipt.update(status=status, error=error)
    write_json(volume / (run_name + ".receipt.json"), receipt)
    progress("durable_receipt")
    return receipt
