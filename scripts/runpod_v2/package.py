"""Build a relocatable, offline Linux training kit without starting a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

VERSION = "ayaka-runpod-offline-1"
SNAPSHOT_FILES = (
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def digest(path):
    path = Path(path)
    # Portable Python's bundled OpenSSL can hash large tensors slowly. Ubuntu's
    # coreutils uses the same SHA-256 and avoids minutes of billed startup work.
    if path.stat().st_size >= 64 * 1024**2 and Path("/usr/bin/sha256sum").is_file():
        checksum = (
            subprocess.run(
                ["/usr/bin/sha256sum", "--zero", str(path)],
                check=True,
                capture_output=True,
            )
            .stdout[:64]
            .decode("ascii")
        )
        if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("invalid system SHA-256 output")
        return checksum
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def fingerprint_tree(root):
    """Never package an external or absolute symlink as a portable dependency."""
    root = Path(root).resolve()
    files = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            if os.path.isabs(target) or not path.resolve().is_relative_to(root):
                raise ValueError(f"nonportable link: {name}")
            files[name] = {"symlink": target}
        elif path.is_file():
            files[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    return files


def validate_weights(snapshot, audit):
    snapshot = Path(snapshot)
    for name, expected in audit["shards"].items():
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("invalid pinned weight layout")
        path = snapshot / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"missing or incomplete pretrained weight: {name}")
        if digest(path) != expected["sha256"]:
            raise ValueError(f"pretrained weight checksum mismatch: {name}")
    for name in SNAPSHOT_FILES:
        if not (snapshot / name).is_file():
            raise ValueError(f"missing native processor/tokenizer file: {name}")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def archive_kit(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError("archive must be a new file outside the kit")
    # Large BF16 weights are not recompressed on a billed GPU; extraction is tar-only.
    # Buffer cross-filesystem writes too: tar's small default blocks make a
    # 23GB kit unnecessarily slow on mounted host/network volumes.
    with (
        out.open("wb", buffering=8 * 1024 * 1024) as destination,
        tarfile.open(fileobj=destination, mode="w", dereference=False) as archive,
    ):
        archive.add(root, arcname="ayaka-v2", filter=_archive_metadata)
    checksum = digest(out)
    out.with_name(out.name + ".sha256").write_text(f"{checksum}  {out.name}\n", encoding="utf-8")
    return {"path": str(out), "bytes": out.stat().st_size, "sha256": checksum}


def _archive_metadata(row):
    row.uid = row.gid = 0
    row.uname = row.gname = "root"
    return row


def build_kit(repo, bundle, snapshot, runtime, out):
    # Imports occur only when assembling an actual validated kit, not during basic tests.
    from ayaka.training.prepare_v2 import validate_bundle
    from ayaka.training.run_v2 import source_matches

    repo, bundle, snapshot, runtime, out = map(Path, (repo, bundle, snapshot, runtime, out))
    if out.exists():
        raise ValueError("kit output must be a new directory")
    manifest, _ = validate_bundle(bundle)
    if not source_matches(manifest):
        raise ValueError("source does not match the immutable training bundle")
    recipe = json.loads((bundle / "training_config.json").read_text())
    audit = json.loads((bundle / "weight_cache.json").read_text())
    if (
        audit["repo"] != recipe["model"]["backbone"]
        or audit["revision"] != recipe["model"]["backbone_revision"]
        or audit["bundle_manifest_sha256"] != digest(bundle / "manifest.json")
        or audit["optimizer_steps"] != 0
    ):
        raise ValueError("weight audit and immutable training bundle differ")
    if not (runtime / "bin/python3.11").is_file():
        raise ValueError("prepared Linux CPython 3.11 runtime is required")
    print("[package] checking pinned weights", flush=True)
    validate_weights(snapshot, audit)
    out.mkdir(parents=True)
    # Exact source bytes matter; newline conversion would invalidate the prepared bundle.
    shutil.copytree(
        repo / "ayaka", out / "ayaka", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copytree(bundle, out / "bundle")
    shutil.copytree(
        runtime,
        out / "runtime",
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        repo / "scripts/runpod_v2",
        out / "scripts/runpod_v2",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(repo / "LICENSE", out / "LICENSE")
    hub = out / "hf-cache/hub" / ("models--" + audit["repo"].replace("/", "--"))
    destination = hub / "snapshots" / audit["revision"]
    destination.mkdir(parents=True)
    names = [*SNAPSHOT_FILES, *audit["shards"]]
    if (snapshot / "model.safetensors.index.json").is_file():
        names.append("model.safetensors.index.json")
    for name in names:
        shutil.copyfile(snapshot / name, destination / name)
    for name in ("README.md", "LICENSE", "LICENSE.txt"):
        if (snapshot / name).is_file():
            shutil.copyfile(snapshot / name, destination / name)
    write_json(
        out / "launch.json",
        {
            "steps": 1200,
            "max_train_seconds": 14400,
            "checkpoint_every": 100,
            "cpu_threads": 4,
            "gpu_name_contains": "RTX PRO 6000",
            "minimum_gpu_gib": 90,
            "minimum_free_disk_gib": 20,
            "bundle_manifest_sha256": digest(bundle / "manifest.json"),
            "schedule_sha256": json.loads((bundle / "workload-1200.json").read_text())[
                "schedule_sha256"
            ],
        },
    )
    print("[package] hashing offline runtime, dataset and model files", flush=True)
    files = fingerprint_tree(out)
    write_json(out / "kit-manifest.json", {"version": VERSION, "files": files})
    result = subprocess.run(
        [
            str(out.resolve() / "runtime/bin/python3.11"),
            str(out.resolve() / "scripts/runpod_v2/launch.py"),
            "audit",
        ],
        cwd=out,
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if result.returncode != 0:
        raise ValueError("relocated offline runtime audit failed")
    return {
        "path": str(out),
        "files": len(files),
        "bytes": sum(row.get("bytes", 0) for row in files.values()),
        "manifest_sha256": digest(out / "kit-manifest.json"),
        "optimizer_steps": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--archive")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    result = build_kit(repo, args.bundle, args.snapshot, args.runtime, args.out)
    if args.archive:
        result["archive"] = archive_kit(args.out, args.archive)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
