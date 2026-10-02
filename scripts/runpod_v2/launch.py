"""Audit or launch the prepared offline kit. No implicit download or dataset rebuild."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

VERSION = "ayaka-runpod-offline-1"
EXPECTED = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "transformers": "5.17.0",
    "peft": "0.21.0",
    "accelerate": "1.15.0",
    "safetensors": "0.8.0",
    "huggingface_hub": "1.33.0",
    "tokenizers": "0.23.2",
    "numpy": "2.1.2",
    "flash-linear-attention": "0.5.0",
    "datasets": "5.0.1",
    "pillow": "11.3.0",
}


def digest(path):
    path = Path(path)
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


def check_records(root, manifest):
    root = Path(root).resolve()
    if manifest.get("version") != VERSION or not manifest.get("files"):
        raise ValueError("unsupported or empty offline kit manifest")
    for name, expected in manifest["files"].items():
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts or "\\" in name or not parts.parts:
            raise ValueError("unsafe kit manifest path")
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"kit file escapes its root: {name}")
        if "symlink" in expected:
            if not path.is_symlink() or os.readlink(path) != expected["symlink"]:
                raise ValueError(f"kit symlink mismatch: {name}")
        elif (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or digest(path) != expected["sha256"]
        ):
            raise ValueError(f"kit file checksum mismatch: {name}")


def environment(root, threads):
    root = Path(root).resolve()
    if type(threads) is not int or threads != 4:
        raise ValueError("matched native runtime requires four CPU threads")
    return {
        "HF_HOME": str(root / "hf-cache"),
        "HF_HUB_CACHE": str(root / "hf-cache/hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TRITON_CACHE_DIR": str(root / "outputs/kernel-cache"),
    }


def check_pinned_weights(model, weight_audit, files):
    if (
        weight_audit["repo"] != model["backbone"]
        or weight_audit["revision"] != model["backbone_revision"]
        or weight_audit["optimizer_steps"] != 0
    ):
        raise ValueError("pretrained weight audit differs from the pinned model")
    snapshot = "hf-cache/hub/models--" + model["backbone"].replace("/", "--")
    snapshot += "/snapshots/" + model["backbone_revision"] + "/"
    for name, expected in weight_audit["shards"].items():
        if files.get(snapshot + name) != expected:
            raise ValueError("packaged weight bytes/digest differ from the original pinned audit")


def run_path(root, name):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name):
        raise ValueError("run name must be a simple directory name")
    return Path(root) / "outputs" / name


def train_arguments(root, config, name):
    return [
        "--bundle",
        str(Path(root) / "bundle"),
        "--execute",
        "--steps",
        str(config["steps"]),
        "--max-train-seconds",
        str(config["max_train_seconds"]),
        "--checkpoint-every",
        str(config["checkpoint_every"]),
        "--out",
        str(run_path(root, name)),
    ]


def audit(root):
    root = Path(root).resolve()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("prepared runtime targets Linux x86_64")
    if sys.version_info[:3] != (3, 11, 14):
        raise ValueError("prepared runtime requires CPython 3.11.14")
    if Path(sys.executable).resolve() != (root / "runtime/bin/python3.11").resolve():
        raise ValueError("use the bundled interpreter, not the host Python")
    config = json.loads((root / "launch.json").read_text())
    os.environ.update(environment(root, config["cpu_threads"]))
    sys.path.insert(0, str(root))
    print("[runpod] verifying packaged source, dataset, weights and runtime", flush=True)
    manifest = json.loads((root / "kit-manifest.json").read_text())
    check_records(root, manifest)
    for name, version in EXPECTED.items():
        if importlib.metadata.version(name).split("+")[0] != version:
            raise ValueError(f"runtime dependency version mismatch: {name}")
    import torch
    import torchvision  # noqa: F401 -- verify linked native vision libraries
    from transformers import AutoConfig, AutoProcessor

    from ayaka.training.prepare_v2 import validate_bundle
    from ayaka.training.run_v2 import source_matches
    from ayaka.training.workload import finite_workload

    torch.set_num_threads(config["cpu_threads"])
    if torch.version.cuda != "12.8":
        raise ValueError("prepared Torch must carry the verified CUDA 12.8 runtime")
    prepared, splits = validate_bundle(root / "bundle")
    if (
        not source_matches(prepared)
        or digest(root / "bundle/manifest.json") != config["bundle_manifest_sha256"]
    ):
        raise ValueError("immutable dataset/source binding mismatch")
    recipe = json.loads((root / "bundle/training_config.json").read_text())
    model = recipe["model"]
    check_pinned_weights(
        model, json.loads((root / "bundle/weight_cache.json").read_text()), manifest["files"]
    )
    AutoConfig.from_pretrained(
        model["backbone"], revision=model["backbone_revision"], local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(
        model["backbone"], revision=model["backbone_revision"], local_files_only=True
    )
    inventory = json.loads((root / "bundle/model_preflight.json").read_text())["train_inventory"]
    workload = finite_workload(
        inventory,
        config["steps"],
        recipe["training"]["questions_per_step"],
        recipe["training"]["seed"],
        recipe["language_sampling"],
    )
    if workload["schedule_sha256"] != config["schedule_sha256"]:
        raise ValueError("full training schedule differs from the prepared plan")
    result = {
        "status": "offline_kit_verified_no_training",
        "optimizer_steps": 0,
        "steps": config["steps"],
        "row_exposures": workload["total_rows"],
        "splits": {key: len(rows) for key, rows in splits.items()},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "tokenizer_vocab": len(processor.tokenizer),
    }
    print(json.dumps(result, indent=2), flush=True)
    return config


def require_gpu(root, config):
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("one CUDA GPU must be visible")
    gpu = torch.cuda.get_device_properties(0)
    if (
        config["gpu_name_contains"] not in gpu.name
        or gpu.total_memory < config["minimum_gpu_gib"] * 1024**3
    ):
        raise ValueError("expected RTX PRO 6000 96GB; allocated hardware differs")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("native BF16 support is required")
    if shutil.disk_usage(root).free < config["minimum_free_disk_gib"] * 1024**3:
        raise ValueError("insufficient free disk for the complete checkpoint schedule")
    print(f"[runpod] GPU {gpu.name}; {gpu.total_memory / 1024**3:.1f} GiB", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "train", "evaluate"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-name", default="v2-main-1200")
    parser.add_argument("--split", choices=("router_train", "dev", "calibration", "test"))
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("off", "auto", "low", "medium", "high"),
        default=["off", "low", "medium", "high"],
    )
    parser.add_argument("--max-evaluation-seconds", type=int)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = run_path(root, args.run_name)
    if args.action == "train" and output.exists():
        raise ValueError("run already exists; refusing to restart fresh training")
    if args.action == "evaluate" and (args.split is None or args.max_evaluation_seconds is None):
        raise ValueError("evaluation requires an explicit split and time allowance")
    config = audit(root)
    if args.action == "audit":
        return
    require_gpu(root, config)
    if args.action == "train":
        from ayaka.training.run_v2 import main as train

        meta = train(train_arguments(root, config, args.run_name))
        if not meta.get("complete") or meta["steps"] != config["steps"]:
            raise ValueError("the full native training plan did not complete")
        print(
            "[runpod] complete checkpoint saved; calibration/router/evaluation remain separate",
            flush=True,
        )
    else:
        from ayaka.eval.pretraining_v2 import main as evaluate

        completion = json.loads((output / "checkpoint/complete.json").read_text())
        if completion.get("complete") is not True or completion["steps"] != config["steps"]:
            raise ValueError("evaluation requires a complete trained checkpoint")
        report = evaluate(
            [
                "--bundle",
                str(root / "bundle"),
                "--checkpoint",
                str(output / "checkpoint"),
                "--split",
                args.split,
                "--out",
                str(output / f"{args.split}-evaluation.json"),
                "--max-evaluation-seconds",
                str(args.max_evaluation_seconds),
                "--proposals",
                "--modes",
                *args.modes,
            ]
        )
        if not report.get("complete"):
            raise ValueError("evaluation was incomplete; its report cannot be promoted")


if __name__ == "__main__":
    main()
