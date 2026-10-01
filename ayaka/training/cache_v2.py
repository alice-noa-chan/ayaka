"""Prepare pinned native pretrained weights on CPU; never start inference or training."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import RemoteEntryNotFoundError

from .prepare_v2 import canonical, sha256, validate_bundle


def prepare_weights(bundle, out):
    root = Path(bundle)
    if Path(out).exists():
        raise ValueError("weight audit output must be a new file")
    validate_bundle(root)
    recipe = json.loads((root / "training_config.json").read_text(encoding="utf-8"))
    config = recipe["model"]
    audit = json.loads((root / "model_preflight.json").read_text(encoding="utf-8"))
    if (
        audit["license"] not in {"apache-2.0", "mit"}
        or audit["repo"] != config["backbone"]
        or audit["revision"] != config["backbone_revision"]
    ):
        raise ValueError("weight cache must match the approved pinned native model audit")
    start = time.monotonic()
    try:
        index = Path(
            hf_hub_download(
                config["backbone"],
                "model.safetensors.index.json",
                revision=config["backbone_revision"],
            )
        )
        metadata = json.loads(index.read_text(encoding="utf-8"))
        shards = sorted(set(metadata["weight_map"].values()))
        size, snapshot = metadata["metadata"]["total_size"], index.parent
    except RemoteEntryNotFoundError:
        info = HfApi().model_info(
            config["backbone"], revision=config["backbone_revision"], files_metadata=True
        )
        weight = next((s for s in info.siblings if s.rfilename == "model.safetensors"), None)
        if weight is None or weight.size is None:
            raise ValueError("pinned repository has no supported safe weight layout") from None
        shards, size = [weight.rfilename], weight.size
        snapshot = Path(
            hf_hub_download(config["backbone"], "config.json", revision=config["backbone_revision"])
        ).parent
    if any(Path(s).name != s or not s.endswith(".safetensors") for s in shards):
        raise ValueError("invalid pretrained shard names")
    cached_size = sum((snapshot / s).stat().st_size for s in shards if (snapshot / s).is_file())
    if shutil.disk_usage(HF_HUB_CACHE).free < max(0, size - cached_size) * 2 + 1024**3:
        raise ValueError("insufficient disk space for pinned weights and download staging")
    snapshot = Path(
        snapshot_download(
            config["backbone"],
            revision=config["backbone_revision"],
            allow_patterns=[*shards, "*.json", "*.jinja", "*.model"],
            max_workers=4,
        )
    )
    fingerprints = {}
    for shard in shards:
        digest = hashlib.sha256()
        with (snapshot / shard).open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
        fingerprints[shard] = {
            "sha256": digest.hexdigest(),
            "bytes": (snapshot / shard).stat().st_size,
        }
    report = {
        "repo": config["backbone"],
        "revision": config["backbone_revision"],
        "bundle_manifest_sha256": sha256((root / "manifest.json").read_bytes()),
        "status": "pinned_native_weights_cached_no_training",
        "optimizer_steps": 0,
        "shards": fingerprints,
        "download_and_audit_seconds": time.monotonic() - start,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(canonical(report) + b"\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = prepare_weights(args.bundle, args.out)
    print(
        json.dumps(
            {
                "status": report["status"],
                "shards": len(report["shards"]),
                "bytes": sum(s["bytes"] for s in report["shards"].values()),
                "optimizer_steps": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
