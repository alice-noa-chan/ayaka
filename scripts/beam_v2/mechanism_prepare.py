"""Build a small private inference overlay before allocating any GPU."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import zstandard

from ayaka.eval.mechanism_v2 import prepare_cohort
from ayaka.training.scoped_calibration import checkpoint_fingerprint
from scripts.beam_v2.worker import ARCHIVE_SHA, canonical, digest, write_json

PARENT = "26109de64d2e6417c64a7e0942ac2d18b43208f7d870ffabf7c8fb814ad606d2"
PILOT = "09ae3c010c2d8ca49acac3e013f95f9f667f59547e155b578826e0b7a33b383d"
RUN_NAME = "ayaka-mechanism-20261003"


def build_overlay(dev, parent, pilot, out):
    root = Path(__file__).resolve().parents[2]
    if checkpoint_fingerprint(parent) != PARENT or checkpoint_fingerprint(pilot) != PILOT:
        raise ValueError("require the two explicitly frozen checkpoints")
    out = Path(out)
    out.mkdir(exist_ok=False, parents=True)
    records = prepare_cohort(dev, per_rule=2)
    plan = {
        "run_name": RUN_NAME,
        "records": records,
        "cohort_sha256": hashlib.sha256(canonical(records)).hexdigest(),
        "questions_per_checkpoint": 36,
        "independent_cases": 12,
        "seed": 20261003,
        "budget": 512,
        "checkpoints": {"parent": PARENT, "pilot": PILOT},
        "archive_sha256": ARCHIVE_SHA,
        "dev_sha256": digest(dev),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "optimizer_steps": 0,
        "full_training": False,
        "test_evaluated": False,
    }
    write_json(out / "mechanism-plan.json", plan)
    files = {
        "mechanism-plan.json": out / "mechanism-plan.json",
        "ayaka/eval/mechanism_v2.py": root / "ayaka/eval/mechanism_v2.py",
        "ayaka/eval/v2.py": root / "ayaka/eval/v2.py",
    }
    pilot = Path(pilot)
    for name in ["ayaka_config.json", "electra_config.json", "head.safetensors"]:
        files["diagnostic-pilot/" + name] = pilot / name
    for path in sorted((pilot / "adapter").iterdir()):
        if path.suffix in {".json", ".safetensors"}:
            files["diagnostic-pilot/adapter/" + path.name] = path
    manifest = {
        "files": {
            name: {"bytes": p.stat().st_size, "sha256": digest(p)} for name, p in files.items()
        }
    }
    archive = out / "mechanism-overlay.tar.zst"
    with (
        archive.open("xb") as raw,
        zstandard.ZstdCompressor(level=3, threads=4, write_checksum=True).stream_writer(
            raw
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as tar,
    ):
        for name, path in files.items():
            if path.is_symlink():
                raise ValueError("overlay must contain regular files only")
            tar.add(path, arcname=name, recursive=False)
        data = canonical(manifest)
        member = tarfile.TarInfo("mechanism-overlay-manifest.json")
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    receipt = {
        "overlay_sha256": digest(archive),
        "plan_sha256": digest(out / "mechanism-plan.json"),
        "bytes": archive.stat().st_size,
        "files": len(files),
        "optimizer_steps": 0,
    }
    write_json(out / "overlay-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("dev", "parent", "pilot", "out"):
        ap.add_argument("--" + name, required=True, type=Path)
    args = ap.parse_args()
    print(json.dumps(build_overlay(args.dev, args.parent, args.pilot, args.out)))
