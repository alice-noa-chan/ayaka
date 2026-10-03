"""Verify the durable result receipt before using any delivered checkpoint."""

import argparse
import json
import tarfile
from pathlib import Path

from scripts.beam_v2.worker import digest


def verify_delivery(archive, receipt, destination, *, completion="training"):
    import zstandard

    archive, destination = Path(archive), Path(destination)
    if completion not in {"training", "mechanism"}:
        raise ValueError("unsupported result completion type")
    if archive.stat().st_size != receipt["bytes"] or digest(archive) != receipt["sha256"]:
        raise ValueError("downloaded result archive does not match its durable receipt")
    destination.mkdir(exist_ok=False, parents=True)
    with (
        archive.open("rb") as raw,
        zstandard.ZstdDecompressor().stream_reader(raw) as decoded,
        tarfile.open(fileobj=decoded, mode="r|") as tar,
    ):
        for member in tar:
            if member.name.split("/")[0] != "result" or member.issym() or member.islnk():
                raise ValueError("result archive contains an unexpected root or link")
            tar.extract(member, destination, filter="data")
    root = destination / "result"
    manifest = json.loads((root / "receipt-manifest.json").read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"receipt-manifest.json"}:
        raise ValueError("result member inventory differs from the receipt manifest")
    if len(manifest["files"]) != receipt["files"]:
        raise ValueError("durable receipt file count mismatch")
    for name, entry in manifest["files"].items():
        path = root / name
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError("delivered result member checksum mismatch: " + name)
    outcome = json.loads((root / "operational-outcome.json").read_text())
    if outcome["status"] != receipt["status"]:
        raise ValueError("result status differs from the durable receipt")
    complete = None
    if receipt["status"] == "complete" and completion == "mechanism":
        complete = json.loads((root / "mechanism/complete.json").read_text())
        plan = json.loads((root / "plan.json").read_text())
        expected_complete = {
            "complete": True,
            "optimizer_steps": 0,
            "full_training_started": False,
            "test_evaluated": False,
        }
        if "active_checkpoints" in plan:
            if plan["active_checkpoints"] not in [["parent"], ["parent", "pilot"]]:
                raise ValueError("diagnostic checkpoint scope is invalid")
            expected_complete["checkpoints_evaluated"] = plan["active_checkpoints"]
        if complete != expected_complete:
            raise ValueError("diagnostic receipt claims training or lacks completion")
        for checkpoint in plan.get("active_checkpoints", ["parent", "pilot"]):
            report = json.loads((root / ("mechanism/" + checkpoint + ".json")).read_text())
            if (
                not report["complete"]
                or report["model_id"] != plan["checkpoints"][checkpoint]
                or report["cohort_sha256"] != plan["cohort_sha256"]
                or report["optimizer_steps"] != 0
            ):
                raise ValueError("diagnostic report differs from the frozen plan")
            expected_keys = {
                f"{c}/{h}"
                for c in ("direct", "empty", "generated", "oracle", "distractor")
                for h in ("lm", "pointer", "hybrid")
            }
            if set(report["rows"]) != expected_keys or any(
                len(rows) != plan["questions_per_checkpoint"] for rows in report["rows"].values()
            ):
                raise ValueError("diagnostic did not finish every context and head")
    elif receipt["status"] == "complete":
        complete = json.loads((root / "recovery/complete.json").read_text())
        if not complete["complete"] or complete["training_steps"] != 200:
            raise ValueError("successful delivery did not complete the fixed 200-step schedule")
        seal = json.loads((root / "recovery/pilot/checkpoint/complete.json").read_text())
        if seal != {"steps": 200, "complete": True}:
            raise ValueError("delivered checkpoint is not sealed after 200 steps")
    return {"files_verified": len(manifest["files"]), "outcome": outcome, "completion": complete}


def verify_tensors(checkpoint):
    import torch
    from safetensors import safe_open

    checkpoint = Path(checkpoint)
    files = [checkpoint / "head.safetensors", checkpoint / "adapter/adapter_model.safetensors"]
    count = 0
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if not keys:
                raise ValueError("delivered model contains an empty tensor file")
            for key in keys:
                if not torch.isfinite(handle.get_tensor(key)).all():
                    raise ValueError("delivered model contains nonfinite tensors")
            count += len(keys)
    return {"all_finite": True, "tensor_count": count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--completion", choices=("training", "mechanism"), default="training")
    args = parser.parse_args()
    result = verify_delivery(
        args.archive,
        json.loads(args.receipt.read_text()),
        args.destination,
        completion=args.completion,
    )
    if result["completion"] and args.completion == "training":
        result["tensors"] = verify_tensors(args.destination / "result/recovery/pilot/checkpoint")
    args.report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
