"""Model-bound temperature fits with strict modality and partition isolation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from .calibrate import MIN_QUESTIONS, fit_temperature
from .path_calibration import PathCalibration, path_key


def checkpoint_fingerprint(path, *, include_calibration=True):
    root, digest = Path(path), hashlib.sha256()
    files = [root / "ayaka_config.json", root / "head.safetensors"]
    files += sorted((root / "adapter").glob("*.json")) + sorted(
        (root / "adapter").glob("*.safetensors")
    )
    if (
        not all(p.is_file() for p in files)
        or not (root / "adapter" / "adapter_model.safetensors").is_file()
    ):
        raise ValueError("scoped artifacts require a native config, safe head and saved adapter")
    calibration = root / "noul_calibration.json"
    if include_calibration and calibration.is_file():
        files.append(calibration)
    for file in files:
        digest.update(str(file.relative_to(root)).replace("\\", "/").encode())
        with file.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


class ScopedCalibration:
    def __init__(self, model_id, modality, partition, temperatures=None, counts=None):
        if (
            not isinstance(model_id, str)
            or len(model_id) != 64
            or any(c not in "0123456789abcdef" for c in model_id)
        ):
            raise ValueError("scoped calibration requires a SHA-256 model identity")
        if modality not in {"text", "image"} or partition not in {"fixed", "generated_finite"}:
            raise ValueError("unsupported calibration modality or partition")
        self.model_id, self.modality, self.partition = model_id, modality, partition
        self.calibration = PathCalibration(temperatures)
        self.counts = counts or {}

    @classmethod
    def fit(cls, rows, model_id, modality, partition):
        if not rows or any(
            r.get("split") != "calibration"
            or r.get("model_id") != model_id
            or r.get("modality") != modality
            or r.get("partition") != partition
            for r in rows
        ):
            raise ValueError(
                "calibration rows must match the reserved split and exact model/domain"
            )
        groups = defaultdict(list)
        for row in rows:
            if not row.get("cluster_id"):
                raise ValueError("calibration requires independent source lineage")
            budget = row["budget"] if row["route"] != "direct" else 0
            groups[path_key(row["type"], row["route"], budget)].append(row)
        temperatures, counts = {}, {}
        for key, group in groups.items():
            clusters = {row["cluster_id"] for row in group}
            counts[key] = {"questions": len(group), "independent_sources": len(clusters)}
            if len(clusters) >= MIN_QUESTIONS:
                temperatures[key] = fit_temperature(
                    [[math.log(max(p, 1e-12)) for p in r["probs"]] for r in group],
                    [r["target"] for r in group],
                )
        return cls(model_id, modality, partition, temperatures, counts)

    def validate_binding(self, model_id, modality, partition):
        if (model_id, modality, partition) != (self.model_id, self.modality, self.partition):
            raise ValueError("calibration model/modality/partition binding mismatch")

    def apply(self, probs, kind, route, budget):
        key = path_key(kind, route, budget)
        if key not in self.calibration.temperatures:
            return list(probs)
        return self.calibration.apply(probs, kind, route, budget)

    def covers(self, kind, route, budget):
        return path_key(kind, route, budget) in self.calibration.temperatures

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                {
                    "version": 1,
                    "model_id": self.model_id,
                    "modality": self.modality,
                    "partition": self.partition,
                    "temperatures": self.calibration.temperatures,
                    "counts": self.counts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.pop("version", None) != 1:
            raise ValueError("unsupported scoped calibration version")
        return cls(**payload)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--modality", choices=("text", "image"), required=True)
    parser.add_argument("--partition", choices=("fixed", "generated_finite"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if (
        report.get("split") != "calibration"
        or not report.get("complete")
        or Path(args.out).exists()
    ):
        raise ValueError("fit requires a complete reserved calibration report and new output file")
    rows = [
        row
        for group in report["rows"].values()
        for row in group
        if row["modality"] == args.modality and row["partition"] == args.partition
    ]
    result = ScopedCalibration.fit(
        rows, checkpoint_fingerprint(args.checkpoint), args.modality, args.partition
    )
    if not result.calibration.temperatures:
        raise ValueError("no path has enough independent calibration sources")
    result.save(args.out)
    return result


if __name__ == "__main__":
    main()
