"""Validated image routing bound to exact checkpoint and fixed partition semantics."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ..routing import BenefitRouter, fit_router, paired_training_rows
from .scoped_calibration import ScopedCalibration, checkpoint_fingerprint


class ScopedRouter:
    def __init__(self, model_id, modality, router):
        self.binding = ScopedCalibration(model_id, modality, "fixed")
        self.model_id, self.modality, self.partition = model_id, modality, "fixed"
        self.router = router.validate_promoted()

    def validate_binding(self, model_id, modality, partition):
        self.binding.validate_binding(model_id, modality, partition)

    def should_reason(self, state, spec, baseline, tok, budget=384):
        is_image = getattr(state, "is_multimodal", False)
        if is_image != (self.modality == "image"):
            raise ValueError("router received a different input modality")
        # Features use only raw text and direct-readout confidence, never image
        # annotations, targets, object repr/memory addresses or proposed traces.
        text = state.state if is_image else state
        return self.router.should_reason(text, spec, baseline, tok, budget)

    @classmethod
    def fit(cls, train_report, dev_report, model_id, modality):
        collections = []
        for report, split in [(train_report, "router_train"), (dev_report, "dev")]:
            if not report.get("complete") or report.get("split") != split:
                raise ValueError("scoped router requires complete router_train and dev reports")
            selected = {}
            for mode, rows in report["rows"].items():
                selected[mode] = [
                    r for r in rows if r["modality"] == modality and r["partition"] == "fixed"
                ]
                if any(r.get("model_id") != model_id for r in selected[mode]):
                    raise ValueError("router measurements belong to a different model")
            if "off" not in selected or not {"low", "medium", "high"} <= selected.keys():
                raise ValueError("router reports require paired off/low/medium/high measurements")
            collections.append(
                [
                    pair
                    for mode in ("low", "medium", "high")
                    for pair in paired_training_rows(selected["off"], selected[mode], split)
                ]
            )
        return cls(model_id, modality, fit_router(*collections))

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                {
                    "version": 1,
                    "model_id": self.model_id,
                    "modality": self.modality,
                    "router": asdict(self.router),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.pop("version", None) != 1:
            raise ValueError("unsupported scoped router version")
        return cls(payload["model_id"], payload["modality"], BenefitRouter(**payload["router"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-report", required=True)
    parser.add_argument("--dev-report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--modality", choices=("text", "image"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if Path(args.out).exists():
        raise ValueError("router output must be a new file")
    reports = [
        json.loads(Path(p).read_text(encoding="utf-8"))
        for p in (args.train_report, args.dev_report)
    ]
    result = ScopedRouter.fit(*reports, checkpoint_fingerprint(args.checkpoint), args.modality)
    result.save(args.out)
    return result


if __name__ == "__main__":
    main()
