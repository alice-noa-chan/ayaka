"""Select a direct Noul correction using held-out source groups only.

Temperature scaling is the reference. A regularized positive-slope affine
logit correction is accepted only when source-fold NLL, Brier, and served
credit do not regress and the fixed 0.2/0.8 policy abstains less often.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import torch

from .calibrate import MIN_QUESTIONS, fit_temperatures

VERSION = 1
REGULARIZATION = (0.01, 0.05)
FOLDS = 5


def _sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-min(value, 700)))
    value = math.exp(max(value, -700))
    return value / (1 + value)


def _temperature(temperatures, tokens, threshold):
    bucket = "long" if tokens >= threshold else "short"
    value = temperatures.get(f"noul@{bucket}", temperatures.get("noul", 1.0))
    if type(value) not in (int, float) or not math.isfinite(value) or not 0.05 <= value <= 20:
        raise ValueError("Noul input temperatures must be finite scalars in 0.05..20")
    return value


def _metrics(rows, probabilities):
    nll, brier, credit, abstentions = 0.0, 0.0, 0.0, 0
    for row, p in zip(rows, probabilities, strict=True):
        target = row["target"][1]
        nll -= target * math.log(max(p, 1e-12)) + (1 - target) * math.log(max(1 - p, 1e-12))
        brier += (p - target) ** 2
        selected = 0 if p <= 0.2 else (1 if p >= 0.8 else None)
        abstentions += selected is None
        credit += 0 if selected is None else row["target"][selected]
    return {
        "nll": nll / len(rows),
        "brier": brier / len(rows),
        "credit": credit / len(rows),
        "abstentions": abstentions,
    }


def _fit_affine(rows, regularization):
    margin = torch.tensor([r["logits"][1] - r["logits"][0] for r in rows], dtype=torch.float64)
    target = torch.tensor([r["target"][1] for r in rows], dtype=torch.float64)
    parameters = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([parameters], lr=0.2, max_iter=150, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        log_scale = parameters[0].clamp(-math.log(20), math.log(20))
        bias = parameters[1].clamp(-5, 5)
        logits = log_scale.exp() * margin + bias
        loss = (torch.nn.functional.softplus(logits) - target * logits).mean()
        loss = loss + regularization * parameters.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    scale = math.exp(float(parameters[0].detach().clamp(-math.log(20), math.log(20))))
    bias = float(parameters[1].detach().clamp(-5, 5))
    return scale, bias


@dataclass
class NoulCalibration:
    scale: float = 1.0
    bias: float = 0.0
    input_temperatures: dict = field(default_factory=dict)
    long_threshold: int = 1024
    report: dict = field(default_factory=lambda: {"status": "not_selected"})

    def __post_init__(self):
        if (
            type(self.scale) not in (int, float)
            or not math.isfinite(self.scale)
            or not 0.05 <= self.scale <= 20
            or type(self.bias) not in (int, float)
            or not math.isfinite(self.bias)
            or not -5 <= self.bias <= 5
            or type(self.long_threshold) is not int
            or self.long_threshold < 1
            or not isinstance(self.report, dict)
            or not isinstance(self.input_temperatures, dict)
        ):
            raise ValueError("invalid direct Noul calibration parameters")
        for tokens in (0, self.long_threshold):
            _temperature(self.input_temperatures, tokens, self.long_threshold)

    @property
    def selected(self):
        return self.report.get("status") == "selected"

    def apply(self, probabilities, kind, tokens, *, route="direct"):
        if not self.selected or kind != "noul" or route != "direct":
            return list(probabilities)
        if (
            len(probabilities) != 2
            or any(not math.isfinite(p) or p < 0 for p in probabilities)
            or abs(sum(probabilities) - 1) > 1e-4
            or type(tokens) is not int
            or tokens < 1
        ):
            raise ValueError("Noul correction requires a normalized binary direct read")
        margin = math.log(max(probabilities[1], 1e-12)) - math.log(max(probabilities[0], 1e-12))
        margin *= _temperature(self.input_temperatures, tokens, self.long_threshold)
        p = min(1 - 1e-12, max(1e-12, _sigmoid(self.scale * margin + self.bias)))
        return [1 - p, p]

    def as_dict(self):
        return {
            "version": VERSION,
            "route": "direct",
            "modality": "text",
            "scale": self.scale,
            "bias": self.bias,
            "input_temperatures": self.input_temperatures,
            "long_threshold": self.long_threshold,
            "report": self.report,
        }

    @classmethod
    def from_dict(cls, payload):
        value = dict(payload)
        if (
            value.pop("version", None) != VERSION
            or value.pop("route", None) != "direct"
            or value.pop("modality", None) != "text"
        ):
            raise ValueError("unsupported Noul calibration format or domain")
        return cls(**value)


def fit_noul_calibration(rows, temperatures, long_threshold=1024):
    """Fit after automatic temperature calibration; never consume dev/test rows.

    ``logits`` are uncalibrated, in canonical [false, true] order. The stored
    input temperatures let serving recover their binary margin exactly once.
    """
    if any(r.get("split") != "calibration" for r in rows):
        raise ValueError("Noul calibration accepts only reserved calibration rows")
    rows = [r for r in rows if r.get("type") == "noul"]
    for row in rows:
        if (
            row.get("candidate_ids") != ["false", "true"]
            or not isinstance(row.get("cluster_id"), str)
            or not row["cluster_id"]
            or type(row.get("tokens")) is not int
            or row["tokens"] < 1
            or len(row.get("logits", [])) != 2
            or any(not math.isfinite(x) for x in row["logits"])
            or len(row.get("target", [])) != 2
            or any(not math.isfinite(y) or not 0 <= y <= 1 for y in row["target"])
            or abs(sum(row["target"]) - 1) > 1e-8
        ):
            raise ValueError("Noul calibration needs binary raw logits and source lineage")
    result = NoulCalibration(input_temperatures=dict(temperatures), long_threshold=long_threshold)
    clusters = {r["cluster_id"] for r in rows}
    result.report = {
        "status": "insufficient_data",
        "questions": len(rows),
        "independent_cases": len(clusters),
        "folds": FOLDS,
        "selection_split": "calibration",
        "thresholds": [0.2, 0.8],
        "regularization_candidates": list(REGULARIZATION),
    }
    fold = {c: int(hashlib.sha256(c.encode()).hexdigest(), 16) % FOLDS for c in sorted(clusters)}
    groups = [
        (
            [r for r in rows if fold[r["cluster_id"]] != f],
            [i for i, r in enumerate(rows) if fold[r["cluster_id"]] == f],
        )
        for f in range(FOLDS)
    ]
    if not rows or any(
        not test or len({r["cluster_id"] for r in train}) < MIN_QUESTIONS for train, test in groups
    ):
        return result
    baseline = [None] * len(rows)
    candidates = {strength: [None] * len(rows) for strength in REGULARIZATION}
    for train, test in groups:
        fitted = fit_temperatures(
            [r["logits"] for r in train],
            [r["target"] for r in train],
            ["noul"] * len(train),
            [r["tokens"] for r in train],
            long_threshold,
        )
        for i in test:
            margin = rows[i]["logits"][1] - rows[i]["logits"][0]
            baseline[i] = _sigmoid(margin / _temperature(fitted, rows[i]["tokens"], long_threshold))
        for strength, probabilities in candidates.items():
            scale, bias = _fit_affine(train, strength)
            for i in test:
                margin = rows[i]["logits"][1] - rows[i]["logits"][0]
                probabilities[i] = _sigmoid(scale * margin + bias)
    reference = _metrics(rows, baseline)
    measured = {strength: _metrics(rows, p) for strength, p in candidates.items()}
    accepted = [
        strength
        for strength, metrics in measured.items()
        if metrics["nll"] < reference["nll"] - 1e-4
        and metrics["brier"] <= reference["brier"] + 1e-8
        and metrics["credit"] >= reference["credit"] - 1e-8
        and metrics["abstentions"] < reference["abstentions"]
    ]
    result.report.update(
        status="not_selected",
        out_of_fold_baseline=reference,
        out_of_fold_candidates={str(k): v for k, v in measured.items()},
    )
    if accepted:
        strength = min(accepted, key=lambda k: (measured[k]["nll"], -k))
        result.scale, result.bias = _fit_affine(rows, strength)
        result.report.update(status="selected", regularization=strength)
    return result
