"""Compare emitted decision policies on saved dev probabilities without fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .continuation_audit import changes, checked_rows, match_rows
from .v2 import paired_report, summarize, typed_row


def commit_noul(rows):
    """Apply the predeclared .199/.801 rule and recompute emitted metrics."""
    emitted = []
    for row in rows:
        probs = list(row["probs"])
        if row["type"] == "noul" and 0.2 < probs[1] < 0.8:
            yes = 0.801 if probs[1] >= 0.5 else 0.199
            probs = [1 - yes, yes]
        spec = SimpleNamespace(type=row["type"], ordinals=row.get("ordinals"))
        prediction = max(range(len(probs)), key=probs.__getitem__)
        emitted.append(
            {
                **row,
                **typed_row(spec, probs, row["target"]),
                "probs": probs,
                "argmax_credit": row["target"][prediction],
            }
        )
    return emitted


def probe_policies(raw, calibrated, calibration, *, replicates=2000):
    if type(replicates) is not int or replicates < 1:
        raise ValueError("bootstrap replicate count must be positive")
    if raw.get("calibration") is not None:
        raise ValueError("raw arm must not already declare a fitted calibration")
    if calibrated.get("calibration") != "reserved_split_scoped_temperature":
        raise ValueError("calibrated arm needs reserved-split calibration provenance")
    model_id = raw.get("model_id")
    if model_id != calibrated.get("model_id") or model_id != calibration.get("model_id"):
        raise ValueError("all policy inputs must use the exact same checkpoint")
    before, fitted = checked_rows(raw), checked_rows(calibrated)
    reserved = checked_rows(calibration, split="calibration")
    match_rows(before, fitted)
    if not raw.get("cohort_sha256") or raw["cohort_sha256"] != calibrated.get("cohort_sha256"):
        raise ValueError("raw and calibrated dev cohort fingerprints differ")
    for field in ("id", "cluster_id"):
        if {row[field] for row in before} & {row[field] for row in reserved}:
            raise ValueError(f"calibration/dev {field} overlap; no policy selection is valid")
    arms = {
        "raw": before,
        "raw_commit": commit_noul(before),
        "calibrated": fitted,
        "calibrated_commit": commit_noul(fitted),
    }
    comparisons = {}
    for baseline, candidate in (
        ("raw", "raw_commit"),
        ("raw", "calibrated"),
        ("calibrated", "calibrated_commit"),
    ):
        comparisons[f"{baseline}->{candidate}"] = {
            **changes(arms[baseline], arms[candidate]),
            "paired": paired_report(arms[baseline], arms[candidate], replicates),
        }
    return {
        "version": 1,
        "scope": "exploratory policies on existing dev reads; not independent confirmation",
        "model_id": model_id,
        "cohort_sha256": raw["cohort_sha256"],
        "n": len(before),
        "independent_cases": len({row["cluster_id"] for row in before}),
        "calibration_n": len(reserved),
        "calibration_independent_cases": len({row["cluster_id"] for row in reserved}),
        "execution": {"model_forwards": 0, "fitting_steps": 0, "new_gpu_seconds": 0},
        "test_opened": False,
        "selected_policy": None,
        "commit_rule": {"lower": 0.199, "upper": 0.801, "tie_at_half": "yes"},
        "arms": {name: summarize(rows) for name, rows in arms.items()},
        "comparisons": comparisons,
        "limitations": [
            "existing ayaka checkpoint reads do not measure a new frozen letter-reader system",
            "saved probabilities cannot recover saturated logits",
            "report tags and disjoint rows do not cryptographically prove which inputs fitted temperatures",
            "input metadata does not independently prove full prompt/candidate-string identity",
            "stored latency is prior inference timing; postprocessing latency was not measured",
            "policy development already examined dev; no promotion or sealed-inclusive claim",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "calibrated", "calibration", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError("probe requires a fresh output receipt")
    inputs = {
        name: getattr(args, name).read_bytes() for name in ("raw", "calibrated", "calibration")
    }
    result = probe_policies(
        **{name: json.loads(data) for name, data in inputs.items()}, replicates=args.replicates
    )
    result["input_sha256"] = {
        name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "n": result["n"], **result["execution"]}))
    return result


if __name__ == "__main__":
    main()
