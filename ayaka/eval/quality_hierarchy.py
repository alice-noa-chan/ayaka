"""Versioned dev screen for v1-on < v2-off < v2-on; no execution attestation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

from ..data.schema import Sample
from ..primitives import QuestionSpec
from ..reasoning import EFFORT_TOKENS
from ..training.batching import _noul_canonical
from .pretraining_v2 import cohort_fingerprint
from .v2 import TIERS, paired_report, summarize, typed_row

VERSION = "ayaka-quality-hierarchy-dev-1"
TYPES = ("choice", "noul", "score")
RULE = {
    "v2_off_cc_gain": 5.0,
    "v2_off_classification_credit_gain_pp": 5.0,
    "v2_on_cc_gain_strictly_above": 0.0,
    "paired_cc_ci_lower_strictly_above": 0.0,
    "minimum_independent_cases": 200,
    "regression_tolerance": 1e-8,
}


def _model_id(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("model identity must be an externally supplied checkpoint SHA-256")
    return value


def _canonical_questions(samples):
    expected = {}
    for sample in samples:
        metadata = sample.metadata
        if metadata.get("split") != "dev":
            raise ValueError("hierarchy screening requires dev samples; test stays unopened")
        for original in sample.questions:
            q = _noul_canonical(original)
            identifier = metadata["source_example_id"] + "/" + q.id
            if identifier in expected:
                raise ValueError("prepared cohort repeats question identities")
            ordinals = [c.ordinal for c in q.candidates] if q.type == "score" else None
            spec = QuestionSpec(
                q.type, q.instruction, [c.description for c in q.candidates], ordinals
            )
            expected[identifier] = (
                spec,
                {
                    "id": identifier,
                    "type": q.type,
                    "target": [q.target_distribution.get(c.id, 0) for c in q.candidates],
                    "candidate_ids": [c.id for c in q.candidates],
                    "ordinals": ordinals,
                    "cluster_id": metadata["source_lineage"],
                    "split": "dev",
                    "language": metadata["language"],
                    "family": metadata["task_family"],
                    "source": metadata.get("source", metadata["task_family"]),
                    "tier": metadata.get("tier", "standard"),
                    "modality": metadata.get("modality", "text"),
                    "partition": "generated_finite"
                    if "proposal_supervision" in metadata
                    else "fixed",
                },
            )
    if not expected or {spec.type for spec, _ in expected.values()} != set(TYPES):
        raise ValueError("hierarchy screening requires all three primitive types")
    if any(
        not values["cluster_id"] or values["tier"] not in TIERS for _, values in expected.values()
    ):
        raise ValueError("prepared questions require case identities and supported tiers")
    return expected


def _checked_rows(report, expected, cohort_sha256, mode, model_id):
    if (
        report.get("complete") is not True
        or report.get("split") != "dev"
        or report.get("cohort_sha256") != cohort_sha256
    ):
        raise ValueError("screening requires complete dev reports bound to the prepared cohort")
    rows = report.get("rows", {}).get(mode, [])
    identifiers = [row.get("id") for row in rows]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(expected):
        raise ValueError("each system must cover the prepared cohort exactly once")
    checked = []
    for original in sorted(rows, key=lambda row: row["id"]):
        spec, fields = expected[original["id"]]
        if original.get("model_id") != model_id:
            raise ValueError("row checkpoint identity differs from its external anchor")
        for key, value in fields.items():
            if key not in original or original[key] != value:
                raise ValueError(f"{original['id']}: canonical field differs: {key}")
        budget = 0 if mode == "off" else EFFORT_TOKENS[mode]
        tokens = original.get("reasoning_tokens")
        if (
            type(original.get("budget")) is not int
            or original["budget"] != budget
            or type(tokens) is not int
            or not 0 <= tokens <= budget
            or (mode == "off" and original.get("route") != "direct")
            or (mode != "off" and original.get("route") not in {"reasoned", "fallback"})
            or (
                original.get("route") == "reasoned"
                and (
                    tokens == 0
                    or original.get("finish_reason") not in {"eos", "length"}
                    or original.get("error") is not None
                )
            )
        ):
            raise ValueError("row reasoning controls or usage differ from the selected mode")
        probabilities = original.get("probs", [])
        if len(probabilities) != len(fields["candidate_ids"]):
            raise ValueError("probabilities differ from the canonical candidate count")
        # Ignore cached metrics: the canonical target and order decide every score.
        metrics = typed_row(spec, probabilities, fields["target"], logits=original.get("logits"))
        checked.append(
            {**fields, **metrics, "route": original["route"], "reasoning_tokens": tokens}
        )
    return checked


def _credit(rows):
    return 100 * mean(row["correct"] for row in rows if row["type"] != "score")


def _comparison(before, after, *, major_gain, replicates):
    old, new = summarize(before), summarize(after)
    paired = paired_report(before, after, replicates)
    delta = new["cc_equal_types"] - old["cc_equal_types"]
    tolerance = RULE["regression_tolerance"]
    checks = {
        "cc_gain": delta >= RULE["v2_off_cc_gain"]
        if major_gain
        else delta > RULE["v2_on_cc_gain_strictly_above"],
        "classification_credit_gain": _credit(after) - _credit(before)
        >= (RULE["v2_off_classification_credit_gain_pp"] if major_gain else -tolerance),
        "paired_cc_ci_lower_above_zero": paired["cc_delta_95ci"][0]
        > RULE["paired_cc_ci_lower_strictly_above"],
        "enough_independent_cases": paired["independent_cases"]
        >= RULE["minimum_independent_cases"],
    }
    for kind in TYPES:
        a, b = old["by_type"][kind], new["by_type"][kind]
        checks[f"{kind}/cc_not_worse"] = b["cc"] >= a["cc"] - tolerance
        for metric in ("nll", "brier", *(("rps",) if kind == "score" else ())):
            checks[f"{kind}/{metric}_not_worse"] = (
                math.isfinite(b[metric]) and b[metric] <= a[metric] + tolerance
            )
    scores_before = [row["nmae"] for row in before if row["type"] == "score"]
    scores_after = [row["nmae"] for row in after if row["type"] == "score"]
    checks["score/nmae_not_worse"] = mean(scores_after) <= mean(scores_before) + tolerance
    groups = {}
    for field in ("source", "language"):
        groups[field] = {}
        for value in sorted({row[field] for row in before}):
            left = summarize([row for row in before if row[field] == value])
            right = summarize([row for row in after if row[field] == value])
            groups[field][value] = {"before": left, "after": right}
            for kind in left["by_type"]:
                checks[f"{field}:{value}/{kind}/cc_not_worse"] = (
                    right["by_type"][kind]["cc"] >= left["by_type"][kind]["cc"] - tolerance
                )
    return {
        "screen_passed": all(checks.values()),
        "checks": checks,
        "before": old,
        "after": new,
        "cc_delta": delta,
        "classification_credit_delta_pp": _credit(after) - _credit(before),
        "score_nmae": {"before": mean(scores_before), "after": mean(scores_after)},
        "paired": paired,
        "groups": groups,
    }


def hierarchy_screen(
    v1_report,
    v2_report,
    samples,
    *,
    v1_model_id,
    v2_model_id,
    v1_mode="high",
    v2_mode="high",
    replicates=2000,
):
    """Recompute three paired systems against canonical dev inputs and fixed rules."""
    if v1_mode not in EFFORT_TOKENS or v2_mode not in EFFORT_TOKENS:
        raise ValueError("select a forced low/medium/high reasoning mode for each model")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("bootstrap repetitions must be a positive integer")
    v1_model_id, v2_model_id = _model_id(v1_model_id), _model_id(v2_model_id)
    if v1_model_id == v2_model_id:
        raise ValueError("v1 and v2 must have distinct checkpoint identities")
    samples = list(samples)
    expected, cohort = _canonical_questions(samples), cohort_fingerprint(samples)
    systems = {
        "v1_on": _checked_rows(v1_report, expected, cohort, v1_mode, v1_model_id),
        "v2_off": _checked_rows(v2_report, expected, cohort, "off", v2_model_id),
        "v2_on": _checked_rows(v2_report, expected, cohort, v2_mode, v2_model_id),
    }
    comparisons = {
        "v2_off_over_v1_on": _comparison(
            systems["v1_on"], systems["v2_off"], major_gain=True, replicates=replicates
        ),
        "v2_on_over_v2_off": _comparison(
            systems["v2_off"], systems["v2_on"], major_gain=False, replicates=replicates
        ),
    }
    reasoned_count = sum(row["route"] == "reasoned" for row in systems["v2_on"])
    return {
        "version": VERSION,
        "rule": dict(RULE),
        "screen_passed": reasoned_count > 0
        and all(c["screen_passed"] for c in comparisons.values()),
        "comparisons": comparisons,
        "cohort_sha256": cohort,
        "model_ids": {"v1": v1_model_id, "v2": v2_model_id},
        "modes": {"v1_on": v1_mode, "v2_on": v2_mode},
        "routes": {
            name: dict(Counter(row["route"] for row in rows)) for name, rows in systems.items()
        },
        "v2_reasoning_completed": reasoned_count > 0,
        "scope": "canonical paired development screening; no execution provenance or fresh independence attestation",
        "final_test_required": True,
        "promotable": False,
        "official_composite": None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v1-report", "v2-report", "samples", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("v1", "v2"):
        parser.add_argument(f"--{name}-model-id", required=True)
        parser.add_argument(f"--{name}-mode", choices=tuple(EFFORT_TOKENS), default="high")
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError("hierarchy reports are written once")
    data = {
        name: getattr(args, name).read_bytes() for name in ("v1_report", "v2_report", "samples")
    }
    samples = [
        Sample.from_json(json.loads(line)) for line in data["samples"].splitlines() if line.strip()
    ]
    report = hierarchy_screen(
        json.loads(data["v1_report"]),
        json.loads(data["v2_report"]),
        samples,
        v1_model_id=args.v1_model_id,
        v2_model_id=args.v2_model_id,
        v1_mode=args.v1_mode,
        v2_mode=args.v2_mode,
        replicates=args.replicates,
    )
    report["input_sha256"] = {
        name: hashlib.sha256(value).hexdigest() for name, value in data.items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        stream.write(
            (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
        )
    print(json.dumps({"out": str(args.out), "screen_passed": report["screen_passed"]}))
    return 0 if report["screen_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
