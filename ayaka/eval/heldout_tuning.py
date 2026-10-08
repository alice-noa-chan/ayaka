"""Fit reasoned-path temperatures and the auto router on held-out tuning reads, then read dev.

Input is the result folder of one held-out comparison job (``results/`` with
``cohorts/``, ``protocols/`` and one ``<split>/<system>.jsonl`` per run). The
order of use keeps each split to one role:

1. ``calibration``: path temperatures (``PathCalibration.fit``), nothing else.
2. ``router_train``: router regression, and the choice between routing
   policies (highest equal-type CC on router_train).
3. ``dev``: ``fit_router``'s own lambda/promotion validation, then a single
   scored read of every policy against v1 and against calibrated v2 direct.

``adapter="merged"`` tunes v2 reads collected with the LoRA folded into the
weights (``<system>-merged.jsonl``); v1 is always read unmerged. Merged reads
generate different traces, so temperatures, the router and the policy are
refitted from them rather than reused.

Nothing is trained on gradients and the private test split is never opened.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from ..data.schema import Sample
from ..routing import fit_router, paired_training_rows
from ..training.path_calibration import PathCalibration
from .checkpoint_comparison import ADAPTERS, checked_rows, read_rows
from .jevbench import speed_axis
from .quality_hierarchy import _comparison
from .v2 import summarize, typed_row

VERSION = "ayaka-heldout-tuning-1"
POLICIES = ("direct", "router", "noul_always", "noul_always+router")


class _RowSpec:
    """The two spec fields typed_row reads, taken from a checked comparison row."""

    def __init__(self, row):
        self.type, self.ordinals = row["type"], row.get("ordinals")


def load_run(results, split, system, noul_order="false_first", adapter="unmerged"):
    samples = [
        Sample.from_json(json.loads(line))
        for line in (results / "cohorts" / f"{split}.jsonl").read_bytes().splitlines()
        if line.strip()
    ]
    protocol = json.loads((results / "protocols" / f"{split}.json").read_bytes())
    suffix = "" if noul_order == "false_first" else f"-{noul_order}"
    suffix += "" if adapter == "unmerged" else f"-{adapter}"
    rows = read_rows(results / split / f"{system}{suffix}.jsonl")
    return checked_rows(rows, samples, protocol, system, adapter=adapter)


def calibrated(rows, calibration):
    """Apply path temperatures and rescore each row with its new probabilities."""
    out = []
    for row in rows:
        probs = calibration.apply(row["probs"], row["type"], row["route"], row["budget"])
        out.append({**row, "probs": probs, **typed_row(_RowSpec(row), probs, row["target"])})
    return out


def compose(direct, reasoned, use):
    """Per question, keep the direct read or switch to the reasoned read.

    A routed request always pays for the direct read first (the router reads its
    features), so a switched row's latency is the sum of both reads.
    """
    by_id = {row["id"]: row for row in reasoned}
    if {row["id"] for row in direct} != set(by_id):
        raise ValueError("direct and reasoned reads must cover the same questions")
    return [
        {**by_id[row["id"]], "latency_s": row["latency_s"] + by_id[row["id"]]["latency_s"]}
        if use(row)
        else row
        for row in sorted(direct, key=lambda r: r["id"])
    ]


def latency_summary(rows):
    values = sorted(row["latency_s"] for row in rows)
    p50 = values[len(values) // 2]
    p95 = values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)]
    return {
        "mean_s": sum(values) / len(values),
        "p50_s": p50,
        "p95_s": p95,
        "speed_axis": speed_axis(p50, p95),
    }


def brief(rows):
    summary = summarize(rows)
    return {
        "cc_equal_types": summary["cc_equal_types"],
        "by_type": {
            kind: {
                key: value[key]
                for key in ("n", "cc", "nll", "ece", "brier", "abstentions")
                if key in value
            }
            for kind, value in summary["by_type"].items()
        },
        "latency": latency_summary(rows),
    }


def gate(before, after, *, major_gain, replicates):
    result = _comparison(before, after, major_gain=major_gain, replicates=replicates)
    return {
        "cc_delta": summarize(after)["cc_equal_types"] - summarize(before)["cc_equal_types"],
        "cc_delta_95ci": result["paired"]["cc_delta_95ci"],
        "screen_passed": result["screen_passed"],
        "failed_checks": [name for name, ok in result["checks"].items() if not ok],
    }


def policy_rule(name, router):
    """Return a row predicate that says whether a policy reasons on that question."""

    def routed(row):
        gain, tokens = router.predict(row["routing_features"][:-1] + [384 / 1024])
        return gain - router.penalty * tokens > 0

    rules = {
        "direct": lambda row: False,
        "router": routed,
        "noul_always": lambda row: row["type"] == "noul",
        "noul_always+router": lambda row: row["type"] == "noul" or routed(row),
    }
    return rules[name]


def tune(results, *, replicates=2000, adapter="unmerged"):
    results = Path(results)
    reads = {
        (split, system): load_run(results, split, system, adapter=adapter)
        for split in ("calibration", "router_train", "dev")
        for system in ("v2_off", "v2_on")
    }
    v1 = load_run(results, "dev", "v1_on")
    calibration = PathCalibration.fit(
        reads["calibration", "v2_off"] + reads["calibration", "v2_on"]
    )
    tuned = {
        key: calibrated(rows, calibration) for key, rows in reads.items() if key[0] != "calibration"
    }
    router = fit_router(
        paired_training_rows(
            tuned["router_train", "v2_off"], tuned["router_train", "v2_on"], "router_train"
        ),
        paired_training_rows(tuned["dev", "v2_off"], tuned["dev", "v2_on"], "dev"),
    )
    policies = {}
    for name in POLICIES:
        rule = policy_rule(name, router)
        policies[name] = {}
        for split in ("router_train", "dev"):
            rows = compose(tuned[split, "v2_off"], tuned[split, "v2_on"], rule)
            policies[name][split] = {
                **brief(rows),
                "reasoned_by_type": dict(
                    Counter(r["type"] for r in rows if r["route"] != "direct")
                ),
            }
            if split == "dev":
                policies[name]["rows"] = rows
    chosen = max(POLICIES, key=lambda n: policies[n]["router_train"]["cc_equal_types"])
    dev_gates = {
        name: {
            "over_v1_on": gate(v1, value["rows"], major_gain=True, replicates=replicates),
            "over_v2_off_calibrated": gate(
                tuned["dev", "v2_off"], value["rows"], major_gain=False, replicates=replicates
            ),
        }
        for name, value in policies.items()
        if name != "direct"
    }
    for value in policies.values():
        del value["rows"]
    return {
        "version": VERSION,
        "v2_adapter": adapter,
        "path_temperatures": calibration.temperatures,
        "router": {
            "promoted": router.promoted,
            "lambda": router.penalty,
            "validation": router.validation,
            "note": "lambda and promotion use dev pairs, as fit_router is designed",
        },
        "systems_dev": {
            "v1_on": brief(v1),
            "v2_off": brief(reads["dev", "v2_off"]),
            "v2_on": brief(reads["dev", "v2_on"]),
            "v2_off_calibrated": brief(tuned["dev", "v2_off"]),
            "v2_on_calibrated": brief(tuned["dev", "v2_on"]),
        },
        "policies": policies,
        "policy_chosen_on_router_train": chosen,
        "dev_gates": dev_gates,
        "optimizer_updates": 0,
        "original_private_test_opened": False,
        "promotable": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="retrieved results/ folder")
    parser.add_argument("--out", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--adapter", choices=ADAPTERS, default="unmerged")
    args = parser.parse_args(argv)
    report = tune(args.results, replicates=args.replicates, adapter=args.adapter)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "chosen": report["policy_chosen_on_router_train"],
                "router_promoted": report["router"]["promoted"],
                "dev_cc": {
                    name: round(value["dev"]["cc_equal_types"], 2)
                    for name, value in report["policies"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
