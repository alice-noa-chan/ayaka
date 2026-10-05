"""Predeclared paired comparison: v2 reasoning off (Swift) against v1 with its reasoning route.

Protocol: docs/experiments/V1ON_V2OFF_PROTOCOL_2026-10-05.md. The rule below is fixed
before any v1 row on this cohort exists.

Systems, all scored by the same Swift scorer on the same ids:
- v2_off: frozen 12B Swift `min` reads, scored with the saved production policy and its
  reasoning route removed (zero generated tokens).
- v1_on: published ayaka-large with the frozen worked-steps route. These are its final
  probabilities, unmodified.
- v1_off: v1's own single pass (diagnostic).

Success for "v2 off beats v1 on" (from V2_DIRECT_OVER_V1_REASONING_2026-10-04.md), all on
the full cohort:
1. thresholded Choice/Noul gold credit +5 points or more
2. equal-type chance-corrected score (mean of per-type CC) +5 or more
3. case-cluster bootstrap 95% CI lower bound of (2) > 0
4. Score normalized RPS not worse

Speed and Cost are not compared: v1 ran on HF eager and v2 on vLLM.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.swift.collect import load_reads  # noqa: E402
from ayaka.swift.evaluate import percentile  # noqa: E402
from ayaka.swift.policy import Policy  # noqa: E402
from ayaka.swift.score import (  # noqa: E402
    TYPES,
    calibration,
    intelligence,
    normalized_rps,
    noul_labels,
    prepare_rows,
)

B, SEED = 2000, 15
SUCCESS = {"credit_pp": 5.0, "typed_cc": 5.0, "typed_cc_ci_lower": 0.0}
IDENTITY = Policy(noul_commit=False)


def credit(item) -> bool:
    if item.type == "noul":
        false, true = noul_labels(item.probs)
        p = item.probs[true]
        prediction = false if p <= 0.2 else true if p >= 0.8 else None
        return prediction == item.gold
    return max(item.labels, key=item.probs.get) == item.gold


def metrics(items) -> dict:
    report = intelligence(items)
    per_type = report["per_type_CC"]
    scored = [i for i in items if i.type in ("choice", "noul")]
    rps = [normalized_rps(i) for i in items if i.type == "score"]
    return {
        "n": len(items),
        "I": report["I"],
        "per_type_CC": per_type,
        "typed_cc": mean(per_type.values()) if len(per_type) == len(TYPES) else None,
        "credit": 100 * mean(map(credit, scored)) if scored else None,
        "score_rps": mean(rps) if rps else None,
        "C": calibration(items)["C"] if any(i.type == "choice" for i in items) else None,
    }


def as_baseline(rows):
    return [{**row, "raw_probs": row["baseline_probs"]} for row in rows]


def compare(v2_rows, v1_rows, policy: Policy) -> dict:
    v1 = {row["id"]: row for row in v1_rows}
    if sorted(v1) != sorted(row["id"] for row in v2_rows) or len(v1) != len(v1_rows):
        raise ValueError("v1 and v2 rows must cover exactly the same ids once")
    v2_rows = sorted(v2_rows, key=lambda r: r["id"])
    v1_rows = [v1[row["id"]] for row in v2_rows]
    for a, b in zip(v2_rows, v1_rows, strict=True):
        if a["labels"] != b["labels"] or a["type"] != b["type"] or a["gold"] != b["gold"]:
            raise ValueError(f"{a['id']}: label order, type or gold differs between systems")
    systems = {
        "v2_off": prepare_rows(v2_rows, policy),
        "v1_on": prepare_rows(v1_rows, IDENTITY),
        "v1_off": prepare_rows(as_baseline(v1_rows), IDENTITY),
    }
    clusters: dict[str, list[int]] = {}
    for index, row in enumerate(v2_rows):
        clusters.setdefault(row["cluster_id"], []).append(index)
    keys = sorted(clusters)

    def slice_report(indices):
        point = {name: metrics([items[i] for i in indices]) for name, items in systems.items()}
        return point

    def deltas(indices):
        a = metrics([systems["v2_off"][i] for i in indices])
        b = metrics([systems["v1_on"][i] for i in indices])
        return {
            key: (a[key] - b[key]) if a[key] is not None and b[key] is not None else None
            for key in ("typed_cc", "I", "credit")
        }

    everything = list(range(len(v2_rows)))
    rng = random.Random(SEED)
    draws = {"typed_cc": [], "I": [], "credit": []}
    for _ in range(B):
        sample = [i for _ in keys for i in clusters[rng.choice(keys)]]
        d = deltas(sample)
        for key, value in d.items():
            if value is not None:
                draws[key].append(value)
    point = slice_report(everything)
    delta = deltas(everything)
    ci = {k: [percentile(v, 0.025), percentile(v, 0.975)] for k, v in draws.items() if v}
    rps_ok = point["v2_off"]["score_rps"] <= point["v1_on"]["score_rps"]
    checks = {
        "credit_plus_5pp": delta["credit"] >= SUCCESS["credit_pp"],
        "typed_cc_plus_5": delta["typed_cc"] >= SUCCESS["typed_cc"],
        "typed_cc_ci_lower_above_0": ci["typed_cc"][0] > SUCCESS["typed_cc_ci_lower"],
        "score_rps_not_worse": rps_ok,
    }
    by_source = {}
    for source in sorted({row["source"] for row in v2_rows}):
        idx = [i for i, row in enumerate(v2_rows) if row["source"] == source]
        by_source[source] = {
            name: {
                "n": len(idx),
                "credit": metrics([items[i] for i in idx])["credit"],
                "per_type_CC": metrics([items[i] for i in idx])["per_type_CC"],
            }
            for name, items in systems.items()
        }
    routes = {}
    for row in v1_rows:
        routes[row["route"]] = routes.get(row["route"], 0) + 1
    return {
        "systems": point,
        "delta_v2off_minus_v1on": delta,
        "ci_95": ci,
        "bootstrap": {"B": B, "seed": SEED, "unit": "cluster", "clusters": len(keys)},
        "success_rule": SUCCESS,
        "checks": checks,
        "v2_off_beats_v1_on": all(checks.values()),
        "by_source": by_source,
        "v1_routes": routes,
        "v1_route_errors": sum(1 for row in v1_rows if row.get("route_error")),
        "comparison_scope": "Intelligence/credit/calibration only; Speed and Cost not compared",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-reads", nargs="+", required=True, type=Path)
    parser.add_argument("--v1-rows", nargs="+", required=True, type=Path)
    parser.add_argument("--policy", type=Path, required=True, help="saved production policy.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    policy = replace(Policy.load(args.policy), reasoning_route=None)
    v2 = [r for r in load_reads(args.v2_reads) if r["readout"] != "grouped_approx"]
    if any(r["public"] for r in v2) or v2[0]["prompt_variant"] != "min":
        raise SystemExit("v2_off must be non-public Swift min reads")
    v1 = load_reads(args.v1_rows)
    report = compare(v2, v1, policy)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    s, d, c = report["systems"], report["delta_v2off_minus_v1on"], report["ci_95"]
    for name in s:
        print(
            f"{name:7s} I {s[name]['I']:.2f} typedCC {s[name]['typed_cc']:.2f} credit {s[name]['credit']:.2f}"
        )
    print(f"delta typedCC {d['typed_cc']:+.2f} CI {c['typed_cc']} credit {d['credit']:+.2f}")
    print("v2_off_beats_v1_on:", report["v2_off_beats_v1_on"], report["checks"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
