"""Fit reasoned-path temperatures and the auto router on held-out tuning reads, then read dev.

Input is the result folder of one held-out comparison job (``results/`` with
``cohorts/``, ``protocols/`` and one ``<split>/<system>.jsonl`` per run). The
order of use keeps each split to one role:

1. ``calibration``: path temperatures (``PathCalibration.fit``), nothing else.
2. ``router_train``: router regression, and the choice between routing
   policies (highest equal-type CC on router_train).
3. ``dev``: ``fit_router``'s own lambda/promotion validation, then a single
   scored read of every policy against v1 and against calibrated v2 direct.

``--extension`` adds the calibration and router_train reads of further cohorts
of the same splits (read under their own protocols) to the tuning splits; dev
always comes from ``--results`` alone.

``adapter="merged"`` tunes v2 reads collected with the LoRA folded into the
weights (``<system>-merged.jsonl``); v1 is always read unmerged. Merged reads
generate different traces, so temperatures, the router and the policy are
refitted from them rather than reused.

The ``levers`` section adds the levers predeclared in
``docs/experiments/V2_GATE_BLOCKERS_PREDECLARATION_2026-10-09.md``. Both reuse
reads a routed request already pays for, so they cost no extra generation:

- a Noul log-linear pool of the direct and reasoned reads, fitted on
  calibration under a no-CC-loss constraint (``fit_noul_pool``);
- a Score direct temperature chosen for CC within an NLL tolerance
  (``fit_cc_temperature``).

The lever policy is chosen on router_train, then gated once on dev. The
numbers in the other sections do not change.

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

VERSION = "ayaka-heldout-tuning-2"
POLICIES = ("direct", "router", "noul_always", "noul_always+router")
LEVER_POLICIES = ("noul_always+router", "noul+choice_always+router")
POOL_GRID = tuple(i / 20 for i in range(31))
TEMPERATURE_GRID = tuple(round(0.5 + 0.05 * i, 2) for i in range(21))
SCORE_NLL_TOLERANCE = 0.05


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


def _softmax(logits):
    top = max(logits)
    exp = [math.exp(x - top) for x in logits]
    return [x / sum(exp) for x in exp]


def _logs(probs):
    return [math.log(max(p, 1e-12)) for p in probs]


def rescored(row, probs):
    return {**row, "probs": probs, **typed_row(_RowSpec(row), probs, row["target"])}


def pool(direct_probs, reasoned_probs, weights):
    """Log-linear pool ``softmax(a * log p_direct + b * log p_reasoned)``."""
    a, b = weights
    return _softmax(
        [a * x + b * y for x, y in zip(_logs(direct_probs), _logs(reasoned_probs), strict=True)]
    )


def pooled(rows, direct, reasoned, weights, kind="noul"):
    """Replace each ``kind`` row by the pool of its raw direct and raw reasoned reads.

    ``rows`` are the (possibly tempered) reasoned rows being served; ``direct``
    and ``reasoned`` are the raw reads the pool is computed from, so the path
    temperature of a pooled row is replaced, not compounded.
    """
    direct = {row["id"]: row["probs"] for row in direct}
    reasoned = {row["id"]: row["probs"] for row in reasoned}
    return [
        rescored(row, pool(direct[row["id"]], reasoned[row["id"]], weights))
        if row["type"] == kind
        else row
        for row in rows
    ]


def _type_summary(rows, kind):
    return summarize([row for row in rows if row["type"] == kind])["by_type"][kind]


def fit_noul_pool(direct, reasoned, baseline, *, grid=POOL_GRID):
    """Fit Noul pool weights on calibration: the lowest NLL with no loss of CC.

    ``direct`` and ``reasoned`` are the raw calibration reads. ``baseline`` is
    the temperature-calibrated reasoned read, and the pool must not fall below
    its Noul CC: without that floor, the NLL-optimal pool softens reads into
    the abstention band. Returns ``(a, b)``, or ``(0, 1)`` (the raw reasoned
    read) when no grid point qualifies.
    """
    if any(row.get("split") != "calibration" for row in direct + reasoned):
        raise ValueError("the Noul pool is fitted on calibration reads only")
    noul = [row for row in reasoned if row["type"] == "noul"]
    floor = _type_summary(baseline, "noul")["cc"] - 1e-8
    best, weights = math.inf, (0.0, 1.0)
    for a in grid:
        for b in grid:
            if b == 0:
                continue
            stats = _type_summary(pooled(noul, direct, noul, (a, b)), "noul")
            if stats["cc"] >= floor and stats["nll"] < best:
                best, weights = stats["nll"], (a, b)
    return weights


def temper(row, raw_probs, temperature):
    return rescored(row, _softmax([x / temperature for x in _logs(raw_probs)]))


def fit_cc_temperature(rows, kind, *, tolerance, grid=TEMPERATURE_GRID):
    """Return the highest-CC temperature whose NLL is within ``tolerance`` of the grid minimum."""
    if any(row.get("split") != "calibration" for row in rows):
        raise ValueError("lever temperatures are fitted on calibration reads only")
    subset = [row for row in rows if row["type"] == kind]
    stats = {t: _type_summary([temper(row, row["probs"], t) for row in subset], kind) for t in grid}
    limit = min(s["nll"] for s in stats.values()) * (1 + tolerance)
    return max((t for t in grid if stats[t]["nll"] <= limit), key=lambda t: stats[t]["cc"])


def tempered(rows, raw, kind, temperature):
    """Re-temper ``kind`` rows from their raw read; rows of other types are kept as they are."""
    raw = {row["id"]: row["probs"] for row in raw}
    return [
        temper(row, raw[row["id"]], temperature) if row["type"] == kind else row for row in rows
    ]


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
        "noul+choice_always+router": lambda row: row["type"] in ("noul", "choice") or routed(row),
    }
    return rules[name]


def tuning_reads(results, extensions, split, system, adapter):
    """One tuning split's reads: the main cohort followed by each extension cohort.

    An extension is a separate cohort of the same split, read under its own
    protocol, so every part is checked on its own before the parts are joined.
    """
    rows = load_run(results, split, system, adapter=adapter)
    for extension in extensions:
        rows = rows + load_run(extension, split, system, adapter=adapter)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{split}: extension cohorts repeat a question of another part")
    return rows


def tune(results, *, replicates=2000, adapter="unmerged", extensions=()):
    results = Path(results)
    extensions = [Path(path) for path in extensions]
    reads = {
        (split, system): tuning_reads(results, extensions, split, system, adapter)
        if split != "dev"
        else load_run(results, split, system, adapter=adapter)
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
    levers = lever_section(reads, calibration, tuned, router, v1, replicates=replicates)
    return {
        "version": VERSION,
        "v2_adapter": adapter,
        "tuning_questions": {
            split: len(reads[split, "v2_off"]) for split in ("calibration", "router_train")
        },
        "extension_results": len(extensions),
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
        "levers": levers,
        "optimizer_updates": 0,
        "original_private_test_opened": False,
        "promotable": False,
    }


def lever_section(reads, calibration, tuned, router, v1, *, replicates):
    """Fit the predeclared levers on calibration, pick a policy on router_train, gate it on dev."""
    weights = fit_noul_pool(
        reads["calibration", "v2_off"],
        reads["calibration", "v2_on"],
        calibrated(reads["calibration", "v2_on"], calibration),
    )
    score_t = fit_cc_temperature(
        reads["calibration", "v2_off"], "score", tolerance=SCORE_NLL_TOLERANCE
    )
    direct, reasoned = {}, {}
    for split in ("router_train", "dev"):
        off, on = reads[split, "v2_off"], reads[split, "v2_on"]
        direct[split] = tempered(tuned[split, "v2_off"], off, "score", score_t)
        reasoned[split] = pooled(tuned[split, "v2_on"], off, on, weights)
    policies = {
        name: {
            split: compose(direct[split], reasoned[split], policy_rule(name, router))
            for split in ("router_train", "dev")
        }
        for name in LEVER_POLICIES
    }
    chosen = max(
        LEVER_POLICIES, key=lambda n: summarize(policies[n]["router_train"])["cc_equal_types"]
    )
    return {
        "predeclaration": "docs/experiments/V2_GATE_BLOCKERS_PREDECLARATION_2026-10-09.md",
        "noul_pool_weights": {"direct": weights[0], "reasoned": weights[1]},
        "score_direct_temperature": score_t,
        "score_nll_tolerance": SCORE_NLL_TOLERANCE,
        "policies": {
            name: {split: brief(rows) for split, rows in value.items()}
            for name, value in policies.items()
        },
        "policy_chosen_on_router_train": chosen,
        "dev_gates": {
            name: {
                "role": "predeclared" if name == chosen else "secondary, not adoptable",
                "over_v1_on": gate(v1, value["dev"], major_gain=True, replicates=replicates),
                "over_v2_off_calibrated": gate(
                    direct["dev"], value["dev"], major_gain=False, replicates=replicates
                ),
            }
            for name, value in policies.items()
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="retrieved results/ folder")
    parser.add_argument("--out", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--adapter", choices=ADAPTERS, default="unmerged")
    parser.add_argument(
        "--extension",
        action="append",
        default=[],
        help="results/ folder of an extension cohort; its calibration and router_train reads are added",
    )
    args = parser.parse_args(argv)
    report = tune(
        args.results, replicates=args.replicates, adapter=args.adapter, extensions=args.extension
    )
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
