"""Evaluate fitted policies, commitment ablations and paired uncertainty."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

from .collect import load_reads
from .policy import Policy
from .score import (
    TYPES,
    calibration,
    composite,
    cost_score,
    decision_cost,
    intelligence,
    prepare_rows,
    score_reads,
    speed_axis,
)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def paired_bootstrap(
    rows: list[dict], baseline: Policy, candidate: Policy, *, B: int = 2000, seed: int = 15
) -> dict:
    """Paired whole-cluster draws; public items are independent singletons."""
    if B < 1:
        raise ValueError("bootstrap repetitions must be positive")
    rows = [row for row in rows if row.get("readout") != "grouped_approx"]
    before = prepare_rows(rows, baseline)
    after = prepare_rows(rows, candidate)
    cases, strata = cluster_strata(rows)
    if set(TYPES) - {item.type for item in before}:
        raise ValueError("bootstrap I requires choice, noul and score items")
    rng = random.Random(seed)
    deltas = []
    for _ in range(B):
        indices = [
            index for stratum in strata.values() for _ in stratum for index in rng.choice(stratum)
        ]
        delta = (
            intelligence([after[i] for i in indices])["I"]
            - intelligence([before[i] for i in indices])["I"]
        )
        deltas.append(delta)
    return {
        "delta_I": intelligence(after)["I"] - intelligence(before)["I"],
        "ci_95": [percentile(deltas, 0.025), percentile(deltas, 0.975)],
        "B": B,
        "seed": seed,
        "unit": "cluster",
        "cluster_count": len(cases),
        "stratification": "primitive_type_coverage",
    }


def cluster_key(row: dict, index: int = 0) -> tuple:
    if row.get("public") or row.get("split") == "public":
        return ("public_singleton", str(row.get("source", "")), str(row.get("id", index)), index)
    metadata = row.get("metadata") or {}
    cluster = (
        row.get("cluster_id") or metadata.get("source_lineage") or metadata.get("case_facts_sha256")
    )
    if cluster is not None:
        return ("cluster", str(cluster))
    case = row.get("case_id") or metadata.get("case_id")
    if case is not None:
        return ("case", str(row.get("source", "")), str(case))
    return ("item", str(row.get("source", "")), str(row.get("id", index)), index)


def cluster_strata(rows: list[dict]) -> tuple[dict, dict]:
    cases = {}
    for index, row in enumerate(rows):
        cases.setdefault(cluster_key(row, index), []).append(index)
    strata = {}
    for indices in cases.values():
        kinds = tuple(sorted({rows[index]["type"] for index in indices}))
        strata.setdefault(kinds, []).append(indices)
    return cases, strata


def case_bootstrap_composite(
    rows: list[dict],
    policy: Policy,
    *,
    speed: float,
    cost: float,
    B: int = 1000,
    seed: int = 15,
    include_grouped: bool = False,
) -> dict:
    """Resample whole cases within strata of primitive-type coverage.

    Use explicit cluster_id/lineage, otherwise case_id, otherwise a singleton.
    Public rows are always singletons. Coverage strata retain
    all three primitives in every draw without splitting multi-question cases.
    The interval is conditional on the already selected policy, not held-out
    performance or an adjustment for searching multiple configurations.
    Each draw uses hard Choice ECE when available, otherwise the same explicit
    all-Choice fallback as the scorer. Report the scope of the supplied reads.
    """
    if B < 1:
        raise ValueError("bootstrap repetitions must be positive")
    if not include_grouped:
        rows = [row for row in rows if row.get("readout") != "grouped_approx"]
    items = prepare_rows(rows, policy)
    if intelligence(items)["missing_types"]:
        raise ValueError("composite bootstrap requires choice, noul and score reads")
    cases, strata = cluster_strata(rows)

    def objective(sample: list) -> float:
        return composite(intelligence(sample)["I"], calibration(sample)["C"], speed, cost)

    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        sample = [
            items[index]
            for stratum in strata.values()
            for _ in stratum
            for index in rng.choice(stratum)
        ]
        draws.append(objective(sample))
    return {
        "composite_A": objective(items),
        "choice_ece_scope": calibration(items)["choice_ece_scope"],
        "ci_95": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "B": B,
        "seed": seed,
        "unit": "case",
        "resampling_unit": "cluster",
        "cluster_count": len(cases),
        "case_count": len(cases),
        "stratification": "primitive_type_coverage",
        "conditional_on_selected_policy": True,
    }


def evaluate(
    rows: list[dict],
    policy: Policy,
    *,
    p50: float | None = None,
    p95: float | None = None,
    usd_in_per_m: float | None = None,
    usd_out_per_m: float | None = None,
    baseline: Policy | None = None,
    B: int = 2000,
    seed: int = 15,
) -> dict:
    nll_temperatures = (policy.search or {}).get("nll_temperatures", {})
    temps = replace(policy, **nll_temperatures, noul_commit=False, commit_margin=None)
    policies = {
        "raw": Policy(commit_margin=None),
        "temps_only": temps,
        "temps_commit": replace(temps, noul_commit=True, commit_margin=0.0),
        "fitted_policy": policy,
    }
    axes = {}
    if (p50 is None) != (p95 is None):
        raise ValueError("provide both --p50 and --p95")
    if (usd_in_per_m is None) != (usd_out_per_m is None):
        raise ValueError("provide both input and output prices")
    if p50 is not None:
        axes["S"] = speed_axis(p50, p95)

    def report(candidate: Policy) -> dict:
        result = {**score_reads(rows, candidate), **axes}
        singles = [row for row in rows if row.get("readout") != "grouped_approx"]
        if usd_in_per_m is not None:
            if singles:
                usd = decision_cost(singles, usd_in_per_m, usd_out_per_m)
                result.update(usd_per_1000_decisions=usd, Cost=cost_score(usd))
            grouped = [row for row in rows if row.get("readout") == "grouped_approx"]
            if grouped:
                usd = decision_cost(grouped, usd_in_per_m, usd_out_per_m)
                result["grouped_approx"].update(usd_per_1000_decisions=usd, Cost=cost_score(usd))
        if (
            result["I"] is not None
            and result["C"] is not None
            and "S" in result
            and "Cost" in result
        ):
            for view in ("A", "B"):
                result[f"composite_{view}"] = composite(
                    result["I"], result["C"], result["S"], result["Cost"], view=view
                )
        return result

    result = report(policy)
    result["ablation"] = {name: report(candidate) for name, candidate in policies.items()}
    if not result["missing_types"]:
        baseline = baseline or policies["raw"]
        result["paired_bootstrap"] = paired_bootstrap(rows, baseline, policy, B=B, seed=seed)
    else:
        result["paired_bootstrap"] = None
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reads", nargs="+")
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--compare-policy", help="baseline policy for the paired I delta; default is raw"
    )
    parser.add_argument("--output", "--out", default="evaluation.json")
    parser.add_argument("--p50", type=float, help="raw seconds")
    parser.add_argument("--p95", type=float, help="raw seconds")
    parser.add_argument("--usd-in-per-m", type=float)
    parser.add_argument("--usd-out-per-m", type=float)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=15)
    args = parser.parse_args(argv)
    try:
        result = evaluate(
            load_reads(args.reads),
            Policy.load(args.policy),
            p50=args.p50,
            p95=args.p95,
            usd_in_per_m=args.usd_in_per_m,
            usd_out_per_m=args.usd_out_per_m,
            baseline=Policy.load(args.compare_policy) if args.compare_policy else None,
            B=args.bootstrap,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print("\nAblation                        I          C          A          B")
    for name, ablation in result["ablation"].items():
        values = [
            "n/a" if ablation.get(key) is None else f"{ablation[key]:.4f}"
            for key in ("I", "C", "composite_A", "composite_B")
        ]
        label = "temps_commit (margin 0)" if name == "temps_commit" else name
        print(f"{label:27s} " + " ".join(f"{value:>10s}" for value in values))


if __name__ == "__main__":
    main()
