"""Fit NLL temperatures, then choose Noul commitment by local composite A."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import asdict

from .collect import load_reads
from .evaluate import case_bootstrap_composite
from .losses import row_nll
from .policy import Policy
from .prompt import validate_prompt_variant
from .score import TYPES, composite, score_reads

COMMIT_MARGINS = (None, *(index / 40 for index in range(13)))
TEMPERATURE_MULTIPLIERS = (0.5, 0.7, 1.0, 1.4, 2.0)
SIMPLICITY_TOLERANCE = 0.25


def nll(rows: list[dict], temperature: float) -> float:
    return math.fsum(row_nll(row, temperature) for row in rows) / len(rows) if rows else 0.0


def golden_section(objective: Callable[[float], float], lower: float, upper: float) -> float:
    ratio = (math.sqrt(5) - 1) / 2
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    f_left, f_right = objective(left), objective(right)
    for _ in range(100):
        if upper - lower < 1e-8:
            break
        if f_left < f_right:
            upper, right, f_right = right, left, f_left
            left = upper - ratio * (upper - lower)
            f_left = objective(left)
        else:
            lower, left, f_left = left, right, f_right
            right = lower + ratio * (upper - lower)
            f_right = objective(right)
    return (lower + upper) / 2


def fit_temperature(rows: list[dict]) -> float:
    if not rows:
        return 1.0
    lower, upper = math.log(0.25), math.log(10)
    log_t = golden_section(lambda value: nll(rows, math.exp(value)), lower, upper)
    candidates = [1.0, 0.25, 10.0, math.exp(log_t)]
    return min(candidates, key=lambda value: nll(rows, value))


def fit_policy(
    rows: list[dict],
    *,
    allow_public: bool = False,
    diagnostic: bool = False,
    include_grouped: bool = False,
    fitted_on: str = "",
    speed_axis: float = 91.0,
    cost_axis: float = 56.4,
) -> Policy:
    """Maximize local A on calibration reads with fixed Speed and Cost.

    Search Noul NLL temperature times {0.5, 0.7, 1, 1.4, 2} jointly with
    {None, 0, .025, ..., .3} commitment margins. If any no-commit candidate
    lies within .25 points of the maximum A, prefer no commitment and the
    smallest absolute log-temperature deviation from the NLL fit among those
    candidates, then higher A. Otherwise maximize A, breaking exact ties by
    that temperature deviation, no commitment, then smaller margin. Require
    all three primitives rather than inventing missing composite axes.
    Choice ECE uses recorded hard-tier items, falling back to all Choice items
    when none exist; Choice TVD uses every available exact gold distribution.
    The case-bootstrap CI measures calibration-fit uncertainty conditional
    on selection; independent dev reads are still needed to assess overfitting.
    """
    if not rows:
        raise ValueError("cannot fit empty reads")
    variants = {row.get("prompt_variant", "min") for row in rows}
    if len(variants) != 1:
        raise ValueError("cannot fit mixed prompt_variant reads")
    prompt_variant = variants.pop()
    validate_prompt_variant(prompt_variant)
    public_count = sum(bool(row.get("public")) for row in rows)
    if public_count and not (allow_public or diagnostic):
        raise ValueError(
            "REFUSING public=True reads; use --allow-public only for an explicit public fit"
        )
    diagnostic = diagnostic or allow_public
    invalid_split = any(
        row.get("split") != "calibration" or row.get("public") is not False for row in rows
    )
    if invalid_split and not diagnostic:
        raise ValueError(
            "fitting accepts non-public split == calibration only; use --diagnostic for exploration"
        )
    grouped = [row for row in rows if row.get("readout") == "grouped_approx"]
    if not include_grouped:
        rows = [row for row in rows if row.get("readout") != "grouped_approx"]
    if not rows:
        raise ValueError("no single-pass reads to fit; grouped_approx excluded by default")
    recipes = {
        (
            row.get("model"),
            row.get("revision"),
            json.dumps(
                {
                    key: value
                    for key, value in ((row.get("binding") or {}).get("runtime") or {}).items()
                    if key != "readout"
                },
                sort_keys=True,
            ),
        )
        for row in rows
    }
    if len(recipes) > 1:
        raise ValueError("calibration reads have different model/runtime recipes")
    if any(row["type"] not in TYPES for row in rows):
        raise ValueError("unknown primitive in reads")
    if set(TYPES) - {row["type"] for row in rows}:
        raise ValueError("composite fitting requires choice, noul and score reads")
    if any(not math.isfinite(axis) or axis <= 0 for axis in (speed_axis, cost_axis)):
        raise ValueError("speed-axis and cost-axis must be finite and positive")
    # Temperature-only callers historically omitted labels and tier metadata.
    reads = [{"labels": list(row["raw_probs"]), **row} for row in rows]
    temperatures = {
        f"t_{kind}": fit_temperature([row for row in rows if row["type"] == kind]) for kind in TYPES
    }
    provenance = f"{fitted_on or 'reads'}; NLL n={len(rows)}; public={public_count}; allow_public={allow_public}; diagnostic={diagnostic}"
    table = []
    for multiplier in TEMPERATURE_MULTIPLIERS:
        for margin in COMMIT_MARGINS:
            candidate = Policy(
                **{**temperatures, "t_noul": temperatures["t_noul"] * multiplier},
                commit_margin=margin,
                prompt_variant=prompt_variant,
            )
            report = score_reads(reads, candidate, include_grouped=include_grouped)
            table.append(
                {
                    "t_noul": candidate.t_noul,
                    "temperature_multiplier": multiplier,
                    "commit_margin": margin,
                    "I": report["I"],
                    "C": report["C"],
                    "choice_ece_scope": report["choice_ece_scope"],
                    "composite_A": composite(report["I"], report["C"], speed_axis, cost_axis),
                    "noul_CC": report["per_type_CC"]["noul"],
                    "noul_ECE": report["calibration_details"]["noul"]["ECE"],
                }
            )

    def simplicity(candidate: dict) -> tuple:
        margin = candidate["commit_margin"]
        return (
            abs(math.log(candidate["temperature_multiplier"])),
            margin is not None,
            margin if margin is not None else 0.0,
            candidate["t_noul"],
        )

    ranked = sorted(
        table, key=lambda candidate: (-candidate["composite_A"], *simplicity(candidate))
    )
    best = ranked[0]
    no_commit = [
        candidate
        for candidate in table
        if candidate["commit_margin"] is None
        and best["composite_A"] - candidate["composite_A"] <= SIMPLICITY_TOLERANCE
    ]
    chosen = (
        min(
            no_commit,
            key=lambda candidate: (
                abs(math.log(candidate["temperature_multiplier"])),
                -candidate["composite_A"],
                candidate["t_noul"],
            ),
        )
        if no_commit
        else best
    )
    runner_up = next(candidate for candidate in ranked if candidate is not chosen)
    policy = Policy(
        **{**temperatures, "t_noul": chosen["t_noul"]},
        commit_margin=chosen["commit_margin"],
        fitted_on=provenance + "; objective=local_composite_A",
        prompt_variant=prompt_variant,
        promotable=not (
            diagnostic
            or invalid_split
            or include_grouped
            or any(row.get("readout") == "alias_sum" for row in rows)
        ),
    )
    policy.search = {
        "objective": "local_composite_A",
        "hard_n": sum(
            not isinstance(row["gold"], dict) and row.get("gold_distribution") is None
            for row in rows
        ),
        "soft_n": sum(
            isinstance(row["gold"], dict) or row.get("gold_distribution") is not None
            for row in rows
        ),
        "grouped_excluded_n": 0 if include_grouped else len(grouped),
        "grouped_approx": score_reads(grouped, policy, include_grouped=True) if grouped else None,
        "choice_ece_scope": chosen["choice_ece_scope"],
        "speed_axis": speed_axis,
        "cost_axis": cost_axis,
        "nll_temperatures": temperatures,
        "simplicity_tolerance": SIMPLICITY_TOLERANCE,
        "selection_rule": "prefer no commit within tolerance, then closest NLL temperature",
        "candidates": table,
        "best": best,
        "chosen": chosen,
        "runner_up": {
            **runner_up,
            "gap_from_best": best["composite_A"] - runner_up["composite_A"],
            "delta_from_chosen": runner_up["composite_A"] - chosen["composite_A"],
            "within_0_25": best["composite_A"] - runner_up["composite_A"] <= SIMPLICITY_TOLERANCE,
        },
        "bootstrap": case_bootstrap_composite(
            reads, policy, speed=speed_axis, cost=cost_axis, include_grouped=include_grouped
        ),
    }
    return policy


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reads", nargs="+")
    parser.add_argument("--output", "--out", default="policy.json")
    parser.add_argument("--allow-public", action="store_true")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="allow non-calibration data; policy is not promotable",
    )
    parser.add_argument(
        "--include-grouped",
        action="store_true",
        help="include grouped approximation diagnostics; policy is not promotable",
    )
    parser.add_argument("--fitted-on", default="")
    parser.add_argument("--speed-axis", type=float, default=91.0)
    parser.add_argument("--cost-axis", type=float, default=56.4)
    args = parser.parse_args(argv)
    try:
        policy = fit_policy(
            load_reads(args.reads),
            allow_public=args.allow_public,
            diagnostic=args.diagnostic,
            include_grouped=args.include_grouped,
            fitted_on=args.fitted_on or ", ".join(args.reads),
            speed_axis=args.speed_axis,
            cost_axis=args.cost_axis,
        )
    except ValueError as exc:
        parser.error(str(exc))
    policy.save(args.output)
    print(json.dumps(asdict(policy), indent=2))
    chosen = policy.search["chosen"]
    interval = policy.search["bootstrap"]["ci_95"]
    runner_up = policy.search["runner_up"]
    print(
        f"\nChosen local A={chosen['composite_A']:.4f}; case-bootstrap 95% CI "
        f"[{interval[0]:.4f}, {interval[1]:.4f}] (B=1000, seed=15; calibration-fit only)."
    )
    print(
        f"Runner-up A={runner_up['composite_A']:.4f}, t_noul={runner_up['t_noul']:.6g}, "
        f"commit_margin={runner_up['commit_margin']}; "
        f"within 0.25 of best={runner_up['within_0_25']}."
    )


if __name__ == "__main__":
    main()
