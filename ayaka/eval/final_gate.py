"""Apply a frozen held-out tuning fit to the final_test reads and gate it once over v1.

``heldout_tuning`` fits path temperatures on calibration, the router on
router_train (with dev for its lambda and promotion) and chooses the serving
policy on router_train. ``frozen_fit`` records exactly those three things.
This module then reads the final_test cohort once:

1. apply the frozen temperatures to v2 direct and reasoned reads;
2. compose the frozen policy with the frozen router;
3. gate the result over v1 on the same questions, with the clustered
   non-inferiority rule for per-source and per-language checks, and report
   the gate over calibrated v2 direct as a secondary check.

Nothing is fitted or chosen here. The decision is ``adopted``: the v1 gate
passes and the policy's Speed axis is at least ``SPEED_AXIS_MINIMUM``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ..routing import LAMBDAS, BenefitRouter
from ..training.path_calibration import PathCalibration
from .checkpoint_comparison import ADAPTERS
from .heldout_tuning import (
    POLICIES,
    brief,
    calibrated,
    compose,
    gate,
    load_run,
    policy_rule,
)
from .quality_hierarchy import FINAL_SPLIT

VERSION = "ayaka-final-gate-1"
FROZEN_VERSION = "ayaka-frozen-tuning-1"
SUBGROUP_RULE = "clustered_noninferiority"
# JevBench multiplies the composite by (x/50)^2 below 50 on the Speed axis.
SPEED_AXIS_MINIMUM = 50.0


def frozen_fit(calibration, router, policy, adapter):
    """The complete serving choice of one tuning run, as plain JSON."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    return validate_frozen(
        {
            "version": FROZEN_VERSION,
            "adapter": adapter,
            "path_temperatures": dict(calibration.temperatures),
            "router": asdict(router),
            "policy": policy,
        }
    )


def validate_frozen(frozen):
    if not isinstance(frozen, dict) or frozen.get("version") != FROZEN_VERSION:
        raise ValueError(f"frozen tuning must declare version {FROZEN_VERSION}")
    if set(frozen) != {"version", "adapter", "path_temperatures", "router", "policy"}:
        raise ValueError("frozen tuning has unknown or missing fields")
    if frozen["adapter"] not in ADAPTERS or frozen["policy"] not in POLICIES:
        raise ValueError("frozen tuning names an unknown adapter or policy")
    PathCalibration(frozen["path_temperatures"])  # validates every temperature
    router = BenefitRouter(**frozen["router"])
    if "router" in frozen["policy"]:
        # Serving never routes with an unpromoted router, so neither does the gate.
        router.validate_promoted()
    elif router.penalty not in (0.0, *LAMBDAS):
        raise ValueError("frozen router has an unknown lambda")
    values = [*router.mean, *router.scale, *router.gain_weights, *router.token_weights]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("frozen router coefficients must be finite")
    return frozen


def final_gate(results, frozen, *, replicates=2000):
    frozen = validate_frozen(frozen)
    results = Path(results)
    adapter = frozen["adapter"]
    off = load_run(results, FINAL_SPLIT, "v2_off", adapter=adapter)
    on = load_run(results, FINAL_SPLIT, "v2_on", adapter=adapter)
    v1 = sorted(load_run(results, FINAL_SPLIT, "v1_on"), key=lambda row: row["id"])
    calibration = PathCalibration(frozen["path_temperatures"])
    router = BenefitRouter(**frozen["router"])
    direct = sorted(calibrated(off, calibration), key=lambda row: row["id"])
    reasoned = calibrated(on, calibration)
    rows = compose(direct, reasoned, policy_rule(frozen["policy"], router))
    if [row["id"] for row in rows] != [row["id"] for row in v1]:
        raise ValueError("v1 and v2 final_test reads must cover the same questions")
    over_v1 = gate(v1, rows, major_gain=True, replicates=replicates, subgroup_rule=SUBGROUP_RULE)
    over_off = gate(
        direct, rows, major_gain=False, replicates=replicates, subgroup_rule=SUBGROUP_RULE
    )
    policy = brief(rows)
    speed = policy["latency"]["speed_axis"]
    return {
        "version": VERSION,
        "split": FINAL_SPLIT,
        "frozen": frozen,
        "systems": {"v1_on": brief(v1), "v2_off_calibrated": brief(direct), "policy": policy},
        "reasoned_by_type": dict(Counter(r["type"] for r in rows if r["route"] != "direct")),
        "gates": {"over_v1_on": over_v1, "over_v2_off_calibrated": over_off},
        "speed_axis_minimum": SPEED_AXIS_MINIMUM,
        "adopted": over_v1["screen_passed"] and speed >= SPEED_AXIS_MINIMUM,
        "decision_rule": "adopt iff the v1 gate passes and the policy Speed axis >= 50",
        "optimizer_updates": 0,
        "fitted_on_final_test": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, help="results/ folder with final_test reads")
    parser.add_argument("--frozen", required=True, type=Path, help="frozen tuning JSON")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError("the final gate report must be a new file")
    report = final_gate(
        args.results, json.loads(args.frozen.read_bytes()), replicates=args.replicates
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "adopted": report["adopted"],
                "cc_delta_over_v1": round(report["gates"]["over_v1_on"]["cc_delta"], 2),
                "failed_checks": report["gates"]["over_v1_on"]["failed_checks"],
                "speed_axis": round(report["systems"]["policy"]["latency"]["speed_axis"], 1),
            }
        )
    )


if __name__ == "__main__":
    main()
