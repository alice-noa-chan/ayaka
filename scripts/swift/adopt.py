"""Fit and gate optional Swift levers; write adoption.json and accepted policy.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.swift.adopt import GATE_CONSTANTS, adopt_levers  # noqa: E402
from ayaka.swift.collect import load_reads  # noqa: E402
from ayaka.swift.policy import Policy  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", nargs="+", required=True)
    parser.add_argument("--dev", nargs="+", required=True)
    parser.add_argument("--baseline-policy", type=Path, required=True)
    parser.add_argument("--levers", nargs="+", choices=GATE_CONSTANTS["lever_order"], default=[])
    parser.add_argument("--latency", nargs="+", help="complete serial non-public dev probes")
    parser.add_argument("--speed-axis", type=float, default=91.0)
    parser.add_argument("--cost-axis", type=float, default=56.4)
    parser.add_argument("--usd-in-per-m", type=float, default=0.0403)
    parser.add_argument("--usd-out-per-m", type=float, default=0.0)
    parser.add_argument(
        "--assume-cost", action="store_true", help="use fixed Cost instead of token cost"
    )
    parser.add_argument("--output", type=Path, default=Path("adoption.json"))
    parser.add_argument("--policy", type=Path, default=Path("policy.json"))
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() == args.policy.resolve():
            raise ValueError("report and policy destinations must differ")
        result = adopt_levers(
            load_reads(args.calibration),
            load_reads(args.dev),
            Policy.load(args.baseline_policy),
            levers=args.levers,
            latency=[json.loads(Path(p).read_text(encoding="utf-8")) for p in args.latency]
            if args.latency
            else None,
            assumed_speed=args.speed_axis,
            assumed_cost=args.cost_axis,
            usd_in_per_m=None if args.assume_cost else args.usd_in_per_m,
            usd_out_per_m=args.usd_out_per_m,
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.policy.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    Policy(**result["final_policy"]).save(args.policy)
    print(
        f"Adopted: {', '.join(result['adopted_levers']) or 'none'}; "
        f"report: {args.output}; policy: {args.policy}"
    )


if __name__ == "__main__":
    main()
