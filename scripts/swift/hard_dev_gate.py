"""Predeclared prompt-variant gate on v2 + non-public hard calibration/dev reads.

Protocol (frozen in docs/experiments/SWIFT_HARD_DEV_PROTOCOL_2026-10-05.md before any hard read):

1. Calibration = v2 calibration reads + hard calibration reads, per variant. Every variant,
   ``min`` included, is refit with the same ``fit_policy`` so the comparison is symmetric.
2. The variant with the highest calibration A is selected (ties: predeclared variant order).
3. If it is not ``min``, it is compared with ``min`` on dev = v2 dev + hard dev using the
   unchanged adoption bootstrap and ``gate_decision`` (same constants as ``adopt.py``).
4. Hard-only and v2-only dev comparisons are reported as diagnostics; they never decide.

Public JevBench reads are refused. The script only reads saved files; it never calls a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.swift.adopt import (  # noqa: E402
    GATE_CONSTANTS,
    gate_decision,
    paired_comparison,
    resource_axes,
    validate_roles,
)
from ayaka.swift.collect import load_reads  # noqa: E402
from ayaka.swift.fit import fit_policy  # noqa: E402
from ayaka.swift.score import composite, score_reads  # noqa: E402

ASSUMED_SPEED = 91.0
ASSUMED_COST = 56.4
USD_PER_M = 0.0403
VARIANTS = tuple(GATE_CONSTANTS["variant_order"])


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hard_ids(path: Path) -> set[str]:
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        ids |= {f"{record['id']}/{q['id']}" for q in record["questions"]}
    return ids


def load_variant(v2_dir: Path, hard_dir: Path, variant: str, role: str, expected: set[str]):
    v2 = load_reads([v2_dir / variant / f"v2_{role}.reads.jsonl"])
    hard = load_reads([hard_dir / variant / f"hard_{role}.reads.jsonl"])
    # Grouped MASSIVE rows have no single-pass readout; the adoption gate excludes them too.
    v2 = [r for r in v2 if r["readout"] != "grouped_approx"]
    if any(r["public"] for r in v2 + hard):
        raise ValueError("public reads are refused by the hard dev protocol")
    ids = {r["id"] for r in hard}
    if len(ids) != len(hard) or not ids <= expected or len(ids) != len(expected):
        raise ValueError(f"{variant}/{role}: hard reads must cover the frozen file exactly once")
    return sorted(v2, key=lambda r: r["id"]), sorted(hard, key=lambda r: r["id"])


def aligned(before, after):
    if [r["id"] for r in before] != [r["id"] for r in after]:
        raise ValueError("variant reads are not aligned by id")
    return before, after


def run(v2_dir: Path, hard_dir: Path, manifest_path: Path, variants) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {}
    for role in ("calibration", "dev"):
        path = manifest_path.parent / manifest["files"][role]["path"]
        if file_sha(path) != manifest["files"][role]["sha256"]:
            raise ValueError(f"hard {role} file does not match the frozen manifest")
        expected[role] = hard_ids(path)
    reads = {
        role: {v: load_variant(v2_dir, hard_dir, v, role, expected[role]) for v in variants}
        for role in ("calibration", "dev")
    }
    cal_all = [r for v in variants for part in reads["calibration"][v] for r in part]
    dev_all = [r for v in variants for part in reads["dev"][v] for r in part]
    # Same model/recipe, role isolation, all primitive types; its row order is adopt.py's order,
    # which the bootstrap draws depend on.
    cal_groups, dev_groups = validate_roles(cal_all, dev_all)
    hard_dev_ids = {r["id"] for r in reads["dev"]["min"][1]}

    policies, cal_A = {}, {}
    for variant in variants:
        rows = cal_groups[variant]
        axes = resource_axes(rows, ASSUMED_SPEED, ASSUMED_COST, USD_PER_M, USD_PER_M)
        policies[variant] = fit_policy(
            rows,
            fitted_on="v2+hard calibration (hard dev protocol)",
            speed_axis=axes["S"],
            cost_axis=axes["Cost"],
        )
        score = score_reads(rows, policies[variant])
        cal_A[variant] = composite(score["I"], score["C"], axes["S"], axes["Cost"])
    selected = max(cal_A, key=cal_A.__getitem__)

    def compare(variant, part):
        def keep(row):
            return part == "union" or (row["id"] in hard_dev_ids) == (part == "hard")

        before = [r for r in dev_groups["min"] if keep(r)]
        after = [r for r in dev_groups[variant] if keep(r)]
        before, after = aligned(before, after)
        comparison = paired_comparison(
            before,
            policies["min"],
            after,
            policies[variant],
            before_speed=ASSUMED_SPEED,
            after_speed=ASSUMED_SPEED,
            assumed_cost=ASSUMED_COST,
            usd_in_per_m=USD_PER_M,
            usd_out_per_m=USD_PER_M,
        )
        return {
            "delta_A": comparison["delta_A"],
            "ci_95_A": comparison["ci_95_A"],
            "delta_I": comparison["delta_I"],
            "ci_95_I": comparison["ci_95_I"],
            "per_type_CC_delta": comparison["per_type_CC_delta"],
            "before": {k: comparison["before"][k] for k in ("A", "I", "C", "S", "Cost")},
            "after": {k: comparison["after"][k] for k in ("A", "I", "C", "S", "Cost")},
            "decisions": len(before),
            "clusters": comparison["cluster_count"],
            "gate": gate_decision(comparison),
        }

    decision = (
        {"adopted": False, "reason": "min_selected_on_calibration"}
        if selected == "min"
        else compare(selected, "union")["gate"]
    )
    parts = ("union", "v2", "hard") if reads["dev"]["min"][1] else ("union", "v2")
    diagnostics = {
        variant: {part: compare(variant, part) for part in parts}
        for variant in variants
        if variant != "min"
    }
    return {
        "protocol": "docs/experiments/SWIFT_HARD_DEV_PROTOCOL_2026-10-05.md",
        "hard_manifest_sha256": file_sha(manifest_path),
        "hard_files": {role: manifest["files"][role]["sha256"] for role in ("calibration", "dev")},
        "gate_constants": GATE_CONSTANTS,
        "assumptions": {
            "speed": ASSUMED_SPEED,
            "cost_axis_fallback": ASSUMED_COST,
            "usd_per_m": USD_PER_M,
        },
        "calibration_A": cal_A,
        "selected_on_calibration": selected,
        "decision": decision,
        "diagnostics_not_gating": diagnostics,
        "policies": {variant: policy.__dict__ for variant, policy in policies.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dir", type=Path, required=True, help="per-variant v2 cal/dev reads")
    parser.add_argument("--hard-dir", type=Path, required=True, help="per-variant hard reads")
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    variants = tuple(v for v in VARIANTS if v in args.variants.split(","))
    if variants[0] != "min":
        raise SystemExit("min must be included")
    report = run(args.v2_dir, args.hard_dir, args.hard_manifest, variants)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    d = report["decision"]
    print(
        "selected",
        report["selected_on_calibration"],
        "| adopted",
        d.get("adopted"),
        "|",
        d.get("reason"),
    )
    for variant, parts in report["diagnostics_not_gating"].items():
        for part, c in parts.items():
            print(
                f"{variant:8s} {part:5s} dA {c['delta_A']:+.3f} CI [{c['ci_95_A'][0]:+.3f}, {c['ci_95_A'][1]:+.3f}]"
                f" dI {c['delta_I']:+.2f} n={c['decisions']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
