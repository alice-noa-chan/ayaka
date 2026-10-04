"""Select prompts using matching non-public dev reads, with policies fitted on v2 calibration.

Pass only recorded dev items as --dev; fit only recorded --calibration.
Cygnet calibration remains a separate diagnostic and is never relabeled dev.
Pick the highest local composite A. If its paired case-bootstrap 95% CI against
the runner-up includes zero, prefer the one with fewer mean input tokens. With
equal token counts, retain the higher A (then variant name for an exact tie).
Default bootstrap: B=2000, seed=15. Policies stay fixed during dev resampling;
Speed stays fixed at the variant's complete serial probe, or the explicitly
reported --speed-axis estimate. Cost uses input tokens only, including all
hierarchical passes, at --usd-in-per-m (default 0.0403). Public accuracy reads
are refused; public serial latency probes contain timing only and are allowed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.swift.collect import load_reads  # noqa: E402
from ayaka.swift.evaluate import cluster_strata, percentile  # noqa: E402
from ayaka.swift.fit import fit_policy  # noqa: E402
from ayaka.swift.policy import Policy  # noqa: E402
from ayaka.swift.provenance import assert_roles_isolated, group_reads  # noqa: E402
from ayaka.swift.score import (  # noqa: E402
    TYPES,
    calibration,
    composite,
    cost_score,
    intelligence,
    prepare_rows,
    score_reads,
    speed_axis,
)


def input_cost(rows: list[dict], usd_in_per_m: float) -> tuple[float, float, float]:
    tokens = mean(row["input_tokens"] for row in rows)
    usd_per_1000 = tokens * usd_in_per_m / 1000
    return tokens, usd_per_1000, cost_score(usd_per_1000)


def choose_variant(reports: dict[str, dict], ci_vs_runner_up: list[float]) -> dict:
    ranked = sorted(reports, key=lambda variant: (-reports[variant]["composite_A"], variant))
    best, runner_up = ranked[:2]
    uncertain = ci_vs_runner_up[0] <= 0 <= ci_vs_runner_up[1]
    selected = (
        min(
            (best, runner_up),
            key=lambda variant: (
                reports[variant]["mean_input_tokens"],
                -reports[variant]["composite_A"],
                variant,
            ),
        )
        if uncertain
        else best
    )
    return {
        "selected": selected,
        "highest_A": best,
        "runner_up": runner_up,
        "ci_95_A_vs_runner_up": ci_vs_runner_up,
        "token_fallback": uncertain,
        "rule": "highest A; if CI vs runner-up includes 0, prefer fewer mean input tokens",
    }


def paired_case_bootstrap(
    dev: dict[str, list[dict]],
    policies: dict[str, Policy],
    speeds: dict[str, float],
    *,
    usd_in_per_m: float = 0.0403,
    B: int = 2000,
    seed: int = 15,
) -> dict:
    """Use identical whole-case draws for all variants, stratified by type coverage.

    Explicit cluster_id/metadata lineage joins questions and language variants;
    legacy case_id falls back to source namespaces. Public items are singletons.
    Recompute I, C and mean input cost
    per draw. This is conditional on fitted policies and measured/estimated Speed.
    """
    if B < 1:
        raise ValueError("bootstrap repetitions must be positive")
    items = {variant: prepare_rows(rows, policies[variant]) for variant, rows in dev.items()}
    cases, strata = cluster_strata(dev["min"])
    if set(TYPES) - {item.type for item in items["min"]}:
        raise ValueError("dev bootstrap requires choice, noul and score reads")

    def axes(variant: str, indices: list[int]) -> tuple[float, float]:
        sample = [items[variant][index] for index in indices]
        i = intelligence(sample)["I"]
        c = calibration(sample)["C"]
        cost = input_cost([dev[variant][index] for index in indices], usd_in_per_m)[2]
        return composite(i, c, speeds[variant], cost), i

    point = {variant: axes(variant, list(range(len(dev[variant])))) for variant in dev}
    draws: dict[str, list[tuple[float, float]]] = {variant: [] for variant in dev}
    rng = random.Random(seed)
    for _ in range(B):
        indices = [
            index for stratum in strata.values() for _ in stratum for index in rng.choice(stratum)
        ]
        for variant in dev:
            draws[variant].append(axes(variant, indices))

    def comparison(candidate: str, baseline: str) -> dict:
        result = {}
        for index, axis in enumerate(("A", "I")):
            delta = [
                after[index] - before[index]
                for after, before in zip(draws[candidate], draws[baseline], strict=True)
            ]
            result[f"delta_{axis}"] = point[candidate][index] - point[baseline][index]
            result[f"ci_95_{axis}"] = [percentile(delta, 0.025), percentile(delta, 0.975)]
        return result

    return {
        "B": B,
        "seed": seed,
        "unit": "case",
        "resampling_unit": "cluster",
        "cluster_count": len(cases),
        "case_count": len(cases),
        "stratification": "primitive_type_coverage",
        "conditional_on_fitted_policies_and_speed": True,
        "vs_min": {variant: comparison(variant, "min") for variant in dev},
        "pairwise": {
            candidate: {
                baseline: comparison(candidate, baseline)
                for baseline in dev
                if baseline != candidate
            }
            for candidate in dev
        },
    }


def select_variants(
    calibration_rows: list[dict],
    dev_rows: list[dict],
    *,
    latency: list[dict] | None = None,
    usd_in_per_m: float = 0.0403,
    estimated_speed: float = 91.0,
    B: int = 2000,
    seed: int = 15,
    exploratory: bool = False,
) -> dict:
    if not math.isfinite(usd_in_per_m) or usd_in_per_m <= 0:
        raise ValueError("usd-in-per-m must be finite and positive")
    if not math.isfinite(estimated_speed) or estimated_speed <= 0:
        raise ValueError("speed-axis must be finite and positive")
    if B < 1:
        raise ValueError("bootstrap repetitions must be positive")
    cal = group_reads(calibration_rows, "calibration", exploratory=exploratory)
    dev = group_reads(dev_rows, "dev", exploratory=exploratory)
    if set(cal) != set(dev):
        raise ValueError("calibration and dev must contain the same variants")
    identities = {(row["model"], row["revision"]) for row in calibration_rows + dev_rows}
    if len(identities) != 1:
        raise ValueError("all variant reads must use the SAME model and revision")
    model, revision = identities.pop()
    assert_roles_isolated(calibration_rows, dev_rows)
    if not exploratory:
        recipes = {
            json.dumps(
                {
                    key: value
                    for key, value in row["binding"]["runtime"].items()
                    if key != "prompt_variant"
                },
                sort_keys=True,
            )
            for row in calibration_rows + dev_rows
        }
        if (
            len(recipes) != 1
            or len({row["tokenizer_revision"] for row in calibration_rows + dev_rows}) != 1
        ):
            raise ValueError("variant reads have different tokenizer/runtime recipes")
    speeds = dict.fromkeys(dev, estimated_speed)
    probes = {}
    for probe in latency or []:
        variant = probe.get("prompt_variant")
        if variant not in dev or variant in probes:
            raise ValueError("latency must have exactly one probe per supplied variant")
        if (
            probe.get("complete") is not True
            or probe.get("concurrency") != 1
            or probe.get("units") != "seconds"
            or probe.get("completed_reads") != probe.get("requested_reads")
            or not probe.get("completed_reads", 0)
            or (probe.get("self_hosted_adjustment") or {}).get("applied")
        ):
            raise ValueError("latency requires complete serial probes in raw seconds")
        if (probe.get("model"), probe.get("revision")) != (model, revision):
            raise ValueError("latency model/revision differs from reads")
        speeds[variant] = speed_axis(probe["p50_s"], probe["p95_s"])
        probes[variant] = probe
    if probes and set(probes) != set(dev):
        raise ValueError("provide latency probes for every variant, or omit all probes")
    if probes:
        signatures = {
            (probe.get("seed"), tuple(sample["id"] for sample in probe.get("samples", [])))
            for probe in probes.values()
        }
        if len(signatures) != 1:
            raise ValueError("latency variants must probe the same ordered items and seed")
    policies, reports = {}, {}
    for variant in dev:
        cal_cost = input_cost(cal[variant], usd_in_per_m)[2]
        policies[variant] = fit_policy(
            cal[variant],
            fitted_on="v2 calibration (variant selector)",
            speed_axis=speeds[variant],
            cost_axis=cal_cost,
            exploratory=exploratory,
        )
        tokens, usd, cost = input_cost(dev[variant], usd_in_per_m)
        report = score_reads(dev[variant], policies[variant])
        if report["missing_types"]:
            raise ValueError("dev requires choice, noul and score reads")
        reports[variant] = {
            **report,
            "prompt_variant": variant,
            "mean_input_tokens": tokens,
            "usd_per_1000_decisions": usd,
            "Cost": cost,
            "S": speeds[variant],
            "speed_source": "serial_latency_probe" if probes else "estimated_speed_axis",
            "composite_A": composite(report["I"], report["C"], speeds[variant], cost),
        }
    bootstrap = paired_case_bootstrap(
        dev,
        policies,
        speeds,
        usd_in_per_m=usd_in_per_m,
        B=B,
        seed=seed,
    )
    ranked = sorted(reports, key=lambda variant: (-reports[variant]["composite_A"], variant))
    ci = bootstrap["pairwise"][ranked[0]][ranked[1]]["ci_95_A"]
    for variant, report in reports.items():
        report["paired_vs_min"] = bootstrap["vs_min"][variant]
    return {
        "promotable": not exploratory and all(policy.promotable for policy in policies.values()),
        "exploratory": exploratory,
        "model": model,
        "revision": revision,
        "split": "non_public_dev",
        "usd_in_per_m": usd_in_per_m,
        "output_token_cost_included": False,
        "calibration_n_per_variant": len(cal["min"]),
        "dev_n_per_variant": len(dev["min"]),
        "variants": reports,
        "policies": {variant: asdict(policy) for variant, policy in policies.items()},
        "bootstrap": bootstrap,
        "selection": choose_variant(reports, ci),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration", nargs="+", required=True, help="v2 calibration reads, all variants"
    )
    parser.add_argument(
        "--dev", nargs="+", required=True, help="recorded v2 dev reads, all variants"
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="accept unbound/legacy diagnostics; policies never promotable",
    )
    parser.add_argument("--latency", nargs="+", help="complete latency.json for every variant")
    parser.add_argument("--usd-in-per-m", type=float, default=0.0403)
    parser.add_argument(
        "--speed-axis", type=float, default=91.0, help="estimate used only without probes"
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--output", type=Path, default=Path("variant_selection.json"))
    parser.add_argument("--policy-dir", type=Path, default=Path("variant_policies"))
    args = parser.parse_args(argv)
    try:
        result = select_variants(
            load_reads(args.calibration),
            load_reads(args.dev),
            latency=[json.loads(Path(path).read_text(encoding="utf-8")) for path in args.latency]
            if args.latency
            else None,
            usd_in_per_m=args.usd_in_per_m,
            estimated_speed=args.speed_axis,
            B=args.bootstrap,
            seed=args.seed,
            exploratory=args.exploratory,
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    args.policy_dir.mkdir(parents=True, exist_ok=True)
    for variant, policy in result["policies"].items():
        Policy(**policy).save(args.policy_dir / f"{variant}.policy.json")
    selected = result["selection"]["selected"]
    Policy(**result["policies"][selected]).save(args.policy_dir / "policy.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        "Variant       I        C        A       S     Cost   Mean input tokens    ΔA CI vs min        ΔI CI vs min"
    )
    for variant, report in result["variants"].items():
        delta = report["paired_vs_min"]
        print(
            f"{variant:8s} {report['I']:8.3f} {report['C']:8.3f} {report['composite_A']:8.3f} "
            f"{report['S']:7.3f} {report['Cost']:7.3f} {report['mean_input_tokens']:10.2f} "
            f"  {delta['ci_95_A']}  {delta['ci_95_I']}"
        )
    print(f"Selected {selected}: {json.dumps(result['selection'])}")
    print(f"Report: {args.output}; policies: {args.policy_dir}")


if __name__ == "__main__":
    main()
