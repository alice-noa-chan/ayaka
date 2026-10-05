"""Predeclared, sequential adoption on bound non-public calibration/dev reads.

Choose a prompt on calibration only, gate it once on dev, then fit/gate bias
against that accepted system. A failed candidate never causes a dev retry.
The bootstrap is conditional on calibration-fitted parameters and resource axes;
passing these finite-data gates is evidence, not certainty of future improvement.
"""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, replace

from ayaka.eval.read_artifact import fingerprint

from .bias import (
    BIAS_GRADIENT_TOLERANCE,
    BIAS_L2,
    BIAS_MAX_ITERATIONS,
    BIAS_MIN_QUESTIONS,
    fit_letter_bias,
)
from .evaluate import cluster_strata, percentile
from .fit import fit_policy
from .policy import Policy
from .provenance import assert_roles_isolated, group_reads
from .router import (
    ROUTE_RATE_CAP,
    ROUTER_ITERATIONS,
    ROUTER_L2,
    ROUTER_REFIT_TOLERANCE,
    fit_router,
    projected_latency,
    routed_rows,
    router_refit_difference,
    validate_pairs,
)
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

# All adoption choices are predeclared here and copied verbatim to every report.
# The Intelligence CI must also rule out a significantly negative change.
GATE_CONSTANTS = {
    "lever_order": ["variant", "bias", "reasoning_route"],
    "reasoning_route_rate_cap": ROUTE_RATE_CAP,
    "reasoning_router_l2": ROUTER_L2,
    "reasoning_router_iterations": ROUTER_ITERATIONS,
    "reasoning_projected_adjusted_p95_max_ratio": 1.10,
    "reasoning_candidate": "direct_max_prob<=0.95_or_digit_in_state_instructions; K<=26",
    "reasoning_threshold_selection": "highest_calibration_A_under_overall_cap; off_first_ties",
    "variant_order": ["min", "cygnet", "rules", "labeled"],
    "variant_selection": "highest_calibration_A_then_predeclared_order",
    "dev_latency_is_gate_only": True,
    "bootstrap_B": 2000,
    "bootstrap_seed": 15,
    "ci_level": 0.95,
    "delta_A_ci_lower_strictly_greater_than": 0.0,
    "delta_I_ci_upper_at_least": 0.0,
    "delta_I_point_at_least": -0.5,
    "primitive_CC_delta_at_least": -2.0,
    "bias_l2": BIAS_L2,
    "bias_min_questions": BIAS_MIN_QUESTIONS,
    "bias_max_iterations": BIAS_MAX_ITERATIONS,
    "bias_gradient_tolerance": BIAS_GRADIENT_TOLERANCE,
}


def policy_fingerprint(policy: Policy) -> str:
    values = asdict(policy)
    values.pop("adoption")
    return fingerprint(values)


def validate_adopted_policy(policy: Policy) -> None:
    """Serving guard: require fixed passing gates bound to these exact parameters."""
    active = []
    if policy.prompt_variant != "min":
        active.append("variant")
    if policy.letter_bias:
        active.append("bias")
    if policy.reasoning_route is not None:
        active.append("reasoning_route")
    if not active:
        return
    receipt = policy.adoption or {}
    if (
        policy.promotable is not True
        or receipt.get("contract") != "swift_adoption_v1"
        or receipt.get("constants") != GATE_CONSTANTS
        or receipt.get("policy_sha256") != policy_fingerprint(policy)
        or receipt.get("adopted_levers") != active
        or receipt.get("data_roles") != {"fit": "non_public_calibration", "gate": "non_public_dev"}
        or not receipt.get("model")
        or not receipt.get("revision")
    ):
        raise ValueError(
            "optional Swift levers require a matching predeclared adoption gate receipt"
        )
    hashes = receipt.get("data_hashes", {})
    if any(not hashes.get(role) for role in ("calibration", "dev")):
        raise ValueError("adoption receipt requires bound calibration/dev data hashes")
    comparisons = receipt.get("comparisons", {})
    if set(comparisons) != set(active) or any(
        comparison.get("B") != GATE_CONSTANTS["bootstrap_B"]
        or comparison.get("seed") != GATE_CONSTANTS["bootstrap_seed"]
        or comparison.get("resampling_unit") != "cluster"
        or not gate_decision(comparison)["adopted"]
        for comparison in (comparisons[lever] for lever in active)
    ):
        raise ValueError("optional Swift lever failed its predeclared adoption gate")
    if (
        "reasoning_route" in active
        and comparisons["reasoning_route"].get("lever") != "reasoning_route"
    ):
        raise ValueError("reasoning route receipt requires the projected p95 guard")
    if "reasoning_route" in active and any(
        not hashes.get("reasoned", {}).get(role) for role in ("calibration", "dev")
    ):
        raise ValueError("reasoning route receipt requires paired calibration/dev data hashes")


def gate_decision(comparison: dict) -> dict:
    """Conjunction of the fixed composite, Intelligence and primitive guards."""
    checks = {
        "positive_A_lower_bound": comparison["ci_95_A"][0]
        > GATE_CONSTANTS["delta_A_ci_lower_strictly_greater_than"],
        "I_not_significantly_negative": comparison["ci_95_I"][1]
        >= GATE_CONSTANTS["delta_I_ci_upper_at_least"],
        "I_point_guard": comparison["delta_I"] >= GATE_CONSTANTS["delta_I_point_at_least"],
        "primitive_CC_guard": all(
            comparison["per_type_CC_delta"][kind] >= GATE_CONSTANTS["primitive_CC_delta_at_least"]
            for kind in TYPES
        ),
    }
    if comparison.get("lever") == "reasoning_route":
        before = comparison.get("projected_adjusted_p95_before", 0)
        after = comparison.get("projected_adjusted_p95_after", math.inf)
        checks["projected_p95_guard"] = (
            math.isfinite(before)
            and math.isfinite(after)
            and before > 0
            and after <= before * GATE_CONSTANTS["reasoning_projected_adjusted_p95_max_ratio"]
        )
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "adopted": not reasons,
        "checks": checks,
        "reason": "all_predeclared_gates_passed" if not reasons else ", ".join(reasons),
    }


def validate_roles(calibration_rows: list[dict], dev_rows: list[dict]) -> tuple[dict, dict]:
    """Reuse selector role/binding/pairing/isolation checks without overrides."""
    cal = group_reads(calibration_rows, "calibration", require_variants=False)
    dev = group_reads(dev_rows, "dev", require_variants=False)
    if "min" not in cal or set(cal) != set(dev):
        raise ValueError("adoption requires min and matching calibration/dev variants")
    rows = calibration_rows + dev_rows
    if len({(row["model"], row["revision"], row["tokenizer_revision"]) for row in rows}) != 1:
        raise ValueError("adoption reads must use the SAME model/revision/tokenizer")
    recipes = {
        json.dumps(
            {
                key: value
                for key, value in row["binding"]["runtime"].items()
                if key != "prompt_variant"
            },
            sort_keys=True,
        )
        for row in rows
    }
    if len(recipes) != 1:
        raise ValueError("adoption reads have different runtime recipes")
    assert_roles_isolated(calibration_rows, dev_rows)
    for role, groups in (("calibration", cal), ("dev", dev)):
        if any(set(TYPES) - {row["type"] for row in rows} for rows in groups.values()):
            raise ValueError(f"{role} adoption requires all three primitive types")
    return cal, dev


def resource_speeds(dev: dict, latency: list[dict] | None, assumed_speed: float) -> dict:
    """Only complete, serial, non-public dev probes may influence a gate."""
    speeds = dict.fromkeys(dev, assumed_speed)
    probes = {}
    identity = (dev["min"][0]["model"], dev["min"][0]["revision"])
    for probe in latency or []:
        variant = probe.get("prompt_variant")
        if variant not in dev or variant in probes:
            raise ValueError("latency needs one probe per variant")
        if probe.get("public") is not False or probe.get("split") != "dev":
            raise ValueError("adoption latency must record non-public dev; public/test refused")
        if (
            probe.get("complete") is not True
            or probe.get("concurrency") != 1
            or probe.get("units") != "seconds"
            or probe.get("completed_reads") != probe.get("requested_reads")
            or not probe.get("completed_reads", 0)
            or (probe.get("self_hosted_adjustment") or {}).get("applied")
            or (probe.get("model"), probe.get("revision")) != identity
        ):
            raise ValueError(
                "latency requires complete serial raw-second probes for the bound model"
            )
        speeds[variant] = speed_axis(probe["p50_s"], probe["p95_s"])
        probes[variant] = probe
    if probes and set(probes) != set(dev):
        raise ValueError("supply latency for every variant, or omit all probes")
    if probes:
        signatures = {
            (probe.get("seed"), tuple(sample["id"] for sample in probe.get("samples", [])))
            for probe in probes.values()
        }
        if len(signatures) != 1 or not next(iter(signatures))[1]:
            raise ValueError("latency variants must use the same ordered items and seed")
    return speeds


def resource_axes(rows, speed, assumed_cost, usd_in_per_m, usd_out_per_m):
    measured_cost = usd_in_per_m is not None
    cost = (
        cost_score(decision_cost(rows, usd_in_per_m, usd_out_per_m))
        if measured_cost
        else assumed_cost
    )
    return {
        "S": speed,
        "Cost": cost,
        "cost_source": "measured_tokens" if measured_cost else "assumed",
    }


def paired_comparison(
    before_rows,
    before_policy,
    after_rows,
    after_policy,
    *,
    before_speed,
    after_speed,
    assumed_cost,
    usd_in_per_m,
    usd_out_per_m,
):
    """Identical whole-case draws with existing primitive-coverage strata."""
    before, after = prepare_rows(before_rows, before_policy), prepare_rows(after_rows, after_policy)
    cases, strata = cluster_strata(before_rows)

    def axes(items, rows, speed, indices):
        sample = [items[index] for index in indices]
        i = intelligence(sample)
        c = calibration(sample)
        resources = resource_axes(
            [rows[index] for index in indices], speed, assumed_cost, usd_in_per_m, usd_out_per_m
        )
        return {
            **i,
            **resources,
            "C": c["C"],
            "A": composite(i["I"], c["C"], resources["S"], resources["Cost"]),
        }

    indices = list(range(len(before_rows)))
    point_before = axes(before, before_rows, before_speed, indices)
    point_after = axes(after, after_rows, after_speed, indices)
    rng = random.Random(GATE_CONSTANTS["bootstrap_seed"])
    draws = {"A": [], "I": []}
    for _ in range(GATE_CONSTANTS["bootstrap_B"]):
        indices = [
            index for stratum in strata.values() for _ in stratum for index in rng.choice(stratum)
        ]
        a = axes(before, before_rows, before_speed, indices)
        b = axes(after, after_rows, after_speed, indices)
        for axis in draws:
            draws[axis].append(b[axis] - a[axis])
    return {
        "delta_A": point_after["A"] - point_before["A"],
        "delta_I": point_after["I"] - point_before["I"],
        "ci_95_A": [percentile(draws["A"], 0.025), percentile(draws["A"], 0.975)],
        "ci_95_I": [percentile(draws["I"], 0.025), percentile(draws["I"], 0.975)],
        "per_type_CC_delta": {
            kind: point_after["per_type_CC"][kind] - point_before["per_type_CC"][kind]
            for kind in TYPES
        },
        "before": point_before,
        "after": point_after,
        "B": GATE_CONSTANTS["bootstrap_B"],
        "seed": GATE_CONSTANTS["bootstrap_seed"],
        "unit": "case",
        "resampling_unit": "cluster",
        "cluster_count": len(cases),
        "stratification": "primitive_type_coverage",
        "conditional_on_fitted_policies": True,
    }


def adopt_levers(
    calibration_rows: list[dict],
    dev_rows: list[dict],
    baseline: Policy,
    *,
    levers=(),
    latency=None,
    reasoning_calibration=None,
    reasoning_dev=None,
    routed_latency=None,
    direct_system_latency=None,
    assumed_speed=91.0,
    assumed_cost=56.4,
    usd_in_per_m=0.0403,
    usd_out_per_m=0.0403,
    fitted_router=None,
    fitted_router_sha256=None,
) -> dict:
    """Default requests no levers. No exploratory/public/test escape hatch.

    ``fitted_router`` is the saved ``reasoning_fit.json`` content from the fit step.
    When given, the gate still refits on the same calibration reads, but only to verify
    the artifact within ``ROUTER_REFIT_TOLERANCE``; the saved router is then used byte
    for byte, so platform float differences cannot change the gated policy.
    """
    requested = set(levers)
    if requested - set(GATE_CONSTANTS["lever_order"]) or len(requested) != len(levers):
        raise ValueError("supported unique levers are variant, bias and reasoning_route")
    continuing = bool(baseline.adoption) and requested == {"reasoning_route"}
    if baseline.reasoning_route is not None:
        raise ValueError("reasoning route is already adopted; dev gate retries are refused")
    if continuing:
        validate_adopted_policy(baseline)
    if not continuing and (
        baseline.prompt_variant != "min"
        or baseline.letter_bias
        or baseline.adoption
        or baseline.reasoning_route is not None
        or not baseline.promotable
    ):
        raise ValueError("baseline must be the promotable min policy without new levers")
    if any(not math.isfinite(axis) or axis <= 0 for axis in (assumed_speed, assumed_cost)):
        raise ValueError("assumed Speed/Cost axes must be finite and positive")
    if (
        not math.isfinite(usd_out_per_m)
        or usd_out_per_m < 0
        or (usd_in_per_m is not None and (not math.isfinite(usd_in_per_m) or usd_in_per_m <= 0))
    ):
        raise ValueError("token prices must be finite with positive input and nonnegative output")
    cal, dev = validate_roles(calibration_rows, dev_rows)
    if "variant" in requested and len(cal) < 2:
        raise ValueError("variant lever needs at least one alternative prompt")
    speeds = resource_speeds(dev, latency, assumed_speed)
    hashes = {
        role: {variant: fingerprint(rows) for variant, rows in groups.items()}
        for role, groups in (("calibration", cal), ("dev", dev))
    }
    if continuing and baseline.adoption["data_hashes"] != hashes:
        raise ValueError("continuing adoption requires the original exact calibration/dev data")
    current = replace(baseline)
    policies = {"min": asdict(baseline)}
    reports = []
    accepted = list(baseline.adoption["adopted_levers"]) if continuing else []
    for lever in GATE_CONSTANTS["lever_order"]:
        before_hash = fingerprint(asdict(current))
        entry = {
            "lever": lever,
            "requested": lever in requested,
            "current_before_sha256": before_hash,
            "data_hashes": hashes,
        }
        if lever not in requested:
            entry.update(
                fitted_params=None,
                comparison=None,
                gate={"adopted": False, "reason": "not_requested"},
                current_after_sha256=before_hash,
            )
            reports.append(entry)
            continue
        if lever == "variant":
            candidates = {"min": current}
            cal_scores = {}
            for variant in GATE_CONSTANTS["variant_order"]:
                if variant not in cal:
                    continue
                # Dev timing is gate evidence only; it must never influence fitting.
                axes = resource_axes(
                    cal[variant], assumed_speed, assumed_cost, usd_in_per_m, usd_out_per_m
                )
                if variant != "min":
                    candidates[variant] = replace(
                        fit_policy(
                            cal[variant],
                            fitted_on="bound calibration (adoption variant)",
                            speed_axis=axes["S"],
                            cost_axis=axes["Cost"],
                        ),
                        promotable=False,
                    )
                score = score_reads(cal[variant], candidates[variant])
                cal_scores[variant] = composite(score["I"], score["C"], axes["S"], axes["Cost"])
            selected = max(cal_scores, key=cal_scores.__getitem__)  # insertion order fixes ties
            proposal = candidates[selected]
            policies = {variant: asdict(policy) for variant, policy in candidates.items()}
            entry["fitted_params"] = {
                "selected_on_calibration": selected,
                "calibration_A": cal_scores,
                "policies": policies,
            }
        elif lever == "bias":
            params = fit_letter_bias(cal[current.prompt_variant], current)
            proposal = replace(current, letter_bias=params["letter_bias"] or None, promotable=False)
            entry["fitted_params"] = params
        else:
            variant = current.prompt_variant
            cal_paired = validate_pairs(cal[variant], reasoning_calibration or [])
            dev_paired = validate_pairs(dev[variant], reasoning_dev or [])
            if (
                cal_paired
                and dev_paired
                and next(iter(cal_paired.values()))["recipe"]
                != next(iter(dev_paired.values()))["recipe"]
            ):
                raise ValueError("calibration/dev reasoning recipes must match")
            if not cal_paired or not dev_paired:
                entry.update(
                    fitted_params=None,
                    comparison=None,
                    gate={"adopted": False, "reason": "no_paired_candidates"},
                    current_after_sha256=before_hash,
                )
                reports.append(entry)
                continue
            params = fit_router(
                cal[variant],
                cal_paired,
                current,
                usd_in_per_m=usd_in_per_m,
                usd_out_per_m=usd_out_per_m,
                assumed_cost=assumed_cost,
            )
            if fitted_router is not None:
                if not isinstance(fitted_router_sha256, str) or len(fitted_router_sha256) != 64:
                    raise ValueError("a saved router artifact needs its exact byte sha256")
                difference = router_refit_difference(fitted_router["router"], params["router"])
                if difference > ROUTER_REFIT_TOLERANCE:
                    raise ValueError(
                        f"saved router differs from the calibration refit by {difference:.3g}"
                    )
                # Only the verified router bytes are substituted; every other fitted field
                # (role, counts, calibration statistics) stays the independent recomputation.
                params = {
                    **params,
                    "router": fitted_router["router"],
                    "artifact": {
                        "sha256": fitted_router_sha256,
                        "refit_max_abs_difference": difference,
                        "tolerance": ROUTER_REFIT_TOLERANCE,
                    },
                }
            proposal = replace(current, reasoning_route=params["router"], promotable=False)
            entry["fitted_params"] = params
            if params["calibration"]["route_rate"] == 0:
                entry.update(
                    comparison=None,
                    gate={"adopted": False, "reason": "calibration_selected_off"},
                    current_after_sha256=before_hash,
                )
                reports.append(entry)
                continue
            after_rows, flags = routed_rows(dev[variant], params["router"], dev_paired)
            projected_before = projected_latency(dev[variant], [False] * len(flags), dev_paired)
            projected_after = projected_latency(dev[variant], flags, dev_paired)
            before_speed, after_speed = projected_before["S"], projected_after["S"]
            speed_source = "projected_assumed"
            if routed_latency:
                if (
                    len(routed_latency) != 1
                    or routed_latency[0].get("system") != "reasoning_route"
                    or routed_latency[0].get("router_sha256") != fingerprint(params["router"])
                ):
                    raise ValueError("routed latency must bind the fitted router")
                if routed_latency[0].get("policy_sha256") != policy_fingerprint(proposal):
                    raise ValueError("routed latency must bind the fitted readout policy")
                after_speed = resource_speeds(
                    {variant: dev[variant]}, routed_latency, assumed_speed
                )[variant]
                if latency:
                    before_speed = speeds[variant]
                if direct_system_latency:
                    if direct_system_latency[0].get("system") != "direct" or direct_system_latency[
                        0
                    ].get("policy_sha256") != policy_fingerprint(current):
                        raise ValueError("direct system latency must bind the accepted policy")
                    before_speed = resource_speeds(
                        {variant: dev[variant]}, direct_system_latency, assumed_speed
                    )[variant]
                    a, b = direct_system_latency[0], routed_latency[0]
                    if (a.get("seed"), a.get("samples") and [s["id"] for s in a["samples"]]) != (
                        b.get("seed"),
                        b.get("samples") and [s["id"] for s in b["samples"]],
                    ):
                        raise ValueError(
                            "direct/routed system probes must use the same ordered sample"
                        )
                for probe in [*(direct_system_latency or []), *routed_latency]:
                    ids = [s["id"] for s in probe.get("samples", [])]
                    allowed = {r["id"] for r in dev[variant] if r["tier"] in ("standard", "judge")}
                    if (
                        len(ids) != probe["completed_reads"]
                        or len(ids) != len(set(ids))
                        or set(ids) - allowed
                    ):
                        raise ValueError(
                            "system latency samples must bind distinct non-public dev standard/judge items"
                        )
                speed_source = "serial_non_public_dev_probe"
            comparison = paired_comparison(
                dev[variant],
                current,
                after_rows,
                proposal,
                before_speed=before_speed,
                after_speed=after_speed,
                assumed_cost=assumed_cost,
                usd_in_per_m=usd_in_per_m,
                usd_out_per_m=usd_out_per_m,
            )
            comparison.update(
                lever=lever,
                projected_adjusted_p95_before=projected_before["adjusted_p95_s"],
                projected_adjusted_p95_after=projected_after["adjusted_p95_s"],
                latency={
                    "before": projected_before,
                    "after": projected_after,
                    "speed_source": speed_source,
                    "assumed": not bool(routed_latency and (direct_system_latency or latency)),
                },
            )
            entry["reasoned_data_hashes"] = {
                "calibration": fingerprint(reasoning_calibration),
                "dev": fingerprint(reasoning_dev),
            }
            hashes["reasoned"] = entry["reasoned_data_hashes"]
            decision = gate_decision(comparison)
            if decision["adopted"]:
                current = replace(proposal, promotable=True)
                accepted.append(lever)
            entry.update(
                comparison=comparison,
                gate=decision,
                current_after_sha256=fingerprint(asdict(current)),
            )
            reports.append(entry)
            continue
        comparison = paired_comparison(
            dev[current.prompt_variant],
            current,
            dev[proposal.prompt_variant],
            proposal,
            before_speed=speeds[current.prompt_variant],
            after_speed=speeds[proposal.prompt_variant],
            assumed_cost=assumed_cost,
            usd_in_per_m=usd_in_per_m,
            usd_out_per_m=usd_out_per_m,
        )
        decision = gate_decision(comparison)
        if decision["adopted"]:
            current = replace(proposal, promotable=True)
            accepted.append(lever)
        entry.update(
            comparison=comparison, gate=decision, current_after_sha256=fingerprint(asdict(current))
        )
        reports.append(entry)
    if accepted:
        current.adoption = {
            "contract": "swift_adoption_v1",
            "constants": deepcopy(GATE_CONSTANTS),
            "model": dev["min"][0]["model"],
            "revision": dev["min"][0]["revision"],
            "data_roles": {"fit": "non_public_calibration", "gate": "non_public_dev"},
            "data_hashes": hashes,
            "adopted_levers": accepted,
            "comparisons": {
                **(baseline.adoption["comparisons"] if continuing else {}),
                **{
                    entry["lever"]: entry["comparison"]
                    for entry in reports
                    if entry["gate"]["adopted"]
                },
            },
            "policy_sha256": policy_fingerprint(current),
        }
        validate_adopted_policy(current)
    return {
        "constants": deepcopy(GATE_CONSTANTS),
        "promotable": True,
        "model": dev["min"][0]["model"],
        "revision": dev["min"][0]["revision"],
        "data_roles": {"fit": "non_public_calibration", "gate": "non_public_dev"},
        "data_hashes": hashes,
        "baseline_policy_sha256": fingerprint(asdict(baseline)),
        "resource_sources": {
            "speed": "serial_non_public_dev_probe" if latency else "assumed",
            "cost": "measured_tokens" if usd_in_per_m is not None else "assumed",
        },
        "fit_resource_sources": {
            "speed": "assumed",
            "cost": "measured_calibration_tokens" if usd_in_per_m is not None else "assumed",
        },
        "assumed_axes": {"S": assumed_speed, "Cost": assumed_cost},
        "token_prices": {"input_per_m": usd_in_per_m, "output_per_m": usd_out_per_m},
        "latency_sha256": fingerprint(latency) if latency else None,
        "routed_latency_sha256": fingerprint(routed_latency) if routed_latency else None,
        "reasoning_speed": next(
            (
                entry["comparison"]["latency"]
                for entry in reports
                if entry["lever"] == "reasoning_route" and entry["comparison"]
            ),
            None,
        ),
        "direct_system_latency_sha256": fingerprint(direct_system_latency)
        if direct_system_latency
        else None,
        "levers": reports,
        "adopted_levers": accepted,
        "variant_policies": policies,
        "final_policy": asdict(current),
    }
