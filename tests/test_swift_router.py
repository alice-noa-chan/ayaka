"""Calibration-only fitting, honest system resources and strict dev gates."""

import copy
from dataclasses import replace

import pytest
from test_swift_bias import bound_synthetic
from test_swift_reasoning import TraceReader, always_router

from ayaka.eval.read_artifact import fingerprint
from ayaka.swift.adopt import (
    adopt_levers,
    gate_decision,
    policy_fingerprint,
    validate_adopted_policy,
)
from ayaka.swift.collect import DatasetItem, collect, load_reads
from ayaka.swift.policy import Policy
from ayaka.swift.prompt import Question
from ayaka.swift.readers import ReadResult, logmass_probs
from ayaka.swift.router import (
    candidate,
    features,
    fit_router,
    projected_latency,
    public_route_diagnostic,
    routed_rows,
    validate_pairs,
)


def paired_rows(tmp_path, direct, masses=None, *, latency=0.02):
    items = []
    for row in direct:
        b = row["binding"]
        items.append(
            DatasetItem(
                row["id"],
                row["source"],
                row["tier"],
                b["state"],
                Question(**b["question"]),
                row["gold"],
                False,
                row.get("gold_distribution"),
                row.get("family"),
                row["case_id"],
                row["split"],
                row["cluster_id"],
                tuple(row["lineage_ids"]),
                row["adapter"],
            )
        )
    results = []
    for values in masses or [list(r["candidate_log_masses"].values()) for r in direct]:
        logs = {chr(65 + j): v for j, v in enumerate(values)}
        results.append(ReadResult(logmass_probs(logs), 10, 1, 0.01, logs))
    reader = TraceReader(results, latency=latency)
    reader.backend, reader.logprobs_mode = "hf", "raw_logits"
    path = tmp_path / (direct[0]["split"] + "-" + direct[0]["prompt_variant"] + "-reasoned.jsonl")
    collect(
        iter(items),
        reader,
        path,
        model="synthetic",
        revision="a" * 40,
        prompt_variant=direct[0]["prompt_variant"],
        reasoned=True,
        direct_reads=direct,
    )
    return load_reads([path])


def route_roles(tmp_path, *, neutral=False, dev_share=False, latency=0.02, tier="hard"):
    groups, pairs = [], []
    for split in ("calibration", "dev"):
        specs, worked = [], []
        for i in range(150):
            kind = ("choice", "noul", "score")[i % 3]
            gold = 0 if kind == "choice" else 1
            rare = i % (10 if dev_share and split == "dev" else 25) == 0
            base = [0, 1] if rare else [2, 0]
            if gold == 1:
                base = base[::-1]
            specs.append((kind, base, gold))
            good = [3, 0] if gold == 0 else [0, 3]
            worse = [1.9, 0] if gold == 0 else [0, 1.9]
            worked.append(base if neutral else good if rare else worse)
        direct = bound_synthetic(tmp_path, split, "min", specs, tier=tier)
        paired = paired_rows(tmp_path, direct, worked, latency=latency)
        groups.append(direct)
        pairs.append(paired)
    return *groups, *pairs


def test_feature_contract_and_candidate_uses_only_direct_and_text():
    x = features(
        "choice",
        {"a": 0.7, "b": 0.3},
        {"date": "2026-10-04", "cash": "$5"},
        "more than 10 percent before",
    )
    assert x[:5] == [1, 0, 0, 2, 0.7]
    assert x[7:11] == [11, 1, 2, 3]
    assert candidate({"a": 0.95, "b": 0.05}, "plain", "")
    assert not candidate({"a": 0.96, "b": 0.04}, "plain", "")
    assert candidate({"a": 0.99, "b": 0.01}, "plain", "by 2026-10-04")
    assert not candidate(dict.fromkeys(map(str, range(27)), 1 / 27), "12", "")


def test_fit_threshold_maximises_calibration_composite_under_overall_cap(tmp_path):
    cal, dev, cp, dp = route_roles(tmp_path)
    paired = validate_pairs(cal, cp)
    fit = fit_router(cal, paired, Policy())
    assert fit["positive_n"] == 6 and fit["fit_n"] == 150
    assert fit["calibration"]["route_rate"] == 0.04
    assert all(t["route_rate"] <= 0.1 for t in fit["threshold_trials"])
    assert fit["calibration"]["A"] == max(t["A"] for t in fit["threshold_trials"])
    system, flags = routed_rows(cal, fit["router"], paired)
    assert sum(flags) == 6
    assert system[0]["output_tokens"] == 5 and system[0]["input_tokens"] == 121
    assert fit == fit_router(cal, paired, Policy())
    with pytest.raises(ValueError, match="calibration only"):
        fit_router(dev, validate_pairs(dev, dp), Policy())


def test_helpful_rare_route_passes_gate_and_receipt_serves(tmp_path):
    cal, dev, cp, dp = route_roles(tmp_path)
    report = adopt_levers(
        cal, dev, Policy(), levers=("reasoning_route",), reasoning_calibration=cp, reasoning_dev=dp
    )
    entry = report["levers"][2]
    assert entry["gate"]["adopted"] and entry["comparison"]["ci_95_A"][0] > 0
    assert entry["comparison"]["latency"]["assumed"] is True
    assert entry["comparison"]["after"]["Cost"] < entry["comparison"]["before"]["Cost"]
    assert report["adopted_levers"] == ["reasoning_route"]
    policy = Policy(**report["final_policy"])
    validate_adopted_policy(policy)
    policy.save(tmp_path / "policy.json")
    assert Policy.load(tmp_path / "policy.json") == policy
    from ayaka.swift.server import DecisionService

    service = DecisionService(TraceReader(), "synthetic", policy)
    try:
        output = service.handle(
            {
                "state": "calculation 1",
                "questions": {"q": {"type": "choice", "criteria": ["first", "second"]}},
            }
        )
        assert "PRIVATE" not in str(output)
    finally:
        service.close()
    broken = copy.deepcopy(policy)
    broken.adoption["comparisons"]["reasoning_route"].pop("lever")
    with pytest.raises(ValueError, match="p95 guard"):
        validate_adopted_policy(broken)
    with pytest.raises(ValueError, match="adoption gate"):
        validate_adopted_policy(
            replace(policy, reasoning_route={**policy.reasoning_route, "threshold": 0})
        )
    with pytest.raises(ValueError, match="already adopted"):
        adopt_levers(
            cal,
            dev,
            policy,
            levers=("reasoning_route",),
            reasoning_calibration=cp,
            reasoning_dev=dp,
        )


@pytest.mark.parametrize("problem", ["neutral", "p95"])
def test_gate_keeps_reasoning_off_when_not_better_or_dev_p95_worsens(tmp_path, problem):
    cal, dev, cp, dp = route_roles(
        tmp_path,
        neutral=problem == "neutral",
        dev_share=problem == "p95",
        latency=10 if problem == "p95" else 0.02,
    )
    report = adopt_levers(
        cal, dev, Policy(), levers=("reasoning_route",), reasoning_calibration=cp, reasoning_dev=dp
    )
    entry = report["levers"][2]
    assert not entry["gate"]["adopted"]
    assert report["final_policy"]["reasoning_route"] is None
    assert ("projected_p95_guard" if problem == "p95" else "calibration_selected_off") in entry[
        "gate"
    ]["reason"]


def test_p95_guard_requires_projected_values_even_if_other_gates_pass():
    comparison = {
        "lever": "reasoning_route",
        "ci_95_A": [1, 2],
        "ci_95_I": [1, 2],
        "delta_I": 1,
        "per_type_CC_delta": dict.fromkeys(("choice", "noul", "score"), 0),
        "projected_adjusted_p95_before": 1,
        "projected_adjusted_p95_after": 1.1,
    }
    assert gate_decision(comparison)["adopted"]
    assert not gate_decision({**comparison, "projected_adjusted_p95_after": 1.10001})["adopted"]


@pytest.mark.parametrize("problem", ["binding", "usage", "budget", "logits", "missing"])
def test_pair_validation_refuses_wrong_direct_or_incomplete_reasoned_provenance(tmp_path, problem):
    direct = bound_synthetic(tmp_path, "calibration", "min", [("choice", [0, 1], 0)])
    rows = paired_rows(tmp_path, direct)
    nested = rows[0]["reasoned_read"]
    if problem == "binding":
        nested["direct_record_sha256"] = "wrong"
    elif problem == "usage":
        nested["output_tokens"] = 1
    elif problem == "budget":
        nested["recipe"]["max_tokens"] = 2
    elif problem == "logits":
        nested["pass_inputs"][0]["token_logits"]["65"] = 30
    else:
        rows = []
    for row in rows:
        row["record_sha256"] = fingerprint({k: v for k, v in row.items() if k != "record_sha256"})
    with pytest.raises(ValueError):
        validate_pairs(direct, rows)


def test_public_route_report_is_a_standalone_diagnostic(tmp_path):
    rows = bound_synthetic(tmp_path, "calibration", "min", [("choice", [0, 1], 0)] * 3)
    public = [
        {**r, "public": True, "tier": tier}
        for r, tier in zip(rows, ("standard", "judge", "hard"), strict=True)
    ]
    report = public_route_diagnostic(public, Policy(reasoning_route=always_router()))
    assert report == {
        "role": "DIAGNOSTIC_ONLY",
        "used_for_fit_or_gates": False,
        "public": True,
        "tiers": ["standard", "judge"],
        "n": 2,
        "routed_n": 2,
        "route_rate": 1,
    }


def test_projected_mixture_uses_full_routed_cost_and_adjustment_once():
    rows = [{"id": str(i), "latency_s": 0.1} for i in range(100)]
    pair = {r["id"]: {"latency_s": 1} for r in rows}
    rare = projected_latency(rows, [i < 4 for i in range(100)], pair)
    common = projected_latency(rows, [i < 10 for i in range(100)], pair)
    assert rare["p95_s"] == 0.1
    assert common["p95_s"] == 1.1
    assert common["adjusted_p95_s"] == pytest.approx(2.35)


def test_projection_uses_standard_judge_share_without_public_data():
    rows = [
        {"id": str(i), "latency_s": 0.1, "tier": "hard" if i < 10 else "standard"}
        for i in range(100)
    ]
    pair = {r["id"]: {"latency_s": 10} for r in rows}
    projected = projected_latency(rows, [i < 10 for i in range(100)], pair)
    assert projected["overall_route_rate"] == 0.1
    assert projected["route_rate"] == 0
    assert projected["p95_s"] == 0.1
    assert projected["tier_scope"] == "non_public_standard_judge"


def test_projection_does_not_double_count_slow_direct_reads_that_route():
    rows = [{"id": str(i), "latency_s": 5 if i < 4 else 0.1} for i in range(100)]
    paired = {r["id"]: {"latency_s": 1} for r in rows}
    rare = projected_latency(rows, [i < 4 for i in range(100)], paired)
    assert rare["p95_s"] == 0.1
    all_routed = projected_latency(rows, [True] * 100, paired)
    assert all_routed["p95_s"] == 1.1


def test_serial_system_probe_binding_and_measured_speed(tmp_path):
    from ayaka.swift.score import speed_axis

    cal, dev, cp, dp = route_roles(tmp_path, tier="standard")
    params = fit_router(cal, validate_pairs(cal, cp), Policy())
    candidate_policy = Policy(reasoning_route=params["router"], promotable=False)
    direct = {
        "model": "synthetic",
        "revision": "a" * 40,
        "prompt_variant": "min",
        "system": "direct",
        "public": False,
        "split": "dev",
        "complete": True,
        "concurrency": 1,
        "units": "seconds",
        "requested_reads": 3,
        "completed_reads": 3,
        "seed": 15,
        "samples": [{"id": row["id"]} for row in dev[:3]],
        "p50_s": 0.03,
        "p95_s": 0.05,
        "policy_sha256": policy_fingerprint(Policy()),
    }
    routed = {
        **direct,
        "system": "reasoning_route",
        "router_sha256": fingerprint(params["router"]),
        "policy_sha256": policy_fingerprint(candidate_policy),
    }
    report = adopt_levers(
        cal,
        dev,
        Policy(),
        levers=("reasoning_route",),
        reasoning_calibration=cp,
        reasoning_dev=dp,
        direct_system_latency=[direct],
        routed_latency=[routed],
    )
    comparison = report["levers"][2]["comparison"]
    assert comparison["after"]["S"] == speed_axis(0.03, 0.05)
    assert comparison["latency"]["assumed"] is False
    for key, value in (
        ("public", True),
        ("router_sha256", "wrong"),
        ("policy_sha256", "wrong"),
        ("samples", [{"id": "other"}]),
    ):
        with pytest.raises(ValueError):
            adopt_levers(
                cal,
                dev,
                Policy(),
                levers=("reasoning_route",),
                reasoning_calibration=cp,
                reasoning_dev=dp,
                direct_system_latency=[direct],
                routed_latency=[{**routed, key: value}],
            )


def test_dev_loss_regression_cannot_change_calibration_fit(tmp_path):
    cal, dev, cp, _ = route_roles(tmp_path)
    changed = tmp_path / "bad-dev"
    changed.mkdir()
    dp = paired_rows(changed, dev)
    expected = fit_router(cal, validate_pairs(cal, cp), Policy())
    report = adopt_levers(
        cal, dev, Policy(), levers=("reasoning_route",), reasoning_calibration=cp, reasoning_dev=dp
    )
    assert report["levers"][2]["fitted_params"] == expected
    assert not report["levers"][2]["gate"]["adopted"]
    assert report["final_policy"]["reasoning_route"] is None


def test_reasoning_continuation_preserves_accepted_variant_without_dev_retry(tmp_path, monkeypatch):
    from test_swift_adopt import roles

    from ayaka.swift import adopt

    cal, dev = roles(tmp_path)
    prior = adopt_levers(cal, dev, Policy(), levers=("variant",))
    baseline = Policy(**prior["final_policy"])
    assert baseline.prompt_variant == "labeled"
    cp = paired_rows(tmp_path, [r for r in cal if r["prompt_variant"] == "labeled"])
    dp = paired_rows(tmp_path, [r for r in dev if r["prompt_variant"] == "labeled"])
    monkeypatch.setattr(
        adopt, "paired_comparison", lambda *a, **k: pytest.fail("old dev gate retried")
    )
    report = adopt_levers(
        cal, dev, baseline, levers=("reasoning_route",), reasoning_calibration=cp, reasoning_dev=dp
    )
    assert report["final_policy"] == prior["final_policy"]
    validate_adopted_policy(Policy(**report["final_policy"]))


def test_no_candidates_emits_disabled_fit_for_runner(tmp_path):
    from ayaka.swift.collect import adapt_jevbench

    items = [
        adapt_jevbench(
            {
                "id": kind,
                "state": "plain",
                "public": False,
                "source": "private",
                "split": "calibration",
                "labels": labels,
                "expected": labels[0],
                "question": {"type": kind, "criteria": dict.fromkeys(labels, "option")},
            }
        )
        for kind, labels in (
            ("choice", ["a", "b"]),
            ("noul", ["false", "true"]),
            ("score", ["0", "1"]),
        )
    ]
    reader = TraceReader([{"A": 0.99, "B": 0.01}] * 3)
    reader.backend, reader.logprobs_mode = "hf", "raw_logits"
    path = tmp_path / "direct.jsonl"
    collect(iter(items), reader, path, model="synthetic", revision="a" * 40)
    result = fit_router(load_reads([path]), {}, Policy())
    assert result["fit_n"] == 0 and result["router"]["threshold"] == 2
    assert result["calibration"]["route_rate"] == 0
    assert not reader.trace_calls


def test_saved_router_artifact_is_used_when_refit_agrees_within_tolerance(tmp_path):
    from ayaka.swift.router import ROUTER_REFIT_TOLERANCE, router_refit_difference

    cal, dev, cp, dp = route_roles(tmp_path)
    reference = adopt_levers(
        cal, dev, Policy(), levers=("reasoning_route",), reasoning_calibration=cp, reasoning_dev=dp
    )
    fitted = reference["levers"][2]["fitted_params"]
    saved = copy.deepcopy(fitted)
    # A cross-platform refit differs only in the last float bits.
    saved["router"]["weights"][0] += 1e-16
    report = adopt_levers(
        cal,
        dev,
        Policy(),
        levers=("reasoning_route",),
        reasoning_calibration=cp,
        reasoning_dev=dp,
        fitted_router=saved,
        fitted_router_sha256="a" * 64,
    )
    entry = report["levers"][2]
    assert entry["fitted_params"]["router"] == saved["router"]
    assert entry["fitted_params"]["artifact"]["sha256"] == "a" * 64
    assert entry["fitted_params"]["artifact"]["refit_max_abs_difference"] <= ROUTER_REFIT_TOLERANCE
    assert Policy(**report["final_policy"]).reasoning_route == saved["router"]

    drifted = copy.deepcopy(fitted)
    drifted["router"]["weights"][0] += 1e-6
    with pytest.raises(ValueError, match="differs from the calibration refit"):
        adopt_levers(
            cal,
            dev,
            Policy(),
            levers=("reasoning_route",),
            reasoning_calibration=cp,
            reasoning_dev=dp,
            fitted_router=drifted,
            fitted_router_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="exact byte sha256"):
        adopt_levers(
            cal,
            dev,
            Policy(),
            levers=("reasoning_route",),
            reasoning_calibration=cp,
            reasoning_dev=dp,
            fitted_router=saved,
        )
    moved = copy.deepcopy(fitted["router"])
    moved["threshold"] = min(2, moved["threshold"] + 0.5)
    with pytest.raises(ValueError, match="threshold"):
        router_refit_difference(moved, fitted["router"])


def test_saved_artifact_contributes_only_its_router(tmp_path):
    cal, dev, cp, dp = route_roles(tmp_path)
    reference = adopt_levers(
        cal, dev, Policy(), levers=("reasoning_route",), reasoning_calibration=cp, reasoning_dev=dp
    )
    fitted = reference["levers"][2]["fitted_params"]
    contradictory = copy.deepcopy(fitted)
    # Identical router, contradictory metadata: it must not switch the route off or
    # enter the receipt.
    contradictory["calibration"]["route_rate"] = 0
    contradictory["fit_n"] = -1
    report = adopt_levers(
        cal,
        dev,
        Policy(),
        levers=("reasoning_route",),
        reasoning_calibration=cp,
        reasoning_dev=dp,
        fitted_router=contradictory,
        fitted_router_sha256="b" * 64,
    )
    entry = report["levers"][2]
    assert entry["gate"]["adopted"] == reference["levers"][2]["gate"]["adopted"]
    assert entry["fitted_params"]["calibration"] == fitted["calibration"]
    assert entry["fitted_params"].get("fit_n") == fitted.get("fit_n")
    assert entry["fitted_params"]["router"] == fitted["router"]
