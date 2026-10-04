"""Predeclared gates, strict provenance, sequential fitting and serving receipts."""

import copy
import json
from dataclasses import asdict, replace

import pytest
from test_swift_bias import bound_synthetic

from ayaka.swift import adopt
from ayaka.swift.adopt import GATE_CONSTANTS, adopt_levers, gate_decision, validate_adopted_policy
from ayaka.swift.policy import Policy
from ayaka.swift.readers import FakeReader
from ayaka.swift.score import composite, cost_score, speed_axis
from ayaka.swift.server import DecisionService
from scripts.swift.adopt import main


def roles(tmp_path, *, mode="help", n=6):
    cal, dev = [], []
    for split in ("calibration", "dev"):
        for variant in ("min", "labeled"):
            specs = []
            for case in range(n):
                baseline = [("choice", [1, 0], 0), ("noul", [0, 2], 1), ("score", [0, 1], 1)]
                candidate = [("choice", [2, 0], 0), ("noul", [0, 3], 1), ("score", [0, 2], 1)]
                if mode == "neutral" and split == "dev":
                    baseline = candidate = [
                        ("choice", [60, 0], 0),
                        ("noul", [0, 60], 1),
                        ("score", [0, 60], 1),
                    ]
                if mode == "noisy" and split == "dev":
                    # Opposite whole-case successes; each system wins half.
                    direction = 60 if case % 2 == 0 else -60
                    baseline = [
                        ("choice", [direction, 0], 0),
                        ("noul", [0, direction], 1),
                        ("score", [0, direction], 1),
                    ]
                    candidate = [
                        ("choice", [-direction, 0], 0),
                        ("noul", [0, -direction], 1),
                        ("score", [0, -direction], 1),
                    ]
                if mode == "hurt_type":
                    baseline = [("choice", [2, 0], 0), ("noul", [0, 1], 1), ("score", [0, 1], 1)]
                    if split == "dev":
                        candidate[0] = ("choice", [0, 2], 0)
                specs.extend(baseline if variant == "min" else candidate)
            (cal if split == "calibration" else dev).extend(
                bound_synthetic(tmp_path, split, variant, specs)
            )
    return cal, dev


def test_variant_helps_is_adopted_and_receipt_is_served(tmp_path):
    cal, dev = roles(tmp_path)
    result = adopt_levers(cal, dev, Policy(), levers=("variant",))
    assert result["adopted_levers"] == ["variant"]
    assert result["constants"] == GATE_CONSTANTS
    lever = result["levers"][0]
    assert lever["fitted_params"]["selected_on_calibration"] == "labeled"
    assert lever["comparison"]["ci_95_A"][0] > 0
    assert lever["comparison"]["B"] == 2000
    assert lever["comparison"]["seed"] == 15
    assert lever["comparison"]["cluster_count"] == 6
    policy = Policy(**result["final_policy"])
    assert policy.prompt_variant == "labeled" and policy.promotable
    assert policy.letter_bias is None
    validate_adopted_policy(policy)
    service = DecisionService(FakeReader(), "synthetic", policy, prompt_variant="labeled")
    service.close()
    with pytest.raises(ValueError, match="serving model"):
        DecisionService(FakeReader(), "different", policy, prompt_variant="labeled")
    wrong_revision = FakeReader()
    wrong_revision.backend = "hf"
    wrong_revision.revision = "b" * 40
    with pytest.raises(ValueError, match="serving reader"):
        DecisionService(wrong_revision, "synthetic", policy, prompt_variant="labeled")


@pytest.mark.parametrize("mode", ["neutral", "noisy"])
def test_neutral_or_noisy_candidate_is_rejected_and_baseline_survives(tmp_path, mode):
    cal, dev = roles(tmp_path, mode=mode)
    baseline = Policy(fitted_on="already-existing fitted policy")
    result = adopt_levers(cal, dev, baseline, levers=("variant",))
    assert result["adopted_levers"] == []
    assert result["final_policy"] == asdict(baseline)
    comparison = result["levers"][0]["comparison"]
    assert comparison["ci_95_A"][0] <= 0 <= comparison["ci_95_A"][1]
    if mode == "noisy":
        assert comparison["ci_95_I"][0] < 0 < comparison["ci_95_I"][1]
    assert "positive_A_lower_bound" in result["levers"][0]["gate"]["reason"]


def test_variant_with_composite_gain_but_one_harmed_primitive_is_rejected(tmp_path):
    cal, dev = roles(tmp_path, mode="hurt_type")
    result = adopt_levers(cal, dev, Policy(), levers=("variant",))
    lever = result["levers"][0]
    assert lever["comparison"]["delta_A"] > 0
    assert lever["comparison"]["per_type_CC_delta"]["choice"] == -200
    assert not lever["gate"]["adopted"]
    assert "primitive_CC_guard" in lever["gate"]["reason"]
    assert result["final_policy"]["prompt_variant"] == "min"


def test_gate_exact_boundaries_and_intelligence_point_guard():
    comparison = {
        "ci_95_A": [0.01, 1],
        "ci_95_I": [-1, 0],
        "delta_I": -0.5,
        "per_type_CC_delta": dict.fromkeys(("choice", "noul", "score"), -2),
    }
    assert gate_decision(comparison)["adopted"]
    for key, value, reason in [
        ("ci_95_A", [0, 1], "positive_A_lower_bound"),
        ("ci_95_I", [-1, -0.001], "I_not_significantly_negative"),
        ("delta_I", -0.500001, "I_point_guard"),
    ]:
        decision = gate_decision({**comparison, key: value})
        assert not decision["adopted"] and reason in decision["reason"]
    bad = {**comparison, "per_type_CC_delta": {"choice": -2.00001, "noul": 0, "score": 0}}
    assert not gate_decision(bad)["adopted"]


def test_default_requests_no_levers_and_does_not_fit(tmp_path, monkeypatch):
    cal, dev = roles(tmp_path)
    monkeypatch.setattr(adopt, "fit_policy", lambda *a, **k: pytest.fail("unexpected fit"))
    monkeypatch.setattr(
        adopt, "fit_letter_bias", lambda *a, **k: pytest.fail("unexpected bias fit")
    )
    baseline = Policy(t_noul=2, commit_margin=0.1, fitted_on="existing fitted baseline")
    result = adopt_levers(cal, dev, baseline)
    assert result["final_policy"] == asdict(baseline)
    assert all(entry["gate"]["reason"] == "not_requested" for entry in result["levers"])
    assert result["promotable"] is True


def test_order_determinism_and_bias_fits_accepted_variant_only(tmp_path, monkeypatch):
    cal, dev = roles(tmp_path, n=30)
    calls = []
    original = adopt.fit_letter_bias

    def track(rows, policy):
        calls.append(
            (
                {row["split"] for row in rows},
                {row["prompt_variant"] for row in rows},
                policy.prompt_variant,
            )
        )
        return original(rows, policy)

    monkeypatch.setattr(adopt, "fit_letter_bias", track)
    result = adopt_levers(cal, dev, Policy(), levers=("bias", "variant"))
    repeated = adopt_levers(
        list(reversed(cal)), list(reversed(dev)), Policy(), levers=("variant", "bias")
    )
    assert result == repeated
    assert [entry["lever"] for entry in result["levers"]] == ["variant", "bias", "reasoning_route"]
    assert calls == [({"calibration"}, {"labeled"}, "labeled")] * 2
    assert (
        result["levers"][1]["current_before_sha256"] == result["levers"][0]["current_after_sha256"]
    )
    # Whether bias passes or fails is measured against the accepted labeled system.
    assert result["levers"][1]["comparison"]["before"]["A"] == pytest.approx(
        result["levers"][0]["comparison"]["after"]["A"]
    )


def test_rejected_variant_leaves_bias_fit_on_min(tmp_path, monkeypatch):
    cal, dev = roles(tmp_path, mode="neutral", n=3)
    original = adopt.fit_letter_bias
    variants = []

    def track(rows, policy):
        variants.append(policy.prompt_variant)
        return original(rows, policy)

    monkeypatch.setattr(adopt, "fit_letter_bias", track)
    result = adopt_levers(cal, dev, Policy(), levers=("variant", "bias"))
    assert result["adopted_levers"] == []
    assert variants == ["min"]
    assert result["final_policy"]["letter_bias"] is None


def test_planted_bias_adopted_then_cli_exports_only_accepted_parameters(tmp_path):
    cal, dev = [], []
    for split, target in (("calibration", cal), ("dev", dev)):
        specs = []
        for index in range(30):
            signal = 1 if index % 2 == 0 else -1
            specs.extend(
                [
                    ("choice", [3 + signal, -signal], index % 2),
                    ("noul", [0, 4], 1),
                    ("score", [0, 4], 1),
                ]
            )
        target.extend(bound_synthetic(tmp_path, split, "min", specs))
    result = adopt_levers(cal, dev, Policy(), levers=("bias",))
    assert result["adopted_levers"] == ["bias"]
    policy = Policy(**result["final_policy"])
    assert policy.letter_bias["choice"]["2"][0] < 0
    validate_adopted_policy(policy)
    baseline = tmp_path / "baseline.json"
    Policy().save(baseline)
    report, output = tmp_path / "adoption.json", tmp_path / "policy.json"
    main(
        [
            "--calibration",
            str(tmp_path / "calibration-min.jsonl"),
            "--dev",
            str(tmp_path / "dev-min.jsonl"),
            "--baseline-policy",
            str(baseline),
            "--levers",
            "bias",
            "--output",
            str(report),
            "--policy",
            str(output),
        ]
    )
    assert json.loads(report.read_text(encoding="utf-8")) == result
    assert Policy.load(output) == policy
    assert result["levers"][1]["comparison"]["per_type_CC_delta"]["choice"] == 100
    service = DecisionService(FakeReader(), "synthetic", policy)
    service.close()


@pytest.mark.parametrize("role", ["calibration", "dev"])
@pytest.mark.parametrize("bad", ["public", "test", "unbound", "overlap", "source_public"])
def test_forbidden_inputs_reject_before_any_fit(tmp_path, monkeypatch, role, bad):
    cal, dev = roles(tmp_path)
    rows = cal if role == "calibration" else dev
    if bad == "public":
        rows[0]["public"] = True
    elif bad == "test":
        rows[0]["split"] = "test"
    elif bad == "unbound":
        rows[0].pop("binding")
    elif bad == "source_public":
        rows[0]["source"] = "data/jevbench_public/hard.jsonl"
    else:
        # Reuse fully valid calibration receipts in the dev role; isolation/role must reject.
        dev = cal
    monkeypatch.setattr(adopt, "fit_policy", lambda *a, **k: pytest.fail("fit before validation"))
    with pytest.raises(ValueError):
        adopt_levers(cal, dev, Policy(), levers=("variant", "bias"))


def probes(variants):
    return [
        {
            "prompt_variant": v,
            "model": "synthetic",
            "revision": "a" * 40,
            "complete": True,
            "concurrency": 1,
            "units": "seconds",
            "public": False,
            "split": "dev",
            "completed_reads": 2,
            "requested_reads": 2,
            "p50_s": 0.03,
            "p95_s": 0.05,
            "seed": 15,
            "samples": [{"id": "dev-0"}, {"id": "dev-1"}],
        }
        for v in variants
    ]


def test_resource_axes_use_measurements_or_record_assumptions(tmp_path):
    cal, dev = roles(tmp_path)
    measured = adopt_levers(
        cal,
        dev,
        Policy(),
        levers=("variant",),
        latency=probes(["min", "labeled"]),
        usd_in_per_m=0.08,
    )
    after = measured["levers"][0]["comparison"]["after"]
    assert after["S"] == speed_axis(0.03, 0.05)
    assert after["Cost"] == cost_score((100 * 0.08 + 0.0403) / 1000)
    assert after["A"] == composite(after["I"], after["C"], after["S"], after["Cost"])
    assert measured["resource_sources"] == {
        "speed": "serial_non_public_dev_probe",
        "cost": "measured_tokens",
    }
    assumed = adopt_levers(
        cal,
        dev,
        Policy(),
        levers=("variant",),
        assumed_speed=80,
        assumed_cost=60,
        usd_in_per_m=None,
    )
    assert assumed["resource_sources"] == {"speed": "assumed", "cost": "assumed"}
    assert assumed["levers"][0]["comparison"]["after"]["S"] == 80
    assert assumed["levers"][0]["comparison"]["after"]["Cost"] == 60


def test_dev_outcomes_and_latency_do_not_change_calibration_fitted_parameters(tmp_path):
    cal, dev = roles(tmp_path, n=2)
    unchanged = adopt_levers(cal, dev, Policy(), levers=("variant",))
    noisy_directory = tmp_path / "changed-dev"
    noisy_directory.mkdir()
    _, noisy_dev = roles(noisy_directory, mode="noisy", n=2)
    latency = probes(["min", "labeled"])
    latency[1].update(p50_s=2, p95_s=3)
    changed = adopt_levers(cal, noisy_dev, Policy(), levers=("variant",), latency=latency)
    assert unchanged["levers"][0]["fitted_params"] == changed["levers"][0]["fitted_params"]
    assert unchanged["data_hashes"]["calibration"] == changed["data_hashes"]["calibration"]
    assert unchanged["data_hashes"]["dev"] != changed["data_hashes"]["dev"]
    assert unchanged["adopted_levers"] == ["variant"]
    assert changed["adopted_levers"] == []


@pytest.mark.parametrize("problem", ["public", "test", "partial", "missing", "wrong_model"])
def test_latency_probes_cannot_bypass_gate_roles(tmp_path, problem):
    cal, dev = roles(tmp_path)
    latency = probes(["min", "labeled"])
    if problem == "public":
        latency[0]["public"] = True
    elif problem == "test":
        latency[0]["split"] = "test"
    elif problem == "partial":
        latency[0]["complete"] = False
    elif problem == "missing":
        latency.pop()
    else:
        latency[0]["model"] = "different"
    with pytest.raises(ValueError):
        adopt_levers(cal, dev, Policy(), levers=("variant",), latency=latency)


def test_ungated_levers_and_modified_receipts_cannot_serve(tmp_path):
    for policy in (
        Policy(prompt_variant="labeled"),
        Policy(letter_bias={"choice": {"2": [1, -1]}}),
    ):
        with pytest.raises(ValueError, match="adoption gate"):
            DecisionService(FakeReader(), "synthetic", policy, prompt_variant=policy.prompt_variant)
    cal, dev = roles(tmp_path)
    policy = Policy(**adopt_levers(cal, dev, Policy(), levers=("variant",))["final_policy"])
    with pytest.raises(ValueError, match="adoption gate"):
        validate_adopted_policy(replace(policy, t_choice=policy.t_choice * 2))
    broken = copy.deepcopy(policy)
    broken.adoption["comparisons"]["variant"]["delta_I"] = -0.6
    with pytest.raises(ValueError, match="failed"):
        validate_adopted_policy(broken)
    wrong_bootstrap = copy.deepcopy(policy)
    wrong_bootstrap.adoption["comparisons"]["variant"]["B"] = 1
    with pytest.raises(ValueError, match="failed"):
        validate_adopted_policy(wrong_bootstrap)
    with pytest.raises(ValueError, match="--diagnostic"):
        DecisionService(FakeReader(), "synthetic", Policy(promotable=False))
    # A deliberate diagnostic serving path remains visibly non-promotable.
    service = DecisionService(
        FakeReader(),
        "synthetic",
        Policy(prompt_variant="labeled"),
        prompt_variant="labeled",
        diagnostic=True,
    )
    assert service.policy.promotable is False
    service.close()


def test_cli_failure_writes_no_output(tmp_path):
    cal, _ = roles(tmp_path)
    baseline = tmp_path / "baseline.json"
    Policy().save(baseline)
    report, policy = tmp_path / "rejected.json", tmp_path / "rejected-policy.json"
    assert cal
    with pytest.raises(SystemExit):
        main(
            [
                "--calibration",
                str(tmp_path / "calibration-min.jsonl"),
                "--dev",
                str(tmp_path / "calibration-min.jsonl"),
                "--baseline-policy",
                str(baseline),
                "--output",
                str(report),
                "--policy",
                str(policy),
            ]
        )
    assert not report.exists() and not policy.exists()


@pytest.mark.parametrize("levers", [("unknown",), ("bias", "bias")])
def test_unsupported_or_duplicate_levers_refused(tmp_path, levers):
    cal, dev = roles(tmp_path, n=1)
    with pytest.raises(ValueError, match="supported unique levers"):
        adopt_levers(cal, dev, Policy(), levers=levers)
