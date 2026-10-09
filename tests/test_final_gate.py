"""Frozen tuning fits and the once-only final_test gate."""

import copy

import pytest

from ayaka.eval import final_gate as fg
from ayaka.eval.v2 import typed_row
from ayaka.routing import BenefitRouter
from ayaka.training.path_calibration import PathCalibration


def router(promoted=True, penalty=0.001):
    return BenefitRouter(
        [0.0] * 9,
        [1.0] * 9,
        [-1.0] + [0.0] * 9,  # predicted gain is negative: never routes
        [0.1] + [0.0] * 9,
        penalty,
        promoted,
        {"validated_budgets": [384]},
    )


def frozen(policy="noul_always+router", **kwargs):
    return fg.frozen_fit(PathCalibration({"noul": 1.0}), router(**kwargs), policy, "merged")


class _Spec:
    def __init__(self, kind, ordinals=None):
        self.type, self.ordinals = kind, ordinals


def row(qid, kind, probs, target, source, route="direct", latency=1.0):
    spec = _Spec(kind, [0, 1, 2] if kind == "score" else None)
    return {
        **typed_row(spec, probs, target),
        "id": qid,
        "cluster_id": qid,
        "source": source,
        "language": "en",
        "tier": "standard",
        "probs": probs,
        "target": target,
        "route": route,
        "budget": 384,
        "latency_s": latency,
        "ordinals": spec.ordinals,
        "routing_features": [0.0] * 8 + [384 / 1024],
    }


def reads(system):
    """v1 abstains on Noul; v2 reasoned decides it correctly."""
    rows = []
    for i in range(240):
        rows.append(row(f"c{i}", "choice", [0.8, 0.2], [1.0, 0.0], "choice_src"))
        sharp = [0.05, 0.95] if system == "v2_on" else [0.5, 0.5]
        route = "reasoned" if system == "v2_on" else "direct"
        rows.append(row(f"n{i}", "noul", sharp, [0.0, 1.0], "noul_src", route=route))
        rows.append(row(f"s{i}", "score", [0.1, 0.8, 0.1], [0.0, 1.0, 0.0], "score_src"))
    return rows


def test_frozen_fit_round_trips_and_rejects_unknown_fields():
    value = frozen()
    assert fg.validate_frozen(copy.deepcopy(value)) == value
    for damage in (
        {"version": "other"},
        {"policy": "always"},
        {"adapter": "quantized"},
        {"extra": 1},
        {"path_temperatures": {"noul": 0.0}},
    ):
        with pytest.raises(ValueError):
            fg.validate_frozen({**value, **damage})


def test_router_policies_require_a_promoted_router():
    with pytest.raises(ValueError):
        frozen(promoted=False)
    assert frozen(policy="noul_always", promoted=False)["policy"] == "noul_always"


def test_final_gate_reads_final_test_once_and_applies_the_frozen_policy(monkeypatch, tmp_path):
    seen = []

    def load_run(results, split, system, adapter="unmerged"):
        seen.append((split, system, adapter))
        return reads("v1_on" if system == "v1_on" else system)

    monkeypatch.setattr(fg, "load_run", load_run)
    report = fg.final_gate(tmp_path, frozen(), replicates=200)
    assert seen == [
        ("final_test", "v2_off", "merged"),
        ("final_test", "v2_on", "merged"),
        ("final_test", "v1_on", "unmerged"),
    ]
    assert report["reasoned_by_type"] == {"noul": 240}  # the router never routes here
    gate = report["gates"]["over_v1_on"]
    assert gate["subgroup_rule"] == "clustered_noninferiority"
    assert gate["cc_delta"] > 5 and gate["screen_passed"]
    assert report["adopted"] is (report["systems"]["policy"]["latency"]["speed_axis"] >= 50)
    assert report["fitted_on_final_test"] is False


def test_slow_policies_are_not_adopted_even_when_the_gate_passes(monkeypatch, tmp_path):
    def load_run(results, split, system, adapter="unmerged"):
        rows = reads(system)
        for r in rows:
            r["latency_s"] = 60.0
        return rows

    monkeypatch.setattr(fg, "load_run", load_run)
    report = fg.final_gate(tmp_path, frozen(), replicates=100)
    assert report["gates"]["over_v1_on"]["screen_passed"]
    assert report["systems"]["policy"]["latency"]["speed_axis"] < 50
    assert report["adopted"] is False


def test_final_gate_refuses_misaligned_v1_reads(monkeypatch, tmp_path):
    def load_run(results, split, system, adapter="unmerged"):
        rows = reads(system)
        return rows[:-3] if system == "v1_on" else rows

    monkeypatch.setattr(fg, "load_run", load_run)
    with pytest.raises(ValueError):
        fg.final_gate(tmp_path, frozen(), replicates=50)
