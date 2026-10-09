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

    def load_run(results, split, system, adapter="unmerged", complete=True):
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
    def load_run(results, split, system, adapter="unmerged", complete=True):
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
    def load_run(results, split, system, adapter="unmerged", complete=True):
        rows = reads(system)
        return rows[:-3] if system == "v1_on" else rows

    monkeypatch.setattr(fg, "load_run", load_run)
    with pytest.raises(ValueError):
        fg.final_gate(tmp_path, frozen(), replicates=50)


def test_route_ids_and_a_reasoned_read_limited_to_them(monkeypatch, tmp_path):
    def load_run(results, split, system, adapter="unmerged", complete=True):
        rows = reads(system)
        if system == "v2_on":
            assert complete is False
            rows = [r for r in rows if r["type"] == "noul"]
        return rows

    monkeypatch.setattr(fg, "load_run", load_run)
    ids = fg.routed_ids(tmp_path, frozen())
    assert ids == sorted(f"n{i}" for i in range(240))
    report = fg.final_gate(tmp_path, frozen(), replicates=100)
    assert report["reasoned_by_type"] == {"noul": 240}


def test_a_routed_question_without_a_reasoned_read_is_an_error(monkeypatch, tmp_path):
    def load_run(results, split, system, adapter="unmerged", complete=True):
        rows = reads(system)
        return [r for r in rows if r["id"] != "n7"] if system == "v2_on" else rows

    monkeypatch.setattr(fg, "load_run", load_run)
    with pytest.raises(ValueError, match="no reasoned read"):
        fg.final_gate(tmp_path, frozen(), replicates=50)


def test_tune_never_chooses_a_routing_policy_without_a_promoted_router(monkeypatch, tmp_path):
    from ayaka.eval import heldout_tuning as ht

    def load_run(results, split, system, adapter="unmerged", complete=True):
        rows = []
        for r in reads(system):
            if r["type"] == "choice":
                # Reasoning fixes every Choice question, so routing everything wins.
                good = system == "v2_on"
                r = row(r["id"], "choice", [0.9, 0.1] if good else [0.4, 0.6], [1.0, 0.0], "c")
            rows.append({**r, "split": split})
        return sorted(rows, key=lambda r: r["id"])

    always = router(promoted=False, penalty=0.0)
    always.gain_weights = [1.0] + [0.0] * 9  # predicts a gain everywhere: routes everything
    monkeypatch.setattr(ht, "load_run", load_run)
    monkeypatch.setattr(
        ht.PathCalibration, "fit", classmethod(lambda cls, rows: cls({"noul": 1.0}))
    )
    monkeypatch.setattr(ht, "paired_training_rows", lambda *args: [])
    monkeypatch.setattr(ht, "fit_router", lambda *args: always)
    monkeypatch.setattr(ht, "lever_section", lambda *args, **kwargs: {})
    report = ht.tune(tmp_path, replicates=50)
    cc = {n: v["router_train"]["cc_equal_types"] for n, v in report["policies"].items()}
    assert cc["router"] > cc["noul_always"]  # a routing policy scores best ...
    # ... but its router is unpromoted, so serving could not route and it is not chosen.
    assert report["policy_chosen_on_router_train"] == "noul_always"
    assert report["frozen"]["policy"] == "noul_always"
