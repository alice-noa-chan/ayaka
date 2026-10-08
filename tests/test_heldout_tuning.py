"""Policy composition, calibration rescoring and latency accounting of held-out tuning."""

import pytest

from ayaka.eval import heldout_tuning as tuning
from ayaka.training.path_calibration import PathCalibration


def row(i, kind="noul", route="direct", probs=(0.3, 0.7), latency=0.5, budget=0):
    return {
        "id": f"q{i}",
        "type": kind,
        "route": route,
        "budget": budget,
        "probs": list(probs),
        "target": [0.0, 1.0],
        "ordinals": None,
        "latency_s": latency,
        "routing_features": [0.0] * 9,
    }


def test_compose_switches_rows_and_charges_both_reads():
    direct = [row(1, latency=0.2), row(2, kind="choice", latency=0.3)]
    reasoned = [row(i, route="reasoned", latency=10.0, budget=384) for i in (1, 2)]
    out = tuning.compose(direct, reasoned, lambda r: r["type"] == "noul")
    assert [r["route"] for r in out] == ["reasoned", "direct"]
    assert out[0]["latency_s"] == pytest.approx(10.2)
    assert out[1]["latency_s"] == pytest.approx(0.3)


def test_compose_rejects_misaligned_reads():
    with pytest.raises(ValueError, match="same questions"):
        tuning.compose([row(1)], [row(2, route="reasoned")], lambda r: True)


def test_calibrated_rescores_with_path_temperature():
    calibration = PathCalibration({"noul/reasoned/medium": 2.0})
    reasoned = row(1, route="reasoned", probs=(0.1, 0.9), budget=384)
    out = tuning.calibrated([reasoned], calibration)[0]
    # T=2 softens 0.9 to sqrt-odds: 3 / (1 + 3) = 0.75, inside the 0.2/0.8 band.
    assert out["probs"][1] == pytest.approx(0.75)
    assert out["abstained"] is True
    untouched = tuning.calibrated([row(2, probs=(0.1, 0.9))], calibration)[0]
    assert untouched["probs"][1] == pytest.approx(0.9)


def test_latency_summary_uses_rank_quantiles():
    rows = [row(i, latency=float(i)) for i in range(1, 21)]
    summary = tuning.latency_summary(rows)
    assert summary["p50_s"] == 11.0
    assert summary["p95_s"] == 19.0
    assert summary["mean_s"] == pytest.approx(10.5)


def test_type_rules_ignore_the_router_where_they_should():
    class Never:
        penalty = 0.0

        def predict(self, features):
            return -1.0, 0.0

    noul, choice = row(1), row(2, kind="choice")
    assert tuning.policy_rule("noul_always", Never())(noul)
    assert not tuning.policy_rule("noul_always", Never())(choice)
    assert not tuning.policy_rule("router", Never())(noul)
    assert tuning.policy_rule("noul_always+router", Never())(noul)
    assert not tuning.policy_rule("direct", Never())(noul)
