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


def test_pool_is_a_normalized_log_linear_mix():
    assert tuning.pool([0.2, 0.8], [0.5, 0.5], (1.0, 0.0)) == pytest.approx([0.2, 0.8])
    assert tuning.pool([0.2, 0.8], [0.6, 0.4], (0.0, 1.0)) == pytest.approx([0.6, 0.4])
    # Equal halves of 0.2/0.8 and 0.8/0.2 cancel to a uniform read.
    assert tuning.pool([0.2, 0.8], [0.8, 0.2], (0.5, 0.5)) == pytest.approx([0.5, 0.5])


def test_pooled_replaces_only_the_pooled_type_from_raw_reads():
    direct = [row(1, probs=(0.4, 0.6)), row(2, kind="choice", probs=(0.4, 0.6))]
    raw = [
        row(i, kind=k, route="reasoned", probs=(0.1, 0.9)) for i, k in ((1, "noul"), (2, "choice"))
    ]
    served = [{**r, "probs": [0.3, 0.7]} for r in raw]
    out = tuning.pooled(served, direct, raw, (1.0, 1.0))
    assert out[0]["probs"][1] == pytest.approx(0.6 * 0.9 / (0.6 * 0.9 + 0.4 * 0.1))
    assert out[1]["probs"] == [0.3, 0.7]


def calibration_rows(kind, probs, targets, route="direct", ordinals=None):
    return [
        tuning.rescored(
            {
                **row(i, kind=kind, route=route),
                "ordinals": ordinals,
                "target": list(t),
                "split": "calibration",
            },
            list(p),
        )
        for i, (p, t) in enumerate(zip(probs, targets, strict=True))
    ]


def test_noul_pool_never_buys_nll_with_abstentions():
    # Reasoned reads are decisive and right; direct reads are unsure. The
    # NLL-optimal mix would pull confident reads into the 0.2/0.8 band.
    targets = [(0.0, 1.0)] * 20 + [(1.0, 0.0)] * 20
    reasoned = calibration_rows(
        "noul", [(0.1, 0.9)] * 20 + [(0.9, 0.1)] * 20, targets, route="reasoned"
    )
    direct = calibration_rows("noul", [(0.45, 0.55)] * 20 + [(0.55, 0.45)] * 20, targets)
    weights = tuning.fit_noul_pool(direct, reasoned, reasoned)
    out = tuning.pooled(reasoned, direct, reasoned, weights)
    assert not any(r["abstained"] for r in out)
    assert sum(r["nll"] for r in out) <= sum(r["nll"] for r in reasoned) + 1e-9


def test_lever_fits_refuse_non_calibration_rows():
    rows = calibration_rows("noul", [(0.3, 0.7)] * 2, [(0.0, 1.0)] * 2)
    leaked = [{**rows[0], "split": "dev"}]
    with pytest.raises(ValueError, match="calibration"):
        tuning.fit_noul_pool(leaked, rows, rows)
    with pytest.raises(ValueError, match="calibration"):
        tuning.fit_cc_temperature(leaked, "score", tolerance=0.05)


def test_cc_temperature_respects_the_nll_tolerance():
    # Right-leaning reads with gold mostly on the top position: sharpening lowers
    # the expected-position error, while NLL has an interior optimum.
    probs = [(0.1, 0.3, 0.6)] * 40
    targets = [(0.0, 0.0, 1.0)] * 24 + [(0.0, 1.0, 0.0)] * 16
    rows = calibration_rows("score", probs, targets, ordinals=[0, 1, 2])

    def stats(t):
        out = [tuning.temper(r, r["probs"], t) for r in rows]
        return tuning._type_summary(out, "score")

    loose = tuning.fit_cc_temperature(rows, "score", tolerance=10.0)
    strict = tuning.fit_cc_temperature(rows, "score", tolerance=0.0)
    nll_best = min(tuning.TEMPERATURE_GRID, key=lambda t: stats(t)["nll"])
    assert strict == nll_best
    assert loose < strict
    assert stats(loose)["cc"] > stats(strict)["cc"]


def test_lever_policy_reasons_on_noul_and_choice():
    class Never:
        penalty = 0.0

        def predict(self, features):
            return -1.0, 0.0

    rule = tuning.policy_rule("noul+choice_always+router", Never())
    assert rule(row(1)) and rule(row(2, kind="choice"))
    assert not rule(row(3, kind="score"))
