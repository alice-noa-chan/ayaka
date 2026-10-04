import math

import pytest

from ayaka.swift.score import (
    calibration,
    choice_cc,
    composite,
    cost_score,
    decision_cost,
    ece,
    intelligence,
    normalized_rps,
    noul_cc,
    prepare_rows,
    score_cc,
    score_reads,
    speed_axis,
    speed_score,
)


def row(kind, probs, gold, tier="standard", **extras):
    return {
        "type": kind,
        "tier": tier,
        "labels": list(probs),
        "raw_probs": probs,
        "gold": gold,
        **extras,
    }


def test_choice_cc_variable_option_chance():
    items = prepare_rows(
        [
            row("choice", {"a": 0.6, "b": 0.4}, "a"),
            row("choice", {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}, "b"),
        ]
    )
    # acc=1/2, cbar=(1/2+1/4)/2=3/8 -> CC=20.
    assert choice_cc(items) == pytest.approx(20)


def test_noul_cc_abstention_and_inclusive_endpoints():
    items = prepare_rows(
        [
            row("noul", {"false": 1 - p, "true": p}, gold)
            for p, gold in [(0.2, "false"), (0.8, "true"), (0.5, "true"), (0.1, "true")]
        ]
    )
    assert noul_cc(items) == 0  # Two correct, two wrong.
    assert noul_cc(items[2:3]) == -100
    assert noul_cc(items[:2]) == 100


def test_score_cc_mean_normalized_error_over_mean_chance():
    items = prepare_rows(
        [
            row("score", {"0": 0.5, "1": 0.25, "2": 0.25}, "0"),
            row("score", {"0": 0, "1": 0.25, "2": 0.75}, "1"),
        ]
    )
    # Errors .375 each; chance .5 and 1/3, mean 5/12 -> CC=10.
    assert score_cc(items) == pytest.approx(10)
    # Scoring uses ordinal positions even when response labels have gaps.
    assert score_cc(prepare_rows([row("score", {"24": 0.5, "28": 0.25, "31": 0.25}, "24")])) == 25


def test_tier_weights_renormalized_and_type_weights_equal():
    rows = []
    for kind, probabilities, good, bad in [
        ("choice", {"a": 1, "b": 0}, "a", "b"),
        ("noul", {"false": 0, "true": 1}, "true", "false"),
        ("score", {"0": 1, "1": 0}, "0", "1"),
    ]:
        rows += [row(kind, probabilities, good, "easy"), row(kind, probabilities, bad, "hard")]
    report = intelligence(prepare_rows(rows))
    assert report["per_type_CC"] == pytest.approx({"choice": -60, "noul": -60, "score": -60})
    assert report["I"] == pytest.approx(-60)
    all_tiers = [
        row("choice", {"a": 1, "b": 0}, gold, tier)
        for tier, gold in [("easy", "a"), ("standard", "b"), ("judge", "a"), ("hard", "b")]
    ]
    assert intelligence(prepare_rows(all_tiers))["per_type_CC"]["choice"] == pytest.approx(-20)
    all_tiers += [row("noul", {"false": 0, "true": 1}, "true"), row("score", {"0": 1, "1": 0}, "0")]
    assert score_reads(all_tiers)["I"] == pytest.approx(60)


def test_ece_equal_width_bins_endpoints_and_cancellation():
    assert ece([0.1, 0.19, 0.2, 1], [0, 1, 1, 1]) == pytest.approx((abs(0.29 - 1) + 0.8 + 0) / 4)
    assert ece([0.51, 0.59], [0, 1]) == pytest.approx(0.05)


def test_choice_calibration_soft_tvd_and_fallback():
    soft = row("choice", {"a": 0.8, "b": 0.2}, "a", gold_distribution={"a": 0.6, "b": 0.4})
    report = calibration(prepare_rows([soft]))
    assert report["calibration_details"]["choice"]["ECE"] == pytest.approx(0.2)
    assert report["calibration_details"]["choice"]["mean_TVD"] == pytest.approx(0.2)
    assert report["choice_ece_scope"] == "all_choice_fallback"
    assert report["calibration_details"]["choice"]["n_ece"] == 1
    assert report["calibration_details"]["choice"]["n_ece_hard"] == 0
    assert report["calibration_parts"]["choice"] == pytest.approx(70)
    hard = row("choice", {"a": 0.6, "b": 0.4}, "b")
    assert calibration(prepare_rows([soft, hard]))["calibration_parts"]["choice"] == pytest.approx(
        50
    )
    del soft["gold_distribution"]
    assert calibration(prepare_rows([soft]))["calibration_parts"]["choice"] == pytest.approx(60)


def test_choice_ece_hard_only_but_tvd_uses_all_available_distributions():
    rows = [
        row(
            "choice",
            {"a": 0.8, "b": 0.2},
            "a",
            "hard",
            gold_distribution={"a": 0.1, "b": 0.9},
        ),
        row(
            "choice",
            {"a": 0.6, "b": 0.4},
            "b",
            gold_distribution={"a": 0.5, "b": 0.5},
        ),
        row("choice", {"a": 1, "b": 0}, "b", "easy"),
        row("choice", {"a": 1, "b": 0}, "b", "judge"),
    ]
    report = score_reads(rows)
    detail = report["calibration_details"]["choice"]
    assert report["choice_ece_scope"] == "hard"
    assert detail["n"] == 4
    assert detail["n_ece"] == detail["n_ece_hard"] == 1
    # ECE uses the expected label even when the exact distribution's mode differs.
    assert detail["ECE"] == pytest.approx(0.2)
    assert detail["ece_score"] == pytest.approx(60)
    assert detail["mean_TVD"] == pytest.approx((0.7 + 0.1) / 2)
    assert detail["soft_target_n"] == 2
    assert report["calibration_parts"]["choice"] == pytest.approx(60)
    # Removing all distributions leaves only hard-tier ECE in the part.
    for item in rows:
        item.pop("gold_distribution", None)
    without_tvd = score_reads(rows)
    assert without_tvd["calibration_details"]["choice"]["mean_TVD"] is None
    assert without_tvd["calibration_parts"]["choice"] == pytest.approx(60)


def test_choice_ece_fallback_includes_every_recorded_tier_and_missing_tier():
    rows = [
        row("choice", {"a": 0.8, "b": 0.2}, "a", "easy"),
        row("choice", {"a": 0.6, "b": 0.4}, "b", "judge"),
        row("choice", {"a": 1, "b": 0}, "b"),
    ]
    del rows[-1]["tier"]
    report = score_reads(rows)
    detail = report["calibration_details"]["choice"]
    assert report["choice_ece_scope"] == "all_choice_fallback"
    assert detail["n_ece_hard"] == 0
    assert detail["n_ece"] == detail["n"] == 3
    assert detail["ECE"] == pytest.approx((0.2 + 0.6 + 1) / 3)


@pytest.mark.parametrize(
    "kind,probs,gold",
    [("noul", {"false": 0.2, "true": 0.8}, "true"), ("score", {"0": 0.2, "1": 0.8}, "1")],
)
def test_non_choice_calibration_ignores_gold_distribution_and_uses_all_tiers(kind, probs, gold):
    rows = [row(kind, probs, gold, "hard"), row(kind, probs, list(probs)[0], "easy")]
    before = calibration(prepare_rows(rows))
    for item in rows:
        item["gold_distribution"] = probs
    after = calibration(prepare_rows(rows))
    assert after == before
    assert after["choice_ece_scope"] is None
    detail = after["calibration_details"][kind]
    assert detail["ECE" if kind == "noul" else "top_ECE"] == pytest.approx(0.3)


def test_noul_calibration_uses_yes_probability_vs_outcome():
    items = prepare_rows(
        [
            row("noul", {"false": 0.8, "true": 0.2}, "true"),
            row("noul", {"false": 0.2, "true": 0.8}, "true"),
        ]
    )
    report = calibration(items)
    assert report["calibration_details"]["noul"]["ECE"] == pytest.approx(0.5)
    assert report["calibration_parts"]["noul"] == pytest.approx(0)


def test_score_rps_cdf_range_and_calibration_average():
    item = prepare_rows([row("score", {"0": 0.2, "1": 0.3, "2": 0.5}, "1")])[0]
    # CDFs (.2,.5) vs (0,1): RPS=.04+.25=.29; nRPS=.145.
    assert normalized_rps(item) == pytest.approx(0.145)
    assert calibration([item])["calibration_parts"]["score"] == pytest.approx((85.5 + 0) / 2)
    rows = [
        row("choice", {"a": 0.8, "b": 0.2}, "a"),
        row("noul", {"false": 0.2, "true": 0.8}, "true"),
        row("score", {"0": 0.2, "1": 0.3, "2": 0.5}, "1"),
    ]
    assert score_reads(rows)["C"] == pytest.approx((60 + 60 + 42.75) / 3)
    assert normalized_rps(prepare_rows([row("score", {"0": 1, "1": 0}, "1")])[0]) == 1


def test_speed_and_cost_hand_numbers_without_clipping():
    assert speed_score(0.1) == 100
    assert speed_score(1) == 80
    assert speed_score(0.01) == 120
    assert speed_axis(0.425, 4.925) == pytest.approx((80 + 60) / 2)
    assert cost_score(0.001) == 100
    assert cost_score(0.01) == 70
    assert cost_score(0.0001) == 130
    rows = [{"input_tokens": 100, "output_tokens": 1}, {"input_tokens": 300, "output_tokens": 3}]
    assert decision_cost(rows, 0.1, 0.5) == pytest.approx(0.021)


def test_harmonic_views_and_three_low_axis_penalties():
    assert composite(100, 100, 100, 100) == 100
    assert composite(50, 100, 100, 100) == pytest.approx(80)
    assert composite(50, 100, 100, 100, view="B") == pytest.approx(1 / 0.014)
    assert composite(25, 100, 25, 25) == pytest.approx((4 / (3 / 25 + 1 / 100)) * 0.25**3)
    assert composite(100, 25, 100, 100) == pytest.approx(4 / (3 / 100 + 1 / 25))
    assert composite(-1, 100, 100, 100) == 0


def test_cygnet_public_board_fixture():
    assert speed_axis(0.039807, 0.099066) == pytest.approx(90.9726, abs=1e-3)
    assert cost_score(0.0283376) == pytest.approx(56.4291, abs=1e-3)
    assert composite(71.0924, 87.0073, 90.9726, 56.4291) == pytest.approx(73.7013, abs=1e-3)


def test_cygnet_choice_calibration_fixture():
    # Synthetic reads reproduce the live Cygnet aggregate, with 169 open and
    # 108 sealed hard Choice items out of 799, and 50 exact gold distributions.
    board_ece = 0.04126117192486663
    board_tvd = 0.28928215362572635
    confidence = 1 - board_ece
    rows = [
        row(
            "choice",
            {"a": confidence, "b": 1 - confidence},
            "a",
            "hard",
            split="open" if index < 169 else "sealed",
            **(
                {
                    "gold_distribution": {
                        "a": confidence - board_tvd,
                        "b": 1 - confidence + board_tvd,
                    }
                }
                if index < 50
                else {}
            ),
        )
        for index in range(277)
    ]
    rows += [row("choice", {"a": 1, "b": 0}, "b") for _ in range(799 - 277)]
    report = score_reads(rows)
    detail = report["calibration_details"]["choice"]
    assert report["choice_ece_scope"] == "hard"
    assert detail["n"] == 799
    assert detail["n_ece_hard"] == detail["n_ece"] == 277
    assert detail["soft_target_n"] == 50
    assert detail["ECE"] == pytest.approx(board_ece)
    assert detail["mean_TVD"] == pytest.approx(board_tvd)
    assert detail["ece_score"] == pytest.approx(91.7478, abs=5e-5)
    assert report["calibration_parts"]["choice"] == pytest.approx(81.4098, abs=5e-5)


def test_partial_type_report_does_not_invent_missing_axes():
    report = score_reads([row("choice", {"a": 1, "b": 0}, "a")])
    assert report["I"] is None and report["C"] is None
    assert report["missing_types"] == ["noul", "score"]


@pytest.mark.parametrize(
    "call",
    [
        lambda: speed_score(0),
        lambda: speed_axis(1, 0),
        lambda: cost_score(0),
        lambda: cost_score(math.nan),
        lambda: composite(1, 1, 1, 1, view="Z"),
    ],
)
def test_invalid_axis_inputs(call):
    with pytest.raises(ValueError):
        call()
