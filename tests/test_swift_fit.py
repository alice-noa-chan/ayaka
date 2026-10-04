import json
import math
from dataclasses import asdict, replace

import pytest

from ayaka.swift.fit import fit_policy, fit_temperature, main, nll
from ayaka.swift.policy import Policy, temperature_scale
from ayaka.swift.score import composite, score_reads


def fit_exploratory(rows, **kwargs):
    return fit_policy(rows, exploratory=True, **kwargs)


def row(kind, probs, gold, **extra):
    return {
        "type": kind,
        "labels": list(probs),
        "raw_probs": probs,
        "gold": gold,
        "public": False,
        "split": "calibration",
        **extra,
    }


def synthetic_reads(in_band_correct, total_noul=200):
    # Diffuse soft-target choice/score rows have perfect competence but weaker
    # hard-label calibration. Most Noul rows are already committed and correct;
    # marginal gains on the small uncertain cohort must pay their ECE cost.
    choice = {"a": 0.34, "b": 0.33, "c": 0.33}
    score = {"0": 0.33, "1": 0.34, "2": 0.33}
    rows = [row("choice", choice, choice), row("score", score, score)]
    rows += [row("noul", {"false": 0, "true": 1}, "true") for _ in range(total_noul - 10)]
    rows += [
        row("noul", {"false": 0.5, "true": 0.5}, "true" if i < in_band_correct else "false")
        for i in range(10)
    ]
    return rows


def test_nll_hand_calculation():
    rows = [
        {"raw_probs": {"a": 0.9, "b": 0.1}, "gold": "b"},
        {"raw_probs": {"a": 0.9, "b": 0.1}, "gold": {"a": 0.25, "b": 0.75}},
    ]
    expected = (-math.log(0.25) - 0.25 * math.log(0.75) - 0.75 * math.log(0.25)) / 2
    assert nll(rows, 2) == pytest.approx(expected)


def test_recovers_known_temperature_for_each_primitive():
    rows = []
    for kind, temperature, labels in [
        ("choice", 3.4, ["a", "b", "c"]),
        ("noul", 2.1, ["false", "true"]),
        ("score", 0.7, ["0", "1", "2"]),
    ]:
        for weights in ([0.9, 0.1, 0.02], [0.2, 0.5, 0.3], [0.15, 0.8, 0.05]):
            raw = dict(zip(labels, weights[: len(labels)], strict=True))
            rows.append(
                {
                    "type": kind,
                    "raw_probs": raw,
                    "gold": temperature_scale(raw, temperature),
                    "public": False,
                    "split": "calibration",
                }
            )
    policy = fit_exploratory(rows, fitted_on="synthetic private")
    assert policy.t_choice == pytest.approx(3.4, abs=1e-5)
    assert policy.search["nll_temperatures"]["t_noul"] == pytest.approx(2.1, abs=1e-5)
    assert policy.search["nll_temperatures"]["t_score"] == pytest.approx(0.7, abs=1e-5)
    assert "public=0" in policy.fitted_on


def test_recovers_temperature_from_hard_gold_frequencies():
    # sqrt(.9)/[sqrt(.9)+sqrt(.1)] = .75; empirical frequency .75.
    rows = [{"raw_probs": {"a": 0.9, "b": 0.1}, "gold": label} for label in ["a"] * 75 + ["b"] * 25]
    assert fit_temperature(rows) == pytest.approx(2.0, abs=1e-5)
    assert fit_temperature([]) == 1
    assert fit_temperature([{"raw_probs": {"a": 0.5, "b": 0.5}, "gold": "a"}]) == 1


def test_public_refusal_override_and_provenance(tmp_path):
    rows = [
        row("choice", {"a": 0.5, "b": 0.5}, "a", public=True),
        row("noul", {"false": 0, "true": 1}, "true"),
        row("score", {"0": 0, "1": 1}, "1"),
    ]
    with pytest.raises(ValueError, match="REFUSING public=True"):
        fit_exploratory(rows)
    policy = fit_exploratory(rows, allow_public=True)
    assert policy.promotable is False
    assert "allow_public=True" in policy.fitted_on
    assert "public=1" in policy.fitted_on
    source = tmp_path / "reads.jsonl"
    output = tmp_path / "policy.json"
    source.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
    with pytest.raises(SystemExit):
        main([str(source), "--out", str(output)])
    assert not output.exists()
    main(
        [
            str(source),
            "--out",
            str(output),
            "--exploratory",
            "--allow-public",
            "--speed-axis",
            "82",
            "--cost-axis",
            "61",
        ]
    )
    loaded = Policy.load(output)
    assert loaded.fitted_on.startswith(str(source))
    assert loaded.search["speed_axis"] == 82
    assert loaded.search["cost_axis"] == 61
    assert loaded.search["choice_ece_scope"] == "all_choice_fallback"


def test_no_commit_when_calibration_cost_outweighs_competence_gain():
    rows = synthetic_reads(5)
    policy = fit_exploratory(rows)
    assert policy.commit_margin is None
    committed = replace(policy, noul_commit=True, commit_margin=0.0)
    before, after = score_reads(rows, policy), score_reads(rows, committed)
    assert after["I"] > before["I"]
    assert after["C"] < before["C"]
    assert composite(after["I"], after["C"], 91, 56.4) < policy.search["chosen"]["composite_A"]


def test_commit_when_in_band_accuracy_is_high():
    policy = fit_exploratory(synthetic_reads(8))
    assert policy.commit_margin == 0.0
    assert policy.noul_commit
    assert policy.search["chosen"] == policy.search["best"]
    no_commit = [item for item in policy.search["candidates"] if item["commit_margin"] is None]
    assert (
        policy.search["best"]["composite_A"] - max(item["composite_A"] for item in no_commit) > 0.25
    )


def test_joint_search_can_refit_noul_temperature_for_composite():
    rows = synthetic_reads(8)[:2]
    rows += [
        row("noul", {"false": 0.1, "true": 0.9}, "true" if i < 9 else "false") for i in range(10)
    ]
    rows += [
        row("noul", {"false": 0.4, "true": 0.6}, "true" if i < 80 else "false") for i in range(100)
    ]
    policy = fit_exploratory(rows)
    assert policy.t_noul == pytest.approx(2 * policy.search["nll_temperatures"]["t_noul"])
    assert policy.commit_margin == 0.0
    nll_best = max(
        item["composite_A"]
        for item in policy.search["candidates"]
        if item["temperature_multiplier"] == 1
    )
    assert policy.search["chosen"]["composite_A"] > nll_best


def test_prefer_no_commit_close_to_best_and_nll_temperature():
    policy = fit_exploratory(synthetic_reads(8, total_noul=800))
    assert policy.search["best"]["commit_margin"] == 0.0
    assert policy.commit_margin is None
    assert policy.t_noul == policy.search["nll_temperatures"]["t_noul"]
    assert 0 < policy.search["best"]["composite_A"] - policy.search["chosen"]["composite_A"] <= 0.25
    assert policy.search["runner_up"]["within_0_25"]


def score_search_reads(probs, counts):
    return [
        row("choice", {"a": 1, "b": 0}, "a"),
        row("noul", {"false": 0, "true": 1}, "true"),
        *[
            row("score", probs, label)
            for label, count in zip(probs, counts, strict=True)
            for _ in range(count)
        ],
    ]


@pytest.mark.parametrize("speed_axis,cost_axis", [(91, 56.4), (82, 61)])
def test_score_search_sharpens_when_competence_gain_pays_calibration_cost(speed_axis, cost_axis):
    rows = score_search_reads({"0": 0.1, "1": 0.9}, (2, 8))
    policy = fit_exploratory(rows, speed_axis=speed_axis, cost_axis=cost_axis)
    search = policy.search["score_search"]
    nll_t = policy.search["nll_temperatures"]["t_score"]
    assert nll_t == pytest.approx(math.log(9) / math.log(4), abs=1e-5)
    assert policy.t_score == pytest.approx(0.4 * nll_t)
    before = score_reads(rows, replace(policy, t_score=nll_t))
    after = score_reads(rows, policy)
    nmae_gain = (
        before["ordinal_value_diagnostics"]["nMAE"] - after["ordinal_value_diagnostics"]["nMAE"]
    )
    nrps_cost = (
        after["calibration_details"]["score"]["nRPS"]
        - before["calibration_details"]["score"]["nRPS"]
    )
    assert nmae_gain > nrps_cost > 0
    assert after["I"] > before["I"] and after["C"] < before["C"]
    assert search["chosen"] == search["best"]
    assert search["chosen"]["composite_A"] - search["nll"]["composite_A"] > 0.25
    assert search["chosen"]["composite_A"] == pytest.approx(
        composite(after["I"], after["C"], speed_axis, cost_axis)
    )
    for candidate in search["candidates"]:
        assert candidate["composite_A"] == pytest.approx(
            composite(candidate["I"], candidate["C"], speed_axis, cost_axis)
        )
    assert policy.search["bootstrap"]["composite_A"] == search["chosen"]["composite_A"]


def test_score_search_keeps_nll_when_calibration_cost_outweighs_competence_gain():
    rows = score_search_reads({"0": 0.1, "1": 0.2, "2": 0.7}, (1, 2, 7))
    policy = fit_exploratory(rows)
    search = policy.search["score_search"]
    assert policy.t_score == policy.search["nll_temperatures"]["t_score"] == 1
    assert search["chosen"] == search["nll"] == search["best"]
    baseline = search["nll"]
    for sharper in search["candidates"]:
        if sharper["temperature_multiplier"] < 1:
            assert sharper["score_CC"] > baseline["score_CC"]
            assert sharper["score_nRPS"] > baseline["score_nRPS"]
            assert sharper["C"] < baseline["C"]
            assert sharper["composite_A"] < baseline["composite_A"]


def test_score_search_prefers_nll_within_composite_tolerance():
    policy = fit_exploratory(score_search_reads({"0": 0.1, "1": 0.9}, (3, 7)))
    search = policy.search["score_search"]
    assert search["best"]["temperature_multiplier"] == 0.5
    assert 0 < search["best"]["composite_A"] - search["nll"]["composite_A"] <= 0.25
    assert search["chosen"] == search["nll"]
    assert policy.t_score == policy.search["nll_temperatures"]["t_score"]


def test_search_is_complete_deterministic_and_roundtrips(tmp_path):
    rows = synthetic_reads(8)
    first = fit_exploratory(rows, fitted_on="private fixture")
    second = fit_exploratory(rows, fitted_on="private fixture")
    assert asdict(first) == asdict(second)
    table = first.search["candidates"]
    assert len(table) == 70
    assert {item["commit_margin"] for item in table} == {None, *(i / 40 for i in range(13))}
    assert {item["temperature_multiplier"] for item in table} == {0.5, 0.7, 1, 1.4, 2}
    for candidate in table:
        assert candidate["t_noul"] == pytest.approx(
            first.search["nll_temperatures"]["t_noul"] * candidate["temperature_multiplier"]
        )
        assert candidate["t_score"] == first.search["nll_temperatures"]["t_score"]
        assert candidate["composite_A"] == pytest.approx(
            composite(candidate["I"], candidate["C"], 91.0, 56.4)
        )
    assert first.search["strategy"] == "sequential_noul_then_score"
    score_table = first.search["score_search"]["candidates"]
    assert len(score_table) == 6
    assert {item["temperature_multiplier"] for item in score_table} == {0.4, 0.5, 0.7, 1, 1.4, 2}
    for candidate in score_table:
        assert (
            candidate["t_choice"] == first.t_choice == first.search["nll_temperatures"]["t_choice"]
        )
        assert candidate["t_noul"] == first.t_noul == first.search["chosen"]["t_noul"]
        assert (
            candidate["commit_margin"]
            == first.commit_margin
            == first.search["chosen"]["commit_margin"]
        )
        assert candidate["t_score"] == pytest.approx(
            first.search["nll_temperatures"]["t_score"] * candidate["temperature_multiplier"]
        )
        assert candidate["composite_A"] == pytest.approx(
            composite(candidate["I"], candidate["C"], 91, 56.4)
        )
    assert first.t_score == first.search["score_search"]["chosen"]["t_score"]
    interval = first.search["bootstrap"]
    assert interval["B"] == 1000 and interval["seed"] == 15
    assert interval["unit"] == "case"
    assert interval["ci_95"][0] < interval["ci_95"][1]
    assert interval["composite_A"] == first.search["score_search"]["chosen"]["composite_A"]
    path = tmp_path / "policy.json"
    first.save(path)
    assert Policy.load(path) == first


@pytest.mark.parametrize("hard_choice", [True, False])
def test_fit_composite_uses_choice_scope_and_reports_it(hard_choice):
    rows = [
        row(
            "choice",
            {"a": 0.8, "b": 0.2},
            "a",
            **({"tier": "hard"} if hard_choice else {}),
            gold_distribution={"a": 0.7, "b": 0.3},
        ),
        row("choice", {"a": 1, "b": 0}, "b", tier="standard"),
        row("noul", {"false": 0, "true": 1}, "true"),
        row("score", {"0": 0, "1": 1}, "1"),
    ]
    policy = fit_exploratory(rows)
    scope = "hard" if hard_choice else "all_choice_fallback"
    assert policy.search["choice_ece_scope"] == scope
    p = policy.apply("choice", rows[0]["raw_probs"])["a"]
    choice_ece = 1 - p if hard_choice else ((1 - p) + 1) / 2
    choice_part = (100 * (1 - choice_ece / 0.5) + 100 * (1 - abs(p - 0.7))) / 2
    expected_c = (choice_part + 100 + 100) / 3
    for candidate in policy.search["candidates"] + policy.search["score_search"]["candidates"]:
        assert candidate["choice_ece_scope"] == scope
        assert candidate["C"] == pytest.approx(expected_c)
        assert candidate["composite_A"] == pytest.approx(
            composite(candidate["I"], expected_c, 91, 56.4)
        )
    chosen = policy.search["score_search"]["chosen"]
    assert policy.search["bootstrap"]["choice_ece_scope"] == scope
    assert policy.search["bootstrap"]["composite_A"] == chosen["composite_A"]


@pytest.mark.parametrize("kwargs", [{"speed_axis": math.nan}, {"cost_axis": 0}])
def test_fit_rejects_invalid_fixed_axes(kwargs):
    with pytest.raises(ValueError, match="speed-axis and cost-axis"):
        fit_exploratory(synthetic_reads(5), **kwargs)


def test_fit_requires_all_primitives_and_nonempty_reads():
    with pytest.raises(ValueError, match="empty"):
        fit_exploratory([])
    with pytest.raises(ValueError, match="choice, noul and score"):
        fit_exploratory([row("choice", {"a": 0.5, "b": 0.5}, "a")])


def test_fit_records_variant_and_refuses_mixed_variants():
    rows = [dict(item, prompt_variant="rules") for item in synthetic_reads(8)]
    assert fit_exploratory(rows).prompt_variant == "rules"
    rows[0]["prompt_variant"] = "min"
    with pytest.raises(ValueError, match="mixed prompt_variant"):
        fit_exploratory(rows)
