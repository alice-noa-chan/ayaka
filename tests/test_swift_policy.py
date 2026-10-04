import json
import math

import pytest

from ayaka.swift.policy import Policy, temperature_scale


def test_temperature_square_roots_and_zeros():
    assert temperature_scale({"a": 0.9, "b": 0.1, "c": 0}, 2) == pytest.approx(
        {"a": 0.75, "b": 0.25, "c": 0}
    )
    policy = Policy(t_choice=2, t_noul=1, t_score=0.5, noul_commit=False)
    answer = policy.decide("choice", {"a": 0.9, "b": 0.1})
    assert answer["type"] == "choice" and answer["choice"] == "a"
    assert answer["probabilities"] == pytest.approx({"a": 0.75, "b": 0.25})
    assert policy.apply("score", {"0": 0.75, "1": 0.25}) == pytest.approx({"0": 0.9, "1": 0.1})


@pytest.mark.parametrize(
    "p,expected",
    [
        (0, 0),
        (0.199, 0.199),
        (0.2, 0.2),
        (0.200001, 0.199),
        (0.499999, 0.199),
        (0.5, 0.801),
        (0.799999, 0.801),
        (0.8, 0.8),
        (0.801, 0.801),
        (1, 1),
    ],
)
def test_commit_inclusive_endpoints_and_yes_tie(p, expected):
    assert Policy(commit_margin=0.0).decide("noul", {"false": 1 - p, "true": p})[
        "noul"
    ] == pytest.approx(expected)
    assert Policy(noul_commit=False).decide("noul", {"false": 1 - p, "true": p})[
        "noul"
    ] == pytest.approx(p)


def test_commit_after_temperature_and_custom_boundaries():
    policy = Policy(t_noul=2, commit_lo=0.1, commit_hi=0.9, commit_margin=0.0)
    assert policy.decide("noul", {"no": 0.1, "yes": 0.9})["noul"] == 0.9
    assert policy.decide("noul", {"no": 0.9, "yes": 0.1})["noul"] == 0.1


def test_score_response_uses_numeric_levels():
    answer = Policy().decide("score", {"2": 0.25, "4": 0.75})
    assert answer == {"type": "score", "score": 3.5, "probabilities": {"2": 0.25, "4": 0.75}}


def test_policy_roundtrip(tmp_path):
    policy = Policy(
        t_choice=3.4,
        fitted_on="private calibration",
        commit_margin=0.15,
        search={"objective": "local_composite_A", "candidates": [{"commit_margin": None}]},
        prompt_variant="rules",
    )
    path = tmp_path / "policy.json"
    policy.save(path)
    assert Policy.load(path) == policy


@pytest.mark.parametrize("enabled", [False, True])
def test_policy_loads_legacy_commit_flag(tmp_path, enabled):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"t_noul": 2, "noul_commit": enabled}), encoding="utf-8")
    policy = Policy.load(path)
    assert policy.prompt_variant == "min"
    assert policy.noul_commit is enabled
    assert policy.commit_margin == (0.0 if enabled else None)
    assert policy.decide("noul", {"no": 0.1, "yes": 0.9})["noul"] == pytest.approx(
        0.801 if enabled else 0.75
    )


@pytest.mark.parametrize(
    "p,expected",
    [
        (0.2, 0.2),
        (0.399999, 0.199),
        (0.4, 0.199),
        (0.400001, 0.400001),
        (0.5, 0.5),
        (0.599999, 0.599999),
        (0.6, 0.801),
        (0.600001, 0.801),
        (0.8, 0.8),
    ],
)
def test_margin_inclusive_boundaries_and_abstention(p, expected):
    policy = Policy(commit_margin=0.1)
    assert policy.decide("noul", {"false": 1 - p, "true": p})["noul"] == pytest.approx(expected)


def test_margin_after_temperature_and_none_disables():
    raw = {"no": 0.1, "yes": 0.9}
    assert Policy(t_noul=2, commit_margin=0.25).decide("noul", raw)["noul"] == 0.801
    assert Policy(t_noul=2, commit_margin=0.26).decide("noul", raw)["noul"] == pytest.approx(0.75)
    disabled = Policy(t_noul=2, commit_margin=None)
    assert not disabled.noul_commit
    assert disabled.decide("noul", raw)["noul"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"t_choice": 0},
        {"t_noul": math.inf},
        {"t_score": math.nan},
        {"commit_lo": 0.21},
        {"commit_hi": 0.79},
        {"noul_commit": "false"},
        {"commit_margin": -0.01},
        {"commit_margin": math.nan},
        {"commit_margin": math.inf},
    ],
)
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        Policy(**kwargs)


def test_invalid_distributions_and_primitive():
    for distribution in ({}, {"a": 0}, {"a": -1}, {"a": math.nan}):
        with pytest.raises(ValueError):
            temperature_scale(distribution, 1)
    with pytest.raises(ValueError):
        Policy().decide("unknown", {"a": 1})
