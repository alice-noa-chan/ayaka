"""Bound, CPU-only synthetic calibration and position transport checks."""

import copy
import math
from dataclasses import replace

import pytest

from ayaka.swift.bias import BIAS_L2, BIAS_MIN_QUESTIONS, fit_letter_bias
from ayaka.swift.collect import adapt_jevbench, collect, load_reads
from ayaka.swift.policy import Policy
from ayaka.swift.readers import FakeReader, ReadResult, logmass_probs


def bound_synthetic(tmp_path, split, variant, specs, *, tier="hard"):
    """specs = [(kind, raw log masses in position order, gold), ...]."""
    items, results = [], []
    for index, (kind, masses, gold) in enumerate(specs):
        labels = {
            "choice": [f"label-{j}" for j in range(len(masses))],
            "noul": ["false", "true"],
            "score": [str(j) for j in range(len(masses))],
        }[kind]
        item = adapt_jevbench(
            {
                "id": f"{split}/{index}/{kind}",
                "source": "synthetic-private",
                "case_id": f"{split}/{index // 3}",
                "state": f"{split}/{index}",
                "public": False,
                "split": split,
                "tier": tier,
                "labels": labels,
                "expected": labels[gold] if isinstance(gold, int) else labels[0],
                "question": {"type": kind, "criteria": dict.fromkeys(labels, "option")},
            }
        )
        if isinstance(gold, list):
            item = replace(item, gold_distribution=dict(zip(labels, gold, strict=True)))
        items.append(item)
        letter_masses = {chr(65 + j): mass for j, mass in enumerate(masses)}
        results.append(ReadResult(logmass_probs(letter_masses), 100, 1, 0.01, letter_masses))
    reader = FakeReader(results)
    reader.backend = "hf"
    reader.logprobs_mode = "raw_logits"
    path = tmp_path / f"{split}-{variant}.jsonl"
    collect(iter(items), reader, path, model="synthetic", revision="a" * 40, prompt_variant=variant)
    return load_reads([path])


@pytest.mark.parametrize(
    "kind,planted", [("choice", [0.7, -0.2, -0.5]), ("score", [0.7, -0.2, -0.5]), ("noul", [0.8])]
)
def test_recovers_planted_position_bias_from_raw_masses_and_soft_targets(tmp_path, kind, planted):
    offsets = [0, planted[0]] if kind == "noul" else planted
    specs = []
    for index in range(60):
        logits = [math.sin(index + j) for j in range(len(offsets))]
        target = list(logmass_probs(dict(enumerate(logits))).values())
        specs.append((kind, [z + b for z, b in zip(logits, offsets, strict=True)], target))
    rows = bound_synthetic(tmp_path, "calibration", "min", specs)
    fitted = fit_letter_bias(rows, Policy())
    assert fitted["l2"] == BIAS_L2
    assert fitted["minimum_questions"] == BIAS_MIN_QUESTIONS
    assert fitted["letter_bias"][kind][str(len(offsets))] == pytest.approx(
        [-b for b in planted], abs=0.015
    )
    report = fitted["buckets"][f"{kind}/{len(offsets)}"]
    assert report["objective"] < report["initial_objective"]


def test_bucket_threshold_and_frozen_temperature(tmp_path):
    specs = [("choice", [0, 1], [0.5, 0.5])] * 30
    specs += [("choice", [0, 0, 1], [1 / 3] * 3)] * 29
    specs += [("score", [0, 1], [0.5, 0.5])] * 29
    result = fit_letter_bias(
        bound_synthetic(tmp_path, "calibration", "min", specs), Policy(t_choice=2)
    )
    assert set(result["letter_bias"]) == {"choice"}
    assert set(result["letter_bias"]["choice"]) == {"2"}
    assert result["letter_bias"]["choice"]["2"] == pytest.approx([0.5, -0.5], abs=0.01)
    assert result["buckets"]["score/2"]["reason"] == "too_few_questions"


def test_policy_absent_bias_is_noop_and_saved_without_bias(tmp_path):
    probs = {"a": 0.3, "b": 0.7}
    masses = {"b": 1, "a": 0}
    for logs in (None, masses):
        assert Policy().apply("choice", probs, candidate_log_masses=logs) == Policy(
            letter_bias={}
        ).apply("choice", probs, candidate_log_masses=logs)
    path = tmp_path / "policy.json"
    Policy().save(path)
    assert '"letter_bias"' not in path.read_text(encoding="utf-8")
    assert Policy.load(path).letter_bias is None


def test_bias_is_by_position_before_temperature_when_labels_move(tmp_path):
    policy = Policy(t_choice=2, letter_bias={"choice": {"2": [2, -2]}})
    first = policy.apply(
        "choice", {"cat": 0.5, "dog": 0.5}, candidate_log_masses={"dog": 0, "cat": 0}
    )
    moved = policy.apply(
        "choice", {"dog": 0.5, "cat": 0.5}, candidate_log_masses={"cat": 0, "dog": 0}
    )
    assert first["cat"] == moved["dog"] == pytest.approx(1 / (1 + math.exp(-2)))
    path = tmp_path / "bias.json"
    policy.save(path)
    assert Policy.load(path) == policy
    with pytest.raises(ValueError, match="raw canonical"):
        policy.apply("choice", {"cat": 0.5, "dog": 0.5})
    assert policy.apply("score", {"0": 0.5, "1": 0.5}) == {"0": 0.5, "1": 0.5}


def test_noul_intercept_is_on_true_logit_and_recovers_underflow():
    policy = Policy(letter_bias={"noul": {"2": [1000]}})
    assert policy.apply(
        "noul", {"true": 0, "false": 1}, candidate_log_masses={"false": 0, "true": -1000}
    ) == {"true": 0.5, "false": 0.5}


@pytest.mark.parametrize("split,public", [("dev", False), ("test", False), ("calibration", True)])
def test_bias_refuses_holdouts_and_public_before_fitting(tmp_path, split, public):
    rows = bound_synthetic(tmp_path, "calibration", "min", [("choice", [0, 1], 0)] * 30)
    bad = copy.deepcopy(rows)
    bad[0].update(split=split, public=public)
    with pytest.raises(ValueError):
        fit_letter_bias(bad, Policy())


@pytest.mark.parametrize(
    "bias",
    [
        {"choice": {"2": [0]}},
        {"score": {"1": [0]}},
        {"noul": {"2": [0, 0]}},
        {"noul": {"3": [0]}},
        {"choice": {"2": [0, math.nan]}},
        {"unknown": {}},
    ],
)
def test_invalid_bias_rejected(bias):
    with pytest.raises(ValueError, match="letter_bias"):
        Policy(letter_bias=bias)
