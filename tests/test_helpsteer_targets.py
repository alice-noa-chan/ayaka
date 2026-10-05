"""Keep human rating means across generic, verified-direct and wire conversion."""

import copy
import math

import pytest

from ayaka.data.direct_natural import _ordinal_target
from ayaka.data.schema import Sample
from ayaka.data.transforms import HELPSTEER_LEVELS, helpsteer2_scores, helpsteer2_target
from ayaka.swift.collect import adapt_canonical
from ayaka.swift.score import target_distribution


@pytest.mark.parametrize("rating", [0, 1, 2, 3, 4, 0.0, 2.0, 4.0])
def test_integral_ratings_keep_original_one_hot_candidates_and_mean(rating):
    row = {"prompt": "Question", "response": "Response", **dict.fromkeys(HELPSTEER_LEVELS, rating)}
    metadata = {"source": "original", "source_lineage": "original/prompt"}
    saved = copy.deepcopy((row, metadata))
    [sample] = helpsteer2_scores(row, metadata=metadata)
    assert len(sample.questions) == 5
    assert sample.state == {"prompt": row["prompt"], "response": row["response"]}
    assert sample.metadata == metadata and sample.metadata is not metadata
    for question in sample.questions:
        assert question.type == "score"
        assert [c.ordinal for c in question.candidates] == list(range(5))
        assert question.target_distribution == {f"s{i}": float(i == rating) for i in range(5)}
    assert (row, metadata) == saved


@pytest.mark.parametrize("rating", [0.1, 0.5, 1.25, 2.5, 3.75, 3.999])
def test_fractional_mean_survives_json_and_canonical_wire_instead_of_floor(rating):
    [sample] = helpsteer2_scores(
        {"prompt": "Question", "response": "Response", "correctness": rating},
        metadata={"split": "calibration", "source_lineage": "prompt-family"},
    )
    [question] = sample.questions
    target = question.target_distribution
    low, high = math.floor(rating), math.ceil(rating)
    assert {k for k, p in target.items() if p} == {f"s{low}", f"s{high}"}
    assert sum(target.values()) == pytest.approx(1)
    assert sum(c.ordinal * target[c.id] for c in question.candidates) == pytest.approx(rating)
    restored = Sample.from_json(sample.to_json())
    assert restored == sample
    [wire] = adapt_canonical(restored.to_json(), "fixture")
    assert wire.gold_distribution == {str(i): target[f"s{i}"] for i in range(5)}
    observed = {
        "labels": wire.question.labels,
        "raw_probs": dict.fromkeys(wire.question.labels, 0.2),
        "gold": wire.gold,
        "gold_distribution": wire.gold_distribution,
    }
    actual = target_distribution(observed)
    assert sum(float(level) * p for level, p in actual.items()) == pytest.approx(rating)


def test_verified_direct_and_generic_source_share_the_same_rating_contract():
    assert _ordinal_target is helpsteer2_target
    for rating in (0, 2.5, 4):
        assert _ordinal_target(rating) == helpsteer2_target(rating)


@pytest.mark.parametrize(
    "rating", [-0.1, -1, 4.1, 5, 10**400, math.nan, math.inf, -math.inf, True, False, "2.5", {}, []]
)
def test_present_invalid_rating_is_rejected_without_silent_truncation_or_cast(rating):
    with pytest.raises(ValueError, match="finite in 0..4"):
        helpsteer2_scores({"prompt": "Question", "response": "Response", "correctness": rating})
    with pytest.raises(ValueError, match="finite in 0..4"):
        _ordinal_target(rating)


def test_missing_attributes_still_skip_but_valid_attributes_are_retained():
    row = {"prompt": "Question", "response": "Response", "correctness": None}
    assert helpsteer2_scores(row) == []
    row["helpfulness"] = 3.5
    [sample] = helpsteer2_scores(row)
    assert [q.id for q in sample.questions] == ["helpfulness"]
    assert sample.questions[0].target_distribution == {
        "s0": 0,
        "s1": 0,
        "s2": 0,
        "s3": 0.5,
        "s4": 0.5,
    }
