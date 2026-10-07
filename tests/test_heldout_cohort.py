"""The held-out cohort must stay disjoint from training text and select reproducibly."""

import copy

import pytest

from ayaka.data.decontam import Decontaminator
from ayaka.data.schema import Sample
from ayaka.eval import heldout_cohort as cohort

FEATURES = {
    "info": {"features": {"intent": {"names": ["alarm_set", "weather_query", "play_music"]}}}
}


def helpsteer_row(prompt, response="An answer."):
    return {
        "prompt": prompt,
        "response": response,
        **dict.fromkeys(("helpfulness", "correctness", "coherence", "complexity", "verbosity"), 3),
    }


def rows():
    massive = [
        {"id": str(i), "utt": f"utterance {i}", "intent": i % 3, "partition": "dev"}
        for i in range(4)
    ]
    return {
        "helpsteer2": {
            "rows": [helpsteer_row("Prompt A", "first"), helpsteer_row("Prompt A", "second")]
            + [helpsteer_row(f"Prompt {i}") for i in range(3)],
            "features": {},
        },
        "commonsense_qa": {
            "rows": [
                {
                    "id": f"c{i}",
                    "question": f"Where is item {i}?",
                    "choices": {"label": ["A", "B"], "text": ["here", "there"]},
                    "answerKey": "A",
                }
                for i in range(4)
            ],
            "features": {},
        },
        "massive_ko": {"rows": copy.deepcopy(massive), "features": FEATURES},
        "massive_ja": {"rows": copy.deepcopy(massive[:3]), "features": FEATURES},
        "hotpotqa": {
            "rows": [
                {
                    "id": f"h{i}",
                    "question": f"Is claim {i} true?",
                    "answer": "yes" if i % 2 else "no",
                    "type": "comparison",
                    "context": {"title": ["T"], "sentences": [[f"Evidence {i}."]]},
                    "supporting_facts": {"title": ["T"]},
                }
                for i in range(4)
            ]
            + [
                {
                    "id": "span",
                    "question": "Who?",
                    "answer": "Someone",
                    "type": "bridge",
                    "context": {"title": ["T"], "sentences": [["Evidence."]]},
                    "supporting_facts": {"title": ["T"]},
                }
            ],
            "features": {},
        },
        "strategyqa": {
            "rows": [
                {
                    "qid": f"s{i}",
                    "question": f"Can thing {i} happen?",
                    "answer": bool(i % 2),
                    "facts": [f"Fact {i}."],
                }
                for i in range(4)
            ],
            "features": {},
        },
        "contract_nli": {
            "labels": {
                k: {"hypothesis": f"Hypothesis {k}.", "short_description": k}
                for k in cohort.contract_nli.HYPOTHESIS_IDS
            },
            "documents": [
                {
                    "id": i,
                    "text": f"Contract text {i}.",
                    "annotation_sets": [
                        {
                            "annotations": {
                                k: {"choice": "NotMentioned", "spans": []}
                                for k in cohort.contract_nli.HYPOTHESIS_IDS
                            }
                        }
                    ],
                }
                for i in range(2)
            ],
        },
    }


FILES = dict.fromkeys(
    [
        "helpsteer2",
        "commonsense_qa",
        "massive_ko",
        "massive_ja",
        "hotpotqa",
        "strategyqa",
        "contract_nli",
    ],
    "f" * 64,
)
EMPTY_TRAIN = {name: set() for name in FILES}


def settings(**natural):
    quotas = dict.fromkeys(cohort.NATURAL_SOURCES, 1)
    quotas.update(natural)
    return {
        "version": cohort.VERSION,
        "seed": 7,
        "max_input_tokens": 100,
        "natural": quotas,
        "synthetic": {"start": 2, "per_type": 1},
    }


NO_PUBLIC = Decontaminator([])


def test_units_share_lineage_across_translations_and_keep_one_response_per_prompt():
    units = cohort.natural_units(rows(), FILES)
    assert len(units["helpsteer2"]) == 4  # two responses to "Prompt A" form one case
    assert len(units["massive"]) == 3  # utterance 3 has no Japanese translation
    ko, ja = units["massive"][0][0]
    assert ko.metadata["source_lineage"] == ja.metadata["source_lineage"]
    assert (ko.metadata["language"], ja.metadata["language"]) == ("ko", "ja")
    assert len(units["hotpotqa"]) == 4  # the span-answer question is skipped
    contract = units["contract_nli"][0][0][0]
    assert len(contract.questions) == len(cohort.contract_nli.HYPOTHESIS_IDS)
    for source_units in units.values():
        for samples, _ in source_units:
            for sample in samples:
                assert sample.metadata["split"] == "dev"
                assert sample.metadata["data_kind"] == "natural"
                assert sample.metadata["original_split"] in {"validation", "test", "dev"}


def test_selection_is_deterministic_and_excludes_train_reserved_and_long_inputs():
    units = cohort.natural_units(rows(), FILES)
    first, excluded = cohort.build_cohort(settings(), units, EMPTY_TRAIN, public=NO_PUBLIC)
    again, _ = cohort.build_cohort(settings(), units, EMPTY_TRAIN, public=NO_PUBLIC)
    assert [s.to_json() for s in first] == [s.to_json() for s in again]
    assert excluded == {}
    chosen = next(s for s in first if s.metadata["source"] == "commonsense_qa")
    train = {**EMPTY_TRAIN, "commonsense_qa": {cohort.normalized(chosen.state)}}
    second, excluded = cohort.build_cohort(settings(), units, train, public=NO_PUBLIC)
    assert excluded == {"commonsense_qa": {"train_text_overlap": 1}}
    assert all(s.state != chosen.state for s in second)
    reserved = [s for s in second if s.metadata["source"] == "strategyqa"]
    _, excluded = cohort.build_cohort(
        settings(), units, EMPTY_TRAIN, reserved=reserved, public=NO_PUBLIC
    )
    assert excluded == {"strategyqa": {"reserved_overlap": 1}}


def test_long_inputs_are_skipped_and_short_supply_is_an_error():
    units = cohort.natural_units(rows(), FILES)
    first, _ = cohort.build_cohort(settings(), units, EMPTY_TRAIN, public=NO_PUBLIC)
    too_long = next(s for s in first if s.metadata["source"] == "hotpotqa")

    def fits(sample):
        return sample.state != too_long.state

    second, excluded = cohort.build_cohort(
        settings(), units, EMPTY_TRAIN, fits=fits, public=NO_PUBLIC
    )
    assert excluded == {"hotpotqa": {"context_limit": 1}}
    assert all(s.state != too_long.state for s in second)
    with pytest.raises(ValueError, match="massive: only 3 eligible units"):
        cohort.build_cohort(settings(massive=4), units, EMPTY_TRAIN, public=NO_PUBLIC)


def test_massive_overlap_in_either_language_excludes_the_whole_case():
    units = cohort.natural_units(rows(), FILES)
    first, _ = cohort.build_cohort(settings(), units, EMPTY_TRAIN, public=NO_PUBLIC)
    ja = next(s for s in first if s.metadata["source"] == "massive_ja")
    train = {**EMPTY_TRAIN, "massive_ja": {cohort.normalized(ja.state)}}
    second, excluded = cohort.build_cohort(settings(), units, train, public=NO_PUBLIC)
    assert excluded == {"massive": {"train_text_overlap": 1}}
    lineage = ja.metadata["source_lineage"]
    assert all(s.metadata["source_lineage"] != lineage for s in second)


def test_synthetic_questions_skip_the_indices_earlier_cohorts_used():
    from ayaka.data.reasoning_v2 import curriculum

    earlier = {s.metadata["source_example_id"] for s, _ in curriculum("dev", 2)}
    fresh = cohort.synthetic_samples(start=2, per_type=3)
    assert len(fresh) == 9
    assert not earlier & {s.metadata["source_example_id"] for s in fresh}
    assert all(s.metadata["data_kind"] == "synthetic" for s in fresh)
    assert cohort.synthetic_samples(start=2, per_type=0) == []


@pytest.mark.parametrize(
    "change",
    [
        {"version": "other"},
        {"seed": -1},
        {"max_input_tokens": 0},
        {"natural": {"helpsteer2": 1}},
        {"synthetic": {"start": 0}},
        {"extra": 1},
    ],
)
def test_invalid_settings_are_rejected(change):
    with pytest.raises(ValueError):
        cohort.validate_settings({**settings(), **change})


def test_summary_counts_cases_by_lineage():
    samples = [
        Sample(
            "a",
            [],
            {"source": "s", "data_kind": "natural", "language": "ko", "source_lineage": "x"},
        ),
        Sample(
            "b",
            [],
            {"source": "s", "data_kind": "natural", "language": "ja", "source_lineage": "x"},
        ),
    ]
    result = cohort.summary(samples)
    assert result["independent_cases"] == 1
    assert result["by_source"]["s"]["samples"] == 2
