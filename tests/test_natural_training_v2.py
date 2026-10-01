import copy

import pytest

from ayaka.data.natural_training_v2 import SOURCES, evidence, partition_sources
from ayaka.data.schema import Question, Sample
from ayaka.training.prepare_v2 import audit_splits, build_splits


def sources():
    result = {}
    for name in ("massive_ko", "massive_ja"):
        result[name] = [
            Sample(
                f"{name} human request number {i} with distinct evidence",
                [Question.noul("q", "Valid?", 1)],
                {
                    "source": name,
                    "revision": SOURCES[name][1],
                    "license": "CC-BY-4.0",
                    "language": SOURCES[name][3],
                    "label_source": "human",
                    "original_split": "train",
                    "data_kind": "natural",
                    "source_lineage": f"translation-parent-{i}",
                },
            )
            for i in range(200)
        ]
    return result


def test_natural_sources_reserve_existing_evidence_and_group_translations_before_splitting():
    rows = sources()
    reserved = [rows["massive_ko"][0]]
    result, report = partition_sources(
        rows, reserved, fits=lambda _: True, train_limit=8, heldout_limit=3
    )
    seen = {}
    for split, samples in result.items():
        for sample in samples:
            assert evidence(sample) != evidence(reserved[0])
            parent = sample.metadata["source_lineage"]
            assert parent != reserved[0].metadata["source_lineage"]
            assert parent not in seen or seen[parent] == split
            seen[parent] = split
    assert report["removed"]["reserved_or_public_overlap"] >= 1
    assert all(report["counts"][f"{source}/{split}"] > 0 for source in rows for split in result)
    combined = build_splits(1, 1, 1)
    for split in combined:
        combined[split] += result[split]
    audit_splits(combined)
    assert report["split_policy"].startswith("source groups")


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "main"),
        ("label_source", "commercial_teacher"),
        ("original_split", "test"),
        ("license", "unknown"),
    ],
)
def test_natural_training_rejects_unverified_sources(field, value):
    rows = sources()
    for sample in rows["massive_ko"]:
        sample.metadata[field] = value
    with pytest.raises(ValueError, match="unapproved"):
        partition_sources(rows, [], fits=lambda _: True)


def test_prompt_group_blocks_other_responses_and_overflow_drops_whole_samples():
    original = sources()
    reserved = Sample({"prompt": "shared reserved prompt", "response": "reserved answer"}, [])
    duplicate = copy.deepcopy(original["massive_ko"][0])
    duplicate.state = {"prompt": "shared reserved prompt", "response": "new answer"}
    original["massive_ko"].append(duplicate)
    result, report = partition_sources(
        original,
        [reserved],
        fits=lambda s: s.state != original["massive_ko"][2].state,
        train_limit=200,
        heldout_limit=100,
    )
    assert report["removed"]["reserved_or_public_overlap"] == 1
    assert report["removed"]["context_overflow_whole_sample"] == 1
    assert all(evidence(s) != evidence(reserved) for samples in result.values() for s in samples)
