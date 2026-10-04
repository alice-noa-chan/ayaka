import json
from pathlib import Path

import pytest

from ayaka.swift.collect import adapt_canonical, adapt_jevbench, collect, infer_tier, iter_dataset
from ayaka.swift.readers import FakeReader


def public_record():
    return {
        "id": "easy-1",
        "state": "s",
        "split": "public",
        "labels": ["b", "a"],
        "expected": "a",
        "question": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
    }


def test_jevbench_choice_label_order_and_public():
    item = adapt_jevbench(public_record(), "easy.jsonl")
    assert item.question.labels == ["b", "a"]
    assert item.question.descriptions == ["B", "A"]
    assert item.gold == "a"
    assert item.tier == "easy"
    assert item.public


def test_private_and_noul_label_conversion():
    record = {
        "id": "original-1",
        "state": "s",
        "split": "private",
        "labels": ["yes", "no"],
        "expected": {"yes": 0.1, "no": 0.9},
        "question": {"type": "noul", "criteria": {"true": "T", "false": "F"}},
    }
    item = adapt_jevbench(record, "private.jsonl")
    assert not item.public
    assert item.tier == "standard"
    assert item.question.labels == ["false", "true"]
    assert item.question.descriptions == ["F", "T"]
    assert item.gold == {"true": 0.1, "false": 0.9}
    record["expected"] = "yes"
    assert adapt_jevbench(record).gold == "true"


@pytest.mark.parametrize(
    "record,source,tier",
    [
        ({"id": "hard-x"}, "unknown.jsonl", "hard"),
        ({"family": "judge_policy", "id": "hard-x"}, "easy.jsonl", "judge"),
        ({"split": "judge"}, "original.jsonl", "judge"),
        ({}, "original.jsonl", "standard"),
        ({}, "easy.jsonl", "easy"),
        ({}, "hard.jsonl", "hard"),
    ],
)
def test_tier_detection(record, source, tier):
    assert infer_tier(record, source) == tier


def test_canonical_multiple_questions_and_score_ordinals():
    record = {
        "state": {"x": 1},
        "metadata": {
            "source_example_id": "calibration/1",
            "source": "verified",
            "split": "calibration",
        },
        "questions": [
            {
                "id": "c",
                "type": "choice",
                "instruction": "Pick",
                "candidates": [{"id": "b", "description": "B"}, {"id": "a", "description": "A"}],
                "target_distribution": {"a": 0.7, "b": 0.3},
            },
            {
                "id": "n",
                "type": "noul",
                "candidates": [
                    {"id": "true", "description": "T"},
                    {"id": "false", "description": "F"},
                ],
                "target_distribution": {"true": 1, "false": 0},
            },
            {
                "id": "s",
                "type": "score",
                "candidates": [
                    {"id": "high", "description": "H", "ordinal": 10},
                    {"id": "low", "description": "L", "ordinal": 2},
                ],
                "target_distribution": {"high": 0.8, "low": 0.2},
            },
        ],
    }
    choice, noul, score = adapt_canonical(record)
    assert choice.id == "calibration/1/c"
    assert choice.source == "verified"
    assert choice.gold == "a"
    assert choice.gold_distribution == {"a": 0.7, "b": 0.3}
    assert choice.question.labels == ["b", "a"]
    assert not choice.public
    assert noul.question.labels == ["false", "true"]
    assert noul.question.descriptions == ["F", "T"]
    assert score.question.labels == ["2", "10"]
    assert score.question.descriptions == ["L", "H"]
    assert score.gold == "10"
    assert score.gold_distribution == {"10": 0.8, "2": 0.2}


@pytest.mark.parametrize(
    "record_tier,metadata_tier,question_tier,expected",
    [
        (None, None, None, "standard"),
        ("hard", None, None, "hard"),
        (None, "hard", None, "hard"),
        ("easy", "hard", None, "easy"),
        ("hard", None, "judge", "judge"),
        ("original", None, None, "standard"),
    ],
)
def test_canonical_uses_only_explicit_tier(record_tier, metadata_tier, question_tier, expected):
    record = {
        "id": "hard-example",
        "family": "judge_probability",
        "split": "hard",
        "metadata": {"task_family": "hard_task", "split": "hard"},
        "questions": [
            {
                "type": "choice",
                "candidates": [{"id": "a"}, {"id": "b"}],
                "target_distribution": {"a": 0.6, "b": 0.4},
            }
        ],
    }
    for target, tier in (
        (record, record_tier),
        (record["metadata"], metadata_tier),
        (record["questions"][0], question_tier),
    ):
        if tier is not None:
            target["tier"] = tier
    item = adapt_canonical(record, "hard-calibration.jsonl")[0]
    assert item.tier == expected
    assert item.family == "judge_probability"


def test_public_path_prevents_unmarked_public_fit():
    record = public_record()
    del record["split"]
    assert adapt_jevbench(record, "ayaka/eval/data/jevbench_public/easy.jsonl").public
    assert adapt_jevbench(record, r"runs\audit\datasets\public\easy.jsonl").public


def test_collect_resume_limit_and_rows(tmp_path):
    dataset = tmp_path / "easy.jsonl"
    records = [dict(public_record(), id=f"easy-{i}") for i in range(3)]
    dataset.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    output = tmp_path / "reads.jsonl"
    reader = FakeReader()
    assert collect(iter_dataset([dataset]), reader, output, limit=1) == 1
    assert collect(iter_dataset([dataset]), reader, output) == 2
    assert collect(iter_dataset([dataset]), reader, output) == 0
    assert len(reader.calls) == 3
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    expected = {
        "id": "easy-0",
        "model": "fake",
        "revision": None,
        "prompt_variant": "min",
        "source": str(dataset),
        "tier": "easy",
        "type": "choice",
        "labels": ["b", "a"],
        "gold": "a",
        "public": True,
        "raw_probs": {"b": 0.5, "a": 0.5},
        "input_tokens": 10,
        "output_tokens": 1,
        "latency_s": 0.01,
    }
    assert {key: rows[0][key] for key in expected} == expected
    assert rows[0]["split"] == "public"
    assert rows[0]["cluster_id"] == "easy-0"
    assert rows[0]["readout"] == "canonical_letter"
    assert rows[0]["passes"] == 1
    assert rows[0]["binding"]["messages"] == reader.calls[0][0]


def test_all_vendored_public_items_adapt():
    root = Path(__file__).resolve().parents[1] / "ayaka/eval/data/jevbench_public"
    items = list(iter_dataset([root]))
    assert len(items) == 231
    assert all(item.public for item in items)
    assert {item.tier for item in items} == {"easy", "standard", "judge", "hard"}
    soft = [item for item in items if item.gold_distribution is not None]
    assert len(soft) == 10
    assert sum(item.question.type == "choice" for item in soft) == 7
    assert sum(item.question.type == "noul" for item in soft) == 3
    assert all(item.tier == "hard" and item.family == "probability" for item in soft)
