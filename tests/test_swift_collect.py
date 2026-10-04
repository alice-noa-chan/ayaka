import json
import threading
from dataclasses import replace

import pytest

from ayaka.swift.collect import adapt_jevbench, collect, load_reads
from ayaka.swift.readers import FakeReader


def items(count):
    return [
        adapt_jevbench(
            {
                "id": str(index),
                "state": str(index),
                "labels": ["no", "yes"],
                "expected": "yes",
                "question": {"type": "noul"},
            }
        )
        for index in range(count)
    ]


def test_collect_bounded_concurrency_order_resume_limit_and_identity(tmp_path):
    barrier = threading.Barrier(3, timeout=5)
    lock = threading.Lock()
    active = peak = entered = 0

    class ConcurrentFakeReader(FakeReader):
        def read(self, messages, letters):
            nonlocal active, peak, entered
            with lock:
                index = entered
                entered += 1
                active += 1
                peak = max(peak, active)
            try:
                if index < 3:
                    barrier.wait()
                return super().read(messages, letters)
            finally:
                with lock:
                    active -= 1

    reader = ConcurrentFakeReader()
    dataset = items(8)
    output = tmp_path / "nested/reads.jsonl"
    options = {"concurrency": 3, "model": "fixture", "revision": "a" * 40}
    assert collect(iter(dataset), reader, output, limit=5, **options) == 5
    assert peak == 3
    assert len(reader.calls) == 5
    rows = load_reads([output])
    assert [row["id"] for row in rows] == [str(index) for index in range(5)]
    assert all(row["model"] == "fixture" and row["revision"] == "a" * 40 for row in rows)
    assert collect(iter(dataset), reader, output, limit=1, **options) == 1
    assert collect(iter(dataset), reader, output, **options) == 2
    assert collect(iter(dataset), reader, output, **options) == 0
    assert len(reader.calls) == 8
    with pytest.raises(ValueError, match="cached read differs"):
        collect(iter(dataset), reader, output, **dict(options, revision="b" * 40))


@pytest.mark.parametrize("concurrency", [0, -1])
def test_collect_rejects_invalid_concurrency(tmp_path, concurrency):
    with pytest.raises(ValueError, match="concurrency"):
        collect(iter(items(1)), FakeReader(), tmp_path / "reads.jsonl", concurrency=concurrency)


def test_collect_zero_limit_does_not_consume_input(tmp_path):
    def unopened():
        raise AssertionError("input must not be consumed")
        yield

    assert collect(unopened(), FakeReader(), tmp_path / "reads.jsonl", limit=0, concurrency=4) == 0


@pytest.mark.parametrize("variant", ["min", "cygnet", "rules"])
def test_collect_variant_recording_and_resume_refusal(tmp_path, variant):
    reader = FakeReader()
    output = tmp_path / "reads.jsonl"
    assert collect(iter(items(1)), reader, output, prompt_variant=variant) == 1
    assert load_reads([output])[0]["prompt_variant"] == variant
    from ayaka.swift.prompt import render_question

    expected, _ = render_question("0", items(1)[0].question, prompt_variant=variant)
    assert reader.calls[0][0] == expected
    assert collect(iter(items(1)), reader, output, prompt_variant=variant) == 0
    with pytest.raises(ValueError, match="cached read differs"):
        collect(
            iter(items(2)), reader, output, prompt_variant="rules" if variant != "rules" else "min"
        )


def test_collect_preserves_completed_rows_on_backend_failure(tmp_path):
    dataset = items(4)
    dataset[1] = replace(dataset[1], state="broken")

    def fixture(messages, letters):
        if "broken" in messages[-1]["content"]:
            raise RuntimeError("broken backend")
        return dict.fromkeys(letters, 1)

    output = tmp_path / "reads.jsonl"
    with pytest.raises(RuntimeError, match="broken backend"):
        collect(iter(dataset), FakeReader(fixture), output, concurrency=2)
    assert [json.loads(line)["id"] for line in output.read_text().splitlines()] == ["0"]


@pytest.mark.parametrize(
    "kind,labels,expected,gold_probs,gold,distribution",
    [
        ("choice", ["a", "b"], "a", {"a": 0.4, "b": 0.6}, "a", {"a": 0.4, "b": 0.6}),
        (
            "noul",
            ["no", "yes"],
            "yes",
            {"no": 0.9, "yes": 0.1},
            "true",
            {"false": 0.9, "true": 0.1},
        ),
        ("score", ["0", "1"], "1", {"0": 0.8, "1": 0.2}, "1", {"0": 0.8, "1": 0.2}),
    ],
)
def test_collect_jevbench_provenance_distribution_and_family(
    tmp_path, kind, labels, expected, gold_probs, gold, distribution
):
    item = adapt_jevbench(
        {
            "id": "hard-probability-1",
            "family": "probability",
            "labels": labels,
            "expected": expected,
            "question": {"type": kind, "criteria": dict.fromkeys(labels, "option")},
            "provenance": {"gold_probs": gold_probs},
        }
    )
    output = tmp_path / "reads.jsonl"
    assert collect(iter([item]), FakeReader(), output) == 1
    saved = load_reads([output])[0]
    assert saved["gold"] == gold
    assert saved["gold_distribution"] == distribution
    assert saved["family"] == "probability"
    assert saved["tier"] == "hard"
