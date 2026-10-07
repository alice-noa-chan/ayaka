"""Matched results must retain checkpoint, input, order and reasoning identities."""

import copy
import json
from decimal import Decimal

import pytest

from ayaka.data.schema import Candidate, Question, Sample
from ayaka.eval.checkpoint_comparison import (
    SYSTEMS,
    ContentTokenCounter,
    checked_rows,
    collect_rows,
    compare,
    gate_breakdown,
    make_protocol,
    questions,
)
from ayaka.eval.matched_execution import full_decision_context
from ayaka.tokenization import ToyTokenizer


@pytest.fixture
def cohort():
    samples = []
    for index, kind in enumerate(("choice", "noul", "score")):
        question = (
            Question.noul("q", "Is it approved?", 1)
            if kind == "noul"
            else Question(
                "q",
                kind,
                "Use the evidence",
                [
                    Candidate("original-b", "First option", 0 if kind == "score" else None),
                    Candidate("original-a", "Second option", 1 if kind == "score" else None),
                ],
                {"original-b": 1.0},
            )
        )
        samples.append(
            Sample(
                {"fact": "Approved"},
                [question],
                {
                    "split": "dev",
                    "source_example_id": str(index),
                    "source_lineage": f"case/{index}",
                    "language": "en",
                    "task_family": "approval",
                },
            )
        )
    return samples, make_protocol(samples, "a" * 64, "b" * 64)


def observation(system, probability=0.9, calls=None):
    def predict(state, spec):
        if calls is not None:
            calls.append(list(spec.candidate_ids))
        probs = [probability, 1 - probability]
        if spec.type == "noul":
            probs.reverse()
        return {
            "probs": probs,
            "route": "baseline"
            if system == "v1_on"
            else "direct"
            if system == "v2_off"
            else "reasoned",
            "error": None,
            "reasoning_tokens": 5 if system == "v2_on" else 0,
            "finish_reason": "length",
            "checked_contexts": [full_decision_context(state, [spec], ToyTokenizer())],
        }

    return predict


def results(tmp_path, cohort):
    samples, protocol = cohort
    return {
        name: collect_rows(observation(name), samples, protocol, name, tmp_path / f"{name}.jsonl")
        for name in SYSTEMS
    }


def test_collection_preserves_candidate_ids_and_resumes_without_new_reads(tmp_path, cohort):
    samples, protocol = cohort
    calls = []
    path = tmp_path / "off.jsonl"
    first = collect_rows(observation("v2_off", calls=calls), samples, protocol, "v2_off", path)
    assert calls == [["original-b", "original-a"], ["false", "true"], ["original-b", "original-a"]]
    before = path.read_bytes()
    assert (
        collect_rows(observation("v2_off", calls=calls), samples, protocol, "v2_off", path) == first
    )
    assert len(calls) == 3 and path.read_bytes() == before


def test_schema_loaded_decimal_levels_remain_exact_json_numbers(tmp_path, cohort):
    samples, _ = cohort
    score = samples[-1].questions[0]
    score.candidates[0].ordinal = Decimal("9")
    score.candidates[1].ordinal = Decimal("9.25")
    samples = [Sample.from_json(sample.to_json()) for sample in samples]
    protocol = make_protocol(samples, "a" * 64, "b" * 64)
    rows = collect_rows(observation("v2_off"), samples, protocol, "v2_off", tmp_path / "off.jsonl")
    assert rows[-1]["ordinals"] == [9, 9.25]
    assert json.loads((tmp_path / "off.jsonl").read_text().splitlines()[-1])["ordinals"] == [
        9,
        9.25,
    ]


def test_decimal_levels_that_would_lose_precision_are_rejected(cohort):
    samples, _ = cohort
    samples[-1].questions[0].candidates[0].ordinal = Decimal("0.1234567890123456789")
    with pytest.raises(ValueError, match="represented exactly"):
        questions(samples)


@pytest.mark.parametrize(
    ("key", "value"),
    [("target", [0.5, 0.5]), ("input_sha256", "d" * 64), ("model_id", "c" * 64), ("budget", 384)],
)
def test_changed_resume_inputs_or_settings_are_rejected(tmp_path, cohort, key, value):
    samples, protocol = cohort
    path = tmp_path / "off.jsonl"
    rows = collect_rows(observation("v2_off"), samples, protocol, "v2_off", path)
    rows[0][key] = value
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        collect_rows(observation("v2_off"), samples, protocol, "v2_off", path)
    assert path.read_bytes() == before


def test_cached_scores_cannot_determine_comparison_or_override_case_count(tmp_path, cohort):
    samples, protocol = cohort
    rows = results(tmp_path, cohort)
    expected = compare(rows, samples, protocol, replicates=5)
    for group in rows.values():
        for row in group:
            row.update(nll=1e99, correct=1e99, nmae=1e99)
    actual = compare(rows, samples, protocol, replicates=5)
    assert actual == expected
    assert actual["v2_reasoning_completed"] == 3
    assert actual["comparisons"]["v2_off_over_v1_on"]["paired"]["independent_cases"] == 3
    assert actual["screen_passed"] is False and actual["official_composite"] is None


def test_incomplete_or_duplicate_system_cannot_be_scored(tmp_path, cohort):
    samples, protocol = cohort
    rows = results(tmp_path, cohort)
    rows["v1_on"].pop()
    with pytest.raises(ValueError, match="every development question"):
        compare(rows, samples, protocol, replicates=5)
    rows["v2_off"].append(rows["v2_off"][0])
    with pytest.raises(ValueError, match="duplicate"):
        checked_rows(rows["v2_off"], samples, protocol, "v2_off")


def test_clipped_native_context_and_empty_reasoned_trace_are_rejected(tmp_path, cohort):
    samples, protocol = cohort
    rows = results(tmp_path, cohort)
    rows["v1_on"][0]["checked_contexts"][0]["input_tokens"] = [8193]
    with pytest.raises(ValueError, match="clipped"):
        checked_rows(rows["v1_on"], samples, protocol, "v1_on")
    rows["v2_on"][0]["reasoning_tokens"] = 0
    with pytest.raises(ValueError, match="reasoning usage"):
        checked_rows(rows["v2_on"], samples, protocol, "v2_on")


def test_gate_breakdown_separates_v1_reasoned_questions_and_composes_v2(tmp_path, cohort):
    samples, protocol = cohort
    rows = results(tmp_path, cohort)
    # v1's gate reasons on the Noul question; v2 direct abstains on it.
    for row in rows["v1_on"]:
        if row["type"] == "noul":
            row.update(route="reasoned", reasoning_tokens=5)
    for row in rows["v2_off"]:
        if row["type"] == "noul":
            row["probs"] = [0.5, 0.5]
    report = gate_breakdown(rows, samples, protocol, replicates=5)
    assert report["gated_questions"] == 1
    assert report["strata"]["v1_reasoned"]["questions"] == 1
    assert report["strata"]["v1_direct"]["questions"] == 2
    assert set(report["strata"]["v1_reasoned"]["systems"]["v2_off"]["by_type"]) == {"noul"}
    assert report["noul_abstentions_by_source"]["v2_off"] == {"approval": 1}
    assert report["noul_abstentions_by_source"]["v2_composed"] == {}
    composed = report["v2_composed"]["by_type"]
    direct = compare(rows, samples, protocol, replicates=5)["systems"]["v2_off"]["by_type"]
    assert composed["noul"]["abstentions"] == 0
    assert composed["choice"] == direct["choice"] and composed["score"] == direct["score"]
    assert report["promotable"] is False and report["official_composite"] is None


def test_gate_breakdown_without_gated_questions_reports_an_empty_stratum(tmp_path, cohort):
    samples, protocol = cohort
    report = gate_breakdown(results(tmp_path, cohort), samples, protocol, replicates=5)
    assert report["gated_questions"] == 0
    assert report["strata"]["v1_reasoned"] == {"questions": 0, "systems": None}


def test_protocol_rejects_test_inputs_and_identical_checkpoint_identities(cohort):
    samples, _ = cohort
    with pytest.raises(ValueError, match="distinct"):
        make_protocol(samples, "a" * 64, "a" * 64)
    changed = copy.deepcopy(samples)
    changed[0].metadata["split"] = "test"
    with pytest.raises(ValueError, match="dev samples"):
        make_protocol(changed, "a" * 64, "b" * 64)


def test_native_counter_observes_content_without_changing_decoded_text():
    from types import SimpleNamespace

    tok = SimpleNamespace(decode=lambda ids: bytes(ids).decode(), pad_id=0)
    observed = ContentTokenCounter(tok)
    ids = list(b"Worked steps")
    assert observed.decode(ids) == tok.decode(ids)
    assert observed.tokens == len(ids)
    assert observed.pad_id == tok.pad_id
