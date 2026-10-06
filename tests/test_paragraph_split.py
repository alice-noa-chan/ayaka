import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ayaka.config import tiny_config
from ayaka.data import paragraph_split as builder
from ayaka.data.schema import Sample
from ayaka.eval.matched_contract import digest
from ayaka.tokenization import HFTokenizer, ToyTokenizer


def raw(identifier, titles, *, answer="yes", text=None):
    return {
        "id": identifier,
        "question": f"Does evidence establish {identifier}?",
        "answer": answer,
        "type": "bridge",
        "context": {"title": titles, "sentences": [[text or f"{t} evidence."] for t in titles]},
    }


def plan(*, roles=None, quota=2, components=2, blocked=None, limit=1000):
    roles = {"calibration": 1} if roles is None else roles
    return {
        "version": builder.VERSION,
        "seed": 31,
        "namespace": "hotpotqa/article",
        "source": "hotpot_val",
        "hf_split": "validation",
        "license": "CC BY-SA 4.0 (caller declaration)",
        "roles": roles,
        "minimum_counts": {f"{role}/hotpot_val/noul": quota for role in roles},
        "minimum_components": {f"{role}/hotpot_val/noul": components for role in roles},
        "input_encoding": {"encoder": "ayaka_segmented"},
        "context_limit": limit,
        "blocked_raw_ids": blocked or [],
    }


def build(rows, declared=None, *, public_hit=None):
    rows = list(rows)
    return builder.build_hotpot_splits(
        rows,
        declared or plan(),
        ToyTokenizer(),
        tiny_config(),
        expected_raw_sha256=builder.raw_digest(rows),
        decontaminator=SimpleNamespace(sample_hit=public_hit or (lambda sample: False)),
    )


def test_full_raw_builder_connects_converter_encoder_component_audit_and_canonical_output():
    rows = [raw("a", ["A"]), raw("b", ["B"]), raw("unused", ["C"], answer="span")]
    outputs, receipt = build(rows)
    assert len(outputs["calibration"]) == 2
    assert receipt["paragraph_audit"]["inventory_rows"] == 3
    assert receipt["paragraph_audit"]["component_audit"]["components_per_role"] == {
        "calibration": 2
    }
    assert receipt["promotable"] is receipt["historical_inventory_complete"] is False
    assert receipt["dropped"] == {"unsupported_bridge": 1}
    for row in outputs["calibration"]:
        sample = Sample.from_json(row)
        assert sample.metadata["split"] == "calibration"
        assert sample.metadata["source_lineage"].startswith("paragraph-component/")
        assert sample.questions[0].target_distribution == {"false": 0.0, "true": 1.0}


def test_unsupported_unselected_bridge_still_prevents_false_component_coverage():
    rows = [
        raw("a", ["A", "B"]),
        raw("bridge", ["B", "C"], answer="unsupported"),
        raw("c", ["C", "D"]),
    ]
    with pytest.raises(ValueError, match="component minimum"):
        build(rows)


def test_eight_questions_in_one_component_cannot_fill_two_component_minimum():
    rows = [raw(str(i), ["Shared", f"Extra{i}"]) for i in range(8)]
    with pytest.raises(ValueError, match="component minimum"):
        build(rows, plan(quota=8, components=2))


def test_historical_blocked_bridge_excludes_entire_component_before_quota():
    rows = [
        raw("a", ["A", "B"]),
        raw("bridge", ["B", "C"], answer="span"),
        raw("c", ["C", "D"]),
        raw("e", ["E"]),
        raw("f", ["F"]),
    ]
    outputs, receipt = build(rows, plan(blocked=["bridge"]))
    assert {row["id"] for row in outputs["calibration"]} == {"e", "f"}
    assert receipt["dropped"]["blocked_or_public_component"] == 2


def test_question_only_public_hit_removes_previously_queued_siblings():
    rows = [raw("a", ["Shared"]), raw("b", ["Shared"]), raw("e", ["E"]), raw("f", ["F"])]

    def hit(sample):
        return bool(sample.questions) and "establish b?" in sample.questions[0].instruction

    outputs, receipt = build(rows, public_hit=hit)
    assert {row["id"] for row in outputs["calibration"]} == {"e", "f"}
    assert receipt["dropped"]["blocked_or_public_component"] == 2


def test_public_question_on_unsupported_bridge_excludes_its_entire_component():
    rows = [
        raw("a", ["Shared"]),
        raw("bridge", ["Shared"], answer="unsupported"),
        raw("e", ["E"]),
        raw("f", ["F"]),
    ]

    def hit(sample):
        return bool(sample.questions) and "establish bridge?" in sample.questions[0].instruction

    outputs, receipt = build(rows, public_hit=hit)
    assert {row["id"] for row in outputs["calibration"]} == {"e", "f"}
    assert receipt["dropped"]["unsupported_bridge"] == 1
    assert receipt["dropped"]["blocked_or_public_component"] == 1


def test_complete_original_overflow_excludes_decision_but_preserves_raw_closure():
    rows = [raw("a", ["A"]), raw("long", ["A", "B"], text="x" * 3000), raw("b", ["B"])]
    outputs, receipt = build(rows, plan(components=1))
    assert {r["id"] for r in outputs["calibration"]} == {"a", "b"}
    assert receipt["dropped"]["complete_input_overflow"] == 1
    assert receipt["paragraph_audit"]["component_audit"]["components_per_role"] == {
        "calibration": 1
    }


def test_roles_are_assigned_before_selection_without_gold_and_order_effects():
    rows = [raw(str(i), [f"Title{i}"]) for i in range(50)]
    declared = plan(roles={"calibration": 1, "dev": 1, "test": 1}, quota=3, components=3)
    before = copy.deepcopy((rows, declared))
    outputs, receipt = build(iter(rows), declared)
    reverse_outputs, reverse_receipt = build(list(reversed(rows)), declared)
    assert (outputs, receipt) == (reverse_outputs, reverse_receipt)
    assert (rows, declared) == before
    memberships = receipt["paragraph_audit"]["component_audit"]["membership"]
    assert len(set().union(*(set(v.values()) for v in memberships.values()))) == 9
    changed = copy.deepcopy(rows)
    changed[0]["answer"] = "no"
    _, changed_receipt = build(changed, declared)
    assert changed_receipt["role_assignment_sha256"] == receipt["role_assignment_sha256"]


@pytest.mark.parametrize(
    "damage",
    [
        "raw_anchor",
        "duplicate",
        "missing_component_cell",
        "zero",
        "unknown_block",
        "no_public_check",
        "shortfall",
    ],
)
def test_builder_fails_before_any_output_on_incomplete_preparation(damage):
    rows = [raw("a", ["A"]), raw("b", ["B"])]
    declared = plan()
    anchor = builder.raw_digest(rows)
    checker = SimpleNamespace(sample_hit=lambda sample: False)
    if damage == "raw_anchor":
        rows[0]["answer"] = "no"
    elif damage == "duplicate":
        rows[1]["id"] = "a"
    elif damage == "missing_component_cell":
        declared["minimum_components"] = {}
    elif damage == "zero":
        declared["minimum_counts"]["calibration/hotpot_val/noul"] = 0
    elif damage == "unknown_block":
        declared["blocked_raw_ids"] = ["missing"]
    elif damage == "no_public_check":
        checker = None
    else:
        declared = plan(quota=3)
    with pytest.raises(ValueError):
        builder.build_hotpot_splits(
            rows,
            declared,
            ToyTokenizer(),
            tiny_config(),
            expected_raw_sha256=anchor,
            decontaminator=checker,
        )


def test_round_robin_prefers_distinct_components_over_many_sibling_questions():
    rows = [raw(f"shared{i}", ["Shared"]) for i in range(8)] + [raw("a", ["A"]), raw("b", ["B"])]
    _, receipt = build(rows, plan(quota=3, components=3))
    assert receipt["paragraph_audit"]["component_audit"]["components_per_role"] == {
        "calibration": 3
    }


@pytest.mark.parametrize("drift", [False, True])
def test_actual_fast_swift_recipe_is_stable_for_entire_raw_build(monkeypatch, drift):
    from test_evidence_swift_bridge import tokenizer

    tok = HFTokenizer(tokenizer(), "offline-fixture")
    cfg = replace(tiny_config(), readout="lm", max_seq_len=2048)
    rows = [raw("a", ["A"]), raw("b", ["B"])]
    declared = plan(limit=2048)
    declared["input_encoding"] = {"encoder": "swift_canonical"}
    original = builder.encode_direct_sample
    calls = []

    def encode(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(1)
        if drift and len(calls) == 1:
            tok.hf.add_tokens(["CHANGED_RAW_PREPARATION_TOKEN"])
        return result

    monkeypatch.setattr(builder, "encode_direct_sample", encode)

    def run():
        return builder.build_hotpot_splits(
            rows,
            declared,
            tok,
            cfg,
            expected_raw_sha256=builder.raw_digest(rows),
            decontaminator=SimpleNamespace(sample_hit=lambda sample: False),
        )

    if drift:
        with pytest.raises(ValueError, match="tokenizer changed|recipe changed"):
            run()
    else:
        outputs, receipt = run()
        assert len(outputs["calibration"]) == 2
        assert receipt["input_recipe"]["readout"] == "canonical_letter_raw"


@pytest.mark.parametrize(
    "damage",
    [None, "empty", "irrelevant", "modified", "policy", "escape", "canonical", "candidate_object"],
)
def test_production_public_inventory_is_pinned_nonempty_and_effective(tmp_path, damage):
    data = (
        json.dumps(
            {
                "state": "Publicly evaluated confidential finance policy",
                "question": {"instructions": "Does evidence establish a?"},
            }
        )
        + "\n"
    ).encode()
    if damage == "empty":
        data = b"\n"
    elif damage == "irrelevant":
        data = b'{"state": "x"}\n'
    elif damage == "canonical":
        data = (
            json.dumps(
                {
                    "state": "Unrelated effective public state text",
                    "questions": [
                        {
                            "instruction": "Does evidence establish a?",
                            "candidates": [{"description": "Public candidate description"}],
                        }
                    ],
                }
            )
            + "\n"
        ).encode()
    elif damage == "candidate_object":
        data = (
            json.dumps(
                {
                    "state": "Effective public state text",
                    "question": {
                        "instructions": "Public question for evidence",
                        "criteria": {"a": {"description": "Ignored candidate content"}},
                    },
                }
            )
            + "\n"
        ).encode()
    path = tmp_path / "public.jsonl"
    path.write_bytes(data)
    manifest = {
        "version": "ayaka-public-exclusion-1",
        "format": "jevbench-singular-question",
        "policy": copy.deepcopy(builder.POLICY),
        "files_sha256": {"public.jsonl": digest(data)},
    }
    if damage == "modified":
        path.write_bytes(data + b" ")
    elif damage == "policy":
        manifest["policy"]["word_ngram"] = 12
    elif damage == "escape":
        manifest["files_sha256"] = {"../escape.jsonl": digest(data)}
    if damage:
        with pytest.raises(ValueError):
            builder.pinned_public_checker(json.dumps(manifest).encode(), tmp_path)
    else:
        checker, receipt = builder.pinned_public_checker(json.dumps(manifest).encode(), tmp_path)
        assert receipt["records"] == 1
        assert checker.sample_hit(Sample("Does evidence establish a?", []))
