"""Offline prompt-only teachers with distinct native student/teacher bindings.

This is finite-candidate forward KL preparation, not a reproduction of OPCD.
No rationale is generated or inserted into the short-input student. Saved raw
gathers establish content consistency, not that the named weights executed.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, replace

from ..eval.read_artifact import fingerprint
from .direct_distillation import question_fingerprint
from .swift_direct import (
    direct_readout_binding,
    encode_direct_sample,
    normalize_input_encoding,
    swift_question,
)
from .teacher_artifacts import _anchor, _identity, _scope, teacher_dataset_items
from .tokenizer_identity import scoped_tokenizer_preparation

VERSION = "ayaka-prompt-context-teacher-1"


def _encodings(student, teacher):
    student, teacher = normalize_input_encoding(student), normalize_input_encoding(teacher)
    if (
        student["encoder"] != "swift_canonical"
        or teacher["encoder"] != "swift_canonical"
        or student["prompt_variant"] != "min"
        or teacher["prompt_variant"] != "cygnet"
        or {k: v for k, v in student.items() if k != "prompt_variant"}
        != {k: v for k, v in teacher.items() if k != "prompt_variant"}
    ):
        raise ValueError("prompt-only teachers require min -> cygnet with identical transport")
    return student, teacher


def _observation(row, declared, sample, wire, item, identity, encoding):
    bound, actual = row["binding"], direct_readout_binding(item)
    if any(key in row for key in ("reasoned_read", "trace", "trace_tokens", "generated_tokens")):
        raise ValueError("prompt-only observations cannot contain a generated trace")
    if (
        any(bound.get(key) != identity[key] for key in ("model", "revision", "tokenizer_revision"))
        or bound["runtime"].get("adapter_sha256") != identity["adapter_sha256"]
        or bound["runtime"].get("state_format") != encoding["state_format"]
        or bound["runtime"].get("chat_template_kwargs") != encoding["chat_template_kwargs"]
        or row["prompt_variant"] != encoding["prompt_variant"]
    ):
        raise ValueError("prompt teacher model, adapter or input recipe mismatch")
    if (
        fingerprint(bound.get("state")) != fingerprint(sample.state)
        or fingerprint(bound.get("question")) != fingerprint(asdict(wire))
        or row["labels"] != wire.labels
        or any(
            row.get(key) != value
            for key, value in {
                "id": declared.id,
                "case_id": declared.case_id,
                "cluster_id": declared.cluster_id,
                "lineage_ids": list(declared.lineage_ids),
                "source": declared.source,
                "tier": declared.tier,
                "type": wire.type,
                "family": declared.family,
                "adapter": "canonical",
                "gold": declared.gold,
                "gold_distribution": declared.gold_distribution,
            }.items()
        )
    ):
        raise ValueError("prompt teacher original evidence, gold or lineage mismatch")
    observed = bound["token_inputs"][0]
    if (
        list(observed["canonical_token_ids"]) != [chr(65 + i) for i in range(len(wire.labels))]
        or fingerprint(bound["messages"]) != actual["messages_sha256"]
        or observed["input_token_ids_sha256"] != actual["input_token_ids_sha256"]
        or observed["canonical_token_ids_sha256"] != actual["canonical_token_ids_sha256"]
    ):
        raise ValueError("prompt teacher tokens differ from the actual native encoding")
    if (
        type(row.get("input_tokens")) is not int
        or row["input_tokens"] != item.length
        or type(row.get("output_tokens")) is not int
        or row["output_tokens"] != 1
        or type(row.get("passes")) is not int
        or row["passes"] != 1
        or type(row.get("latency_s")) not in (int, float)
        or not math.isfinite(row["latency_s"])
        or row["latency_s"] <= 0
    ):
        raise ValueError("prompt teacher single-pass usage mismatch")
    return actual


def _payload(
    sample,
    original,
    tok,
    cfg,
    student_row,
    teacher_row,
    *,
    input_encoding,
    teacher_input_encoding,
    teacher_identity,
    direct_artifact_sha256,
    teacher_artifact_sha256,
    mechanics_only,
):
    if type(mechanics_only) is not bool:
        raise ValueError("explicit mechanics scope is required")
    _identity(teacher_identity, cfg, mechanics_only)
    student_encoding, teacher_encoding = _encodings(input_encoding, teacher_input_encoding)
    for row in (student_row, teacher_row):
        _scope([row], mechanics_only=mechanics_only)  # includes raw-logit -> probability checks
    single = replace(sample, questions=[original])
    declared = teacher_dataset_items([single])[0]
    q, wire, semantic = swift_question(original)
    student = encode_direct_sample(single, tok, cfg, input_encoding=student_encoding)[0]
    teacher = encode_direct_sample(
        single,
        tok,
        cfg,
        input_encoding=teacher_encoding,
        context_limit=max(cfg.max_seq_len, cfg.serve_max_seq_len),
    )[0]
    direct_binding = _observation(
        student_row, declared, sample, wire, student, teacher_identity, student_encoding
    )
    teacher_binding = _observation(
        teacher_row, declared, sample, wire, teacher, teacher_identity, teacher_encoding
    )
    runtimes = [row["binding"]["runtime"] for row in (student_row, teacher_row)]
    if {k: v for k, v in runtimes[0].items() if k != "prompt_variant"} != {
        k: v for k, v in runtimes[1].items() if k != "prompt_variant"
    } or teacher.length <= student.length:
        raise ValueError(
            "prompt-only pairs must keep the runtime fixed and lengthen only the prompt"
        )
    candidate_ids = [c.id for c in q.candidates]
    label_for_id = {candidate_id: label for label, candidate_id in semantic.items()}
    recipe = {
        "teacher_identity": teacher_identity,
        "input_encoding": student_encoding,
        "teacher_input_encoding": teacher_encoding,
        "student_runtime": runtimes[0],
        "teacher_runtime": runtimes[1],
    }
    return {
        "version": VERSION,
        "route": "prompt_context",
        "question_sha256": question_fingerprint(sample, original),
        "candidate_ids": candidate_ids,
        "probs": [teacher_row["raw_probs"][label_for_id[c]] for c in candidate_ids],
        "direct_probs": [student_row["raw_probs"][label_for_id[c]] for c in candidate_ids],
        "model_sha256": fingerprint(teacher_identity),
        "recipe_sha256": fingerprint(recipe),
        "teacher_identity": copy.deepcopy(teacher_identity),
        "input_encoding": student_encoding,
        "teacher_input_encoding": teacher_encoding,
        "direct_readout_binding": direct_binding,
        "teacher_readout_binding": teacher_binding,
        "observations": {
            "student": copy.deepcopy(student_row),
            "teacher": copy.deepcopy(teacher_row),
        },
        "observation_provenance": {
            "direct_artifact_sha256": direct_artifact_sha256,
            "teacher_artifact_sha256": teacher_artifact_sha256,
            "mechanics_only": mechanics_only,
        },
        "generated_tokens": 0,
        "finish_reason": "not_generated",
        "execution_attested": False,
    }


def check_prompt_read(read, sample, q, item, tok, cfg):
    """Rebuild both native contexts, then compare the complete derived contract."""
    from .direct_distillation import _distribution, _sha

    observations, provenance = read.get("observations"), read.get("observation_provenance")
    if (
        not isinstance(observations, dict)
        or set(observations) != {"student", "teacher"}
        or not isinstance(provenance, dict)
        or set(provenance)
        != {"direct_artifact_sha256", "teacher_artifact_sha256", "mechanics_only"}
    ):
        raise ValueError("prompt teacher requires both raw observations and their provenance")
    for key in ("direct_artifact_sha256", "teacher_artifact_sha256"):
        _sha(provenance[key], key)
    original = next(original for original in sample.questions if original.id == q.id)
    expected = _payload(
        sample,
        original,
        tok,
        cfg,
        observations["student"],
        observations["teacher"],
        input_encoding=read.get("input_encoding"),
        teacher_input_encoding=read.get("teacher_input_encoding"),
        teacher_identity=read.get("teacher_identity"),
        **provenance,
    )
    if (
        fingerprint(expected) != fingerprint(read)
        or read["direct_readout_binding"] != direct_readout_binding(item)
        or item.direct_input_binding is None
    ):
        raise ValueError("prompt teacher payload or short student native binding changed")
    return _distribution(read["probs"], len(q.candidates)), _distribution(
        read["direct_probs"], len(q.candidates)
    )


@scoped_tokenizer_preparation
def export_prompt_teachers(
    samples,
    tok,
    cfg,
    direct_reads,
    prompt_reads,
    *,
    input_encoding,
    teacher_input_encoding,
    teacher_identity,
    expected_direct_sha256,
    expected_prompt_sha256,
    verify_gold,
    mechanics_only=False,
):
    """Convert one complete non-public train pair, keeping benefit filtering separate."""
    samples = list(samples)  # cohort validation and construction must see the same iterable
    if not callable(verify_gold):
        raise ValueError("an independent gold verifier is required")
    _identity(teacher_identity, cfg, mechanics_only)
    student_encoding, teacher_encoding = _encodings(input_encoding, teacher_input_encoding)
    _anchor(direct_reads, expected_direct_sha256, "direct artifact")
    _anchor(prompt_reads, expected_prompt_sha256, "prompt artifact")
    _scope(direct_reads, mechanics_only=mechanics_only)
    _scope(prompt_reads, mechanics_only=mechanics_only)
    cohort = teacher_dataset_items(samples)
    expected_ids = {row.id for row in cohort}
    if {r["id"] for r in direct_reads} != expected_ids or {
        r["id"] for r in prompt_reads
    } != expected_ids:
        raise ValueError("prompt artifacts must match the complete declared train cohort")
    direct = {r["id"]: r for r in direct_reads}
    prompts = {r["id"]: r for r in prompt_reads}
    teachers = {}
    for sample in samples:
        for original in sample.questions:
            q, _, _ = swift_question(original)
            verified = verify_gold(copy.deepcopy(sample), copy.deepcopy(q))
            if not isinstance(verified, dict) or verified != {
                c.id: q.target_distribution.get(c.id, 0.0) for c in q.candidates
            }:
                raise ValueError("prompt teacher gold disagrees with the independent verifier")
            identity = sample.metadata["source_example_id"] + "/" + original.id
            teachers[identity] = _payload(
                sample,
                original,
                tok,
                cfg,
                direct[identity],
                prompts[identity],
                input_encoding=student_encoding,
                teacher_input_encoding=teacher_encoding,
                teacher_identity=teacher_identity,
                direct_artifact_sha256=expected_direct_sha256,
                teacher_artifact_sha256=expected_prompt_sha256,
                mechanics_only=mechanics_only,
            )
    if set(teachers) != expected_ids:
        raise ValueError("prompt teacher construction did not cover the declared train cohort")
    report = {
        "version": VERSION,
        "teacher_kind": "prompt_context",
        "teacher_identity": copy.deepcopy(teacher_identity),
        "input_encoding": student_encoding,
        "teacher_input_encoding": teacher_encoding,
        "direct_artifact_sha256": expected_direct_sha256,
        "teacher_artifact_sha256": expected_prompt_sha256,
        "teacher_reads_sha256": fingerprint(teachers),
        "available_teacher_questions": len(teachers),
        "usage": {
            "input_tokens": sum(row["input_tokens"] for row in [*direct_reads, *prompt_reads]),
            "output_tokens": 2 * len(teachers),
            "reasoning_tokens": 0,
            "backend_calls": 2 * len(teachers),
        },
        "usage_scope": "saved reads only; missing failed attempts need the collection runner log",
        "mechanics_only": mechanics_only,
        "execution_attested": False,
        "promotable": False,
        "scope": "native content-consistency conversion; no weight execution or performance attestation",
    }
    return teachers, report
