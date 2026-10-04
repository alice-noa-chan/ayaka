"""CPU preparation of gold-anchored direct items from saved reasoned reads.

This module never loads a model or generates a trace. The caller must supply
an independent gold verifier and saved teacher outputs. Hashes bind declared
question content and candidate order, not an attestation of teacher execution.
Corpus preparation alone is not a promotable checkpoint or a launch approval.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, replace

from ..eval.read_artifact import fingerprint
from ..eval.v2 import typed_row
from ..losses import LossWeights
from ..primitives import QuestionSpec
from .batching import _noul_canonical
from .prepare_v2 import audit_splits
from .swift_direct import (
    direct_readout_binding,
    encode_direct_sample,
    normalize_input_encoding,
    validate_direct_input_items,
)
from .tokenizer_identity import scoped_tokenizer_preparation

VERSION = "ayaka-direct-distillation-preparation-1"


def _sha(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be an exact lowercase SHA256")


def _distribution(values, count):
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)
        or not math.isclose(math.fsum(values), 1, abs_tol=1e-9, rel_tol=0)
    ):
        raise ValueError("probabilities must be finite, normalized and candidate-aligned")
    return list(values)


def question_fingerprint(sample, question):
    """Bind original evidence, full question, ordered options, soft gold and lineage."""
    from ..data.schema import Sample

    q = _noul_canonical(question)
    return fingerprint(
        Sample(
            sample.state,
            [q],
            {
                key: sample.metadata.get(key)
                for key in (
                    "split",
                    "source_example_id",
                    "source_lineage",
                    "generator_template_id",
                    "rule_combination",
                    "document_voice",
                    "translation_of",
                    "derived_from",
                )
            },
        ).to_json()
    )


def make_teacher_read(
    sample,
    question,
    *,
    probs,
    direct_probs,
    model_sha256,
    recipe_sha256,
    trace_sha256,
    generated_tokens,
    finish_reason="eos",
    direct_readout=None,
):
    """Snapshot a declared completed teacher read; preserve raw probabilities.

    A trace digest is provenance only: matching a gold answer does not prove
    that every intermediate step was correct. No trace text enters the student.
    """
    if sample.metadata.get("split") != "train":
        raise ValueError("reasoned teacher labels are restricted to train")
    if not any(q == question for q in sample.questions):
        raise ValueError("teacher question must belong to the sample")
    for name, value in (
        ("model_sha256", model_sha256),
        ("recipe_sha256", recipe_sha256),
        ("trace_sha256", trace_sha256),
    ):
        _sha(value, name)
    if (
        type(generated_tokens) is not int
        or not 1 <= generated_tokens <= 1024
        or finish_reason != "eos"
    ):
        raise ValueError("teacher requires a nonempty completed trace within the v2 budget")
    q = _noul_canonical(question)
    original_ids = [c.id for c in question.candidates]
    order = [original_ids.index(c.id) for c in q.candidates]
    p = _distribution(probs, len(question.candidates))
    d = _distribution(direct_probs, len(question.candidates))
    result = {
        "version": VERSION,
        "question_sha256": question_fingerprint(sample, question),
        "model_sha256": model_sha256,
        "recipe_sha256": recipe_sha256,
        "trace_sha256": trace_sha256,
        "generated_tokens": generated_tokens,
        "finish_reason": finish_reason,
        "route": "reasoned",
        "candidate_ids": [c.id for c in q.candidates],
        "probs": [p[i] for i in order],
        "direct_probs": [d[i] for i in order],
        "execution_attested": False,
    }
    if direct_readout is not None:
        if not isinstance(direct_readout, dict):
            raise ValueError("direct readout binding must be an object")
        result["direct_readout_binding"] = copy.deepcopy(direct_readout)
    return result


def _check_read(read, sample, q, item):
    if (
        not isinstance(read, dict)
        or read.get("version") != VERSION
        or read.get("route") != "reasoned"
    ):
        raise ValueError("a versioned reasoned teacher read is required")
    if read.get("question_sha256") != question_fingerprint(sample, q):
        raise ValueError("teacher content, candidate order, gold or lineage binding mismatch")
    if read.get("candidate_ids") != [c.id for c in q.candidates]:
        raise ValueError("teacher probabilities have a different candidate order")
    for key in ("model_sha256", "recipe_sha256", "trace_sha256"):
        _sha(read.get(key), key)
    tokens = read.get("generated_tokens")
    if type(tokens) is not int or not 1 <= tokens <= 1024 or read.get("finish_reason") != "eos":
        raise ValueError("teacher requires a nonempty completed trace within the v2 budget")
    if read.get("execution_attested") is not False:
        raise ValueError("this preparation contract does not attest backend execution")
    if item.direct_input_binding is not None and read.get(
        "direct_readout_binding"
    ) != direct_readout_binding(item):
        raise ValueError(
            "teacher direct probabilities require the exact Swift input/readout binding"
        )
    return _distribution(read.get("probs"), len(q.candidates)), _distribution(
        read.get("direct_probs"), len(q.candidates)
    )


def _teacher_filter(q, teacher, direct, target):
    spec = QuestionSpec(
        q.type,
        q.instruction,
        [c.description for c in q.candidates],
        [c.ordinal for c in q.candidates] if q.type == "score" else None,
    )
    a, b = typed_row(spec, direct, target), typed_row(spec, teacher, target)
    reasons = []
    if b["nll"] >= a["nll"] - 1e-9:
        reasons.append("no_gold_nll_gain")
    if b["brier"] > a["brier"] + 1e-9:
        reasons.append("gold_brier_regression")
    if q.type == "score":
        for key in ("rps", "nmae"):
            if b[key] > a[key] + 1e-9:
                reasons.append(f"score_{key}_regression")
    else:
        if b["correct"] < a["correct"] - 1e-9:
            reasons.append("thresholded_credit_regression")
        if max(target) == 1 and b["correct"] != 1:
            reasons.append("wrong_or_abstaining_hard_gold")
    return reasons


@scoped_tokenizer_preparation
def prepare_direct_distillation(
    splits,
    tok,
    cfg,
    teacher_reads,
    verify_gold,
    *,
    weights=None,
    development_only=False,
    input_encoding=None,
):
    """Audit reserved splits, then prepare only original-input train items.

    By default require all five splits for standalone legacy preparation.
    ``development_only`` explicitly audits four development splits; direct
    bundle v4 separately validates externally anchored test commitments.

    ``verify_gold(sample, question)`` must independently recompute the gold
    distribution from evidence; reading the stored target is not verification.
    Rejected teacher outputs leave their original gold item in the replay pool.
    The returned TrainItems can be consumed by the existing Trainer with the
    same explicit loss weights. No rationale CE positions are prepared.
    """
    weights = weights or LossWeights(gold_nll_with_teacher=True, distill=0.2)
    if (
        weights.gold_nll_with_teacher is not True
        or not math.isfinite(weights.nll)
        or weights.nll <= 0
        or not math.isfinite(weights.distill)
        or weights.distill < 0
    ):
        raise ValueError("direct distillation must retain positive gold NLL and nonnegative KL")
    if not callable(verify_gold):
        raise ValueError("an independent gold verifier is required")
    counts = audit_splits(splits, development_only=development_only)
    input_encoding = normalize_input_encoding(input_encoding)
    expected = {}
    for sample in splits["train"]:
        source = sample.metadata.get("source_example_id")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("explicit train source_example_id is required")
        for q in sample.questions:
            identity = f"{source}/{q.id}"
            if identity in expected:
                raise ValueError("duplicate train question identity")
            expected[identity] = (sample, q)
    if not isinstance(teacher_reads, dict) or set(teacher_reads) - set(expected):
        raise ValueError("teacher reads must belong exclusively to this train cohort")
    items, records = [], []
    for sample in splits["train"]:
        if sample.metadata.get("modality", "text") != "text" or any(
            key in sample.metadata for key in ("media", "proposal_supervision")
        ):
            raise ValueError("this direct-distillation arm supports text decisions only")
        clean = copy.deepcopy(sample)
        for key in ("verified_traces", "teacher_probs", "teacher"):
            clean.metadata.pop(key, None)
        direct_items = encode_direct_sample(clean, tok, cfg, input_encoding=input_encoding)
        for q, item in zip(clean.questions, direct_items, strict=True):
            q = _noul_canonical(q)
            identity = f"{sample.metadata['source_example_id']}/{q.id}"
            labels = [c.id for c in q.candidates]
            verified = verify_gold(copy.deepcopy(sample), copy.deepcopy(q))
            if not isinstance(verified, dict) or set(verified) != set(labels):
                raise ValueError("gold verifier must return all ordered candidate targets")
            target = _distribution([verified[label] for label in labels], len(labels))
            if target != item.target:
                raise ValueError("independently verified gold disagrees with stored targets")
            read = teacher_reads.get(identity)
            teacher, reasons = None, ["no_saved_teacher"]
            if read is not None:
                teacher, direct = _check_read(read, sample, q, item)
                reasons = _teacher_filter(q, teacher, direct, target)
            accepted = not reasons
            items.append(
                replace(item, teacher=teacher if accepted else None, direct_distillation=True)
            )
            records.append(
                {
                    "id": identity,
                    "question_sha256": question_fingerprint(sample, q),
                    "input_sha256": fingerprint(item.enc.prefix_ids + item.enc.rendered.suffix_ids),
                    "input_tokens": item.length,
                    "teacher_accepted": accepted,
                    "reasons": reasons,
                    "direct_readout_binding": direct_readout_binding(item),
                }
            )
    validate_direct_input_items(items)
    return items, {
        "version": VERSION,
        "input_encoding": input_encoding,
        "promotable": False,
        "execution_attested": False,
        "split_counts": counts,
        "loss_weights": asdict(weights),
        "split_sha256": {
            split: fingerprint([sample.to_json() for sample in samples])
            for split, samples in splits.items()
        },
        "teacher_reads_sha256": fingerprint(teacher_reads),
        "items": records,
        "accepted_teacher_questions": sum(r["teacher_accepted"] for r in records),
        "gold_replay_questions": sum(not r["teacher_accepted"] for r in records),
        "direct_training_tokens": sum(r["input_tokens"] for r in records),
        "reasoning_training_tokens": 0,
        "scope": "CPU preparation; teacher execution, trained weights and launch cost remain unverified",
    }
