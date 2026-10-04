"""Bound native text reads, preserving soft targets and unsaturated log masses.

This CPU-only contract binds declared inputs; it does not attest that a backend
used the declared raw-logit mode or that a tokenizer alias set is complete.
"""

from __future__ import annotations

import hashlib
import json
import math

VERSION = 1
SPLITS = {"train", "calibration", "dev", "test", "public"}
KINDS = {"choice", "noul", "score"}


def _copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def fingerprint(value):
    """Canonical JSON hash; list order and actual rendered strings are retained."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def _sha(value, name, sizes=(64,)):
    if (
        not isinstance(value, str)
        or len(value) not in sizes
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} needs an exact lowercase digest/revision")


def _labels(labels):
    if not isinstance(labels, list) or not 2 <= len(labels) <= 26:
        raise ValueError("native single-position reads require 2–26 ordered labels")
    for label in labels:
        _text(label, "label")
    if len(set(labels)) != len(labels):
        raise ValueError("candidate labels must be unique")


def _distribution(labels, values):
    if not isinstance(values, dict) or set(values) - set(labels):
        raise ValueError("target distribution contains unknown labels")
    result = {label: values.get(label, 0.0) for label in labels}
    if any(
        type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in result.values()
    ) or not math.isclose(math.fsum(result.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("target distribution must be finite, nonnegative and normalized")
    return result


def resolve_target(labels, gold, gold_distribution=None):
    """Keep explicit canonical soft targets instead of their argmax gold hint."""
    _labels(labels)
    if isinstance(gold, str):
        if gold not in labels:
            raise ValueError("gold label is absent from candidates")
        hard = {label: float(label == gold) for label in labels}
    elif isinstance(gold, dict):
        hard = _distribution(labels, gold)
    else:
        raise ValueError("gold must be a label or target distribution")
    if gold_distribution is None:
        return hard
    soft = _distribution(labels, gold_distribution)
    if isinstance(gold, dict) and hard != soft:
        raise ValueError("two explicit target distributions disagree")
    return soft


def _runtime(runtime):
    if not isinstance(runtime, dict):
        raise ValueError("runtime recipe is required")
    for key in ("backend", "dtype"):
        _text(runtime.get(key), key)
    for key in ("model_revision", "tokenizer_revision"):
        _sha(runtime.get(key), key, sizes=(40, 64))
    for key in (
        "tokenizer_sha256",
        "chat_template_sha256",
        "reader_sha256",
        "prompt_recipe_sha256",
        "alias_vocabulary_sha256",
    ):
        _sha(runtime.get(key), key)
    if (
        runtime.get("logits_mode") != "raw"
        or runtime.get("decision_path") != "native_logits"
        or runtime.get("modality") != "text"
    ):
        raise ValueError("this contract supports raw, native, single-position text reads only")
    for key in ("context_limit", "vocab_size"):
        if type(runtime.get(key)) is not int or runtime[key] < 1:
            raise ValueError(f"{key} must be a positive integer")


def _aliases(aliases, labels, vocab_size):
    if not isinstance(aliases, dict) or set(aliases) != set(labels):
        raise ValueError("alias sets must cover every candidate exactly")
    seen = set()
    for ids in aliases.values():
        if not isinstance(ids, list) or not ids:
            raise ValueError("each candidate needs a nonempty native alias set")
        for token_id in ids:
            if type(token_id) is not int or not 0 <= token_id < vocab_size:
                raise ValueError("alias token ids must be valid vocabulary integers")
            if token_id in seen:
                raise ValueError("alias token ids must be distinct within and across candidates")
            seen.add(token_id)


def validate_binding(binding):
    if not isinstance(binding, dict) or type(binding.get("version")) is not int:
        raise ValueError("versioned read binding is required")
    if binding["version"] != VERSION:
        raise ValueError("unsupported read binding version")
    for key in ("question_id", "case_id"):
        _text(binding.get(key), key)
    if (
        not isinstance(binding.get("split"), str)
        or binding["split"] not in SPLITS
        or not isinstance(binding.get("type"), str)
        or binding["type"] not in KINDS
    ):
        raise ValueError("explicit supported split and decision type are required")
    lineages = binding.get("lineage_ids")
    if not isinstance(lineages, list) or not lineages:
        raise ValueError("explicit globally namespaced source lineages are required")
    for value in lineages:
        _text(value, "lineage_id")
    if len(set(lineages)) != len(lineages):
        raise ValueError("lineage ids must be unique")
    labels = binding.get("labels")
    _labels(labels)
    if binding["type"] == "noul" and labels != ["false", "true"]:
        raise ValueError("Noul candidates must be ordered false/true")
    for key in ("model_sha256", "input_sha256", "state_sha256", "runtime_sha256"):
        _sha(binding.get(key), key)
    _runtime(binding.get("runtime"))
    _aliases(binding.get("candidate_token_ids"), labels, binding["runtime"]["vocab_size"])
    if binding["runtime_sha256"] != fingerprint(binding["runtime"]):
        raise ValueError("runtime recipe fingerprint mismatch")
    if (
        type(binding.get("input_tokens")) is not int
        or not 0 < binding["input_tokens"] < binding["runtime"]["context_limit"]
    ):
        raise ValueError("input must fit context with one answer position reserved")
    target = _distribution(labels, binding.get("target_distribution"))
    if set(binding["target_distribution"]) != set(labels):
        raise ValueError("bound target must include every ordered candidate")
    return target


def make_binding(
    *,
    question_id,
    split,
    case_id,
    lineage_ids,
    model_sha256,
    runtime,
    kind,
    state,
    candidates,
    messages,
    input_token_ids,
    target_distribution,
    candidate_token_ids,
):
    """Bind exact rendered inputs without duplicating large prompts in each row."""
    if not isinstance(candidates, list) or any(not isinstance(c, dict) for c in candidates):
        raise ValueError("ordered candidate descriptors are required")
    labels = [candidate.get("id") for candidate in candidates]
    _labels(labels)
    _runtime(runtime)
    _aliases(candidate_token_ids, labels, runtime["vocab_size"])
    if not isinstance(lineage_ids, list) or any(
        not isinstance(value, str) for value in lineage_ids
    ):
        raise ValueError("explicit source lineage list is required")
    if (
        not isinstance(input_token_ids, list)
        or not input_token_ids
        or any(type(i) is not int or not 0 <= i < runtime["vocab_size"] for i in input_token_ids)
    ):
        raise ValueError("actual input token ids must be nonempty vocabulary integers")
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(message, dict)
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            for message in messages
        )
    ):
        raise ValueError("actual rendered text messages are required")
    binding = {
        "version": VERSION,
        "question_id": question_id,
        "split": split,
        "case_id": case_id,
        "lineage_ids": sorted(lineage_ids),
        "model_sha256": model_sha256,
        "runtime": _copy(runtime),
        "runtime_sha256": fingerprint(runtime),
        "type": kind,
        "labels": labels,
        "candidate_token_ids": _copy(candidate_token_ids),
        "target_distribution": _distribution(labels, target_distribution),
        "state_sha256": fingerprint(state),
        "input_sha256": fingerprint(
            {
                "state": state,
                "type": kind,
                "candidates": candidates,
                "messages": messages,
                "input_token_ids": input_token_ids,
            }
        ),
        "input_tokens": len(input_token_ids),
    }
    validate_binding(binding)
    return _copy(binding)


def _lse(values):
    peak = max(values)
    return peak + math.log(math.fsum(math.exp(value - peak) for value in values))


def _probabilities(masses, labels):
    if not isinstance(masses, dict) or set(masses) != set(labels):
        raise ValueError("raw log masses must cover every candidate")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in masses.values()):
        raise ValueError("raw log masses must be finite")
    peak = max(masses.values())
    weights = {label: math.exp(masses[label] - peak) for label in labels}
    total = math.fsum(weights.values())
    return {label: weight / total for label, weight in weights.items()}


def make_record(binding, token_logits):
    """Aggregate all declared aliases; never fill missing logits with fake mass."""
    validate_binding(binding)
    aliases = binding["candidate_token_ids"]
    required = {token_id for ids in aliases.values() for token_id in ids}
    if (
        not isinstance(token_logits, dict)
        or any(type(key) is not int for key in token_logits)
        or set(token_logits) != required
    ):
        raise ValueError(
            "gather every declared raw alias logit exactly; missing values are not zero"
        )
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in token_logits.values()):
        raise ValueError("native raw logits must be finite")
    peak = max(token_logits.values())
    shifted = {token_id: value - peak for token_id, value in token_logits.items()}
    if any(not math.isfinite(value) for value in shifted.values()):
        raise ValueError("native logit span exceeds finite arithmetic")
    masses = {label: _lse([shifted[i] for i in ids]) for label, ids in aliases.items()}
    payload = {
        "version": VERSION,
        "binding": _copy(binding),
        "binding_sha256": fingerprint(binding),
        "logit_offset": peak,
        "candidate_log_masses": masses,
        "raw_probs": _probabilities(masses, binding["labels"]),
    }
    return {**payload, "record_sha256": fingerprint(payload)}


def validate_record(record):
    if (
        not isinstance(record, dict)
        or type(record.get("version")) is not int
        or record["version"] != VERSION
    ):
        raise ValueError("unsupported versioned read record")
    binding = record.get("binding")
    validate_binding(binding)
    if record.get("binding_sha256") != fingerprint(binding):
        raise ValueError("read binding fingerprint mismatch")
    if type(record.get("logit_offset")) not in (int, float) or not math.isfinite(
        record["logit_offset"]
    ):
        raise ValueError("native logit offset must be finite")
    probabilities = _probabilities(record.get("candidate_log_masses"), binding["labels"])
    stored = record.get("raw_probs")
    if (
        not isinstance(stored, dict)
        or set(stored) != set(probabilities)
        or any(
            type(stored[label]) not in (int, float)
            or not math.isfinite(stored[label])
            or stored[label] < 0
            or not math.isclose(stored[label], value, rel_tol=1e-9, abs_tol=1e-12)
            for label, value in probabilities.items()
        )
    ):
        raise ValueError("stored probabilities disagree with native log masses")
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    if record.get("record_sha256") != fingerprint(payload):
        raise ValueError("read record fingerprint mismatch")
    return binding


def logspace_nll(record, temperature=1.0):
    """Retain the true CE even when serialized probabilities underflow to zero."""
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    binding = validate_record(record)
    masses = record["candidate_log_masses"]
    peak = max(masses.values())
    shifted = {label: (mass - peak) / temperature for label, mass in masses.items()}
    if any(not math.isfinite(value) for value in shifted.values()):
        raise ValueError("temperature-scaled logit span exceeds finite arithmetic")
    partition = _lse(list(shifted.values()))
    return math.fsum(
        target * (partition - shifted[label])
        for label, target in binding["target_distribution"].items()
    )


class ReadIndex:
    """Exact reuse only; caller must bind the new requested inputs before lookup."""

    def __init__(self, records=()):
        self._records = {}
        for record in records:
            binding = validate_record(record)
            key = binding["question_id"]
            if key in self._records:
                raise ValueError("duplicate question ids in read artifact")
            self._records[key] = _copy(record)

    def get(self, binding):
        validate_binding(binding)
        record = self._records.get(binding["question_id"])
        if record is None:
            return None
        if record["binding_sha256"] != fingerprint(binding):
            raise ValueError("cached read differs in model, recipe, input, target or split")
        return _copy(record)


def calibration_rows(records):
    """Permit a single declared model/recipe on calibration rows only."""
    rows = list(records)
    if not rows:
        raise ValueError("calibration requires nonempty reads")
    bindings = [validate_record(row) for row in rows]
    if any(binding["split"] != "calibration" for binding in bindings):
        raise ValueError("fitting accepts calibration only; dev/test/public/train are forbidden")
    if len({(b["model_sha256"], b["runtime_sha256"]) for b in bindings}) != 1:
        raise ValueError("calibration reads have different model/runtime recipes")
    ReadIndex(rows)
    return _copy(rows)


def assert_read_splits_isolated(splits):
    """Check declared case/lineage plus exact state/input overlap across splits."""
    seen = {}
    for split, rows in splits.items():
        if split not in SPLITS:
            raise ValueError("unsupported named split")
        rows = list(rows)
        ReadIndex(rows)
        for row in rows:
            binding = validate_record(row)
            if binding["split"] != split:
                raise ValueError("row split differs from its declared partition")
            keys = [
                ("question", binding["question_id"]),
                ("lineage", binding["case_id"]),
                *(("lineage", value) for value in binding["lineage_ids"]),
                ("state", binding["state_sha256"]),
                ("input", binding["input_sha256"]),
            ]
            for key in keys:
                if key in seen and seen[key] != split:
                    raise ValueError(f"{key[0]} overlap across {seen[key]} and {split}")
                seen[key] = split
