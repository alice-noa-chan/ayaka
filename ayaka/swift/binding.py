"""Exact Swift receipts with rendered inputs and canonical raw-token provenance.

Uses the shared fingerprint convention. Swift supports grouped diagnostics,
so this versioned contract remains separate from read_artifact.make_binding.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from pathlib import Path

from ayaka.eval.read_artifact import fingerprint

from .grouping import option_groups
from .prompt import render_options, render_question
from .readers import READOUT, logmass_probs

CONTRACT = "swift_canonical_tokens_v2"


def make_swift_binding(
    item, reader, *, model, revision, prompt_variant, state_format, group_size, diagnostic=False
):
    messages = None
    if len(item.question.labels) <= 26:
        messages, _ = render_question(
            item.state, item.question, state_format=state_format, prompt_variant=prompt_variant
        )
    else:
        messages = []
        for indices in option_groups(len(item.question.labels), group_size):
            rendered, mapping = render_options(
                item.state,
                item.question.instruction,
                [item.question.labels[i] for i in indices],
                [item.question.descriptions[i] for i in indices],
                state_format=state_format,
                prompt_variant=prompt_variant,
                question_type=item.question.type,
            )
            messages.append(
                {"messages": rendered, "labels": list(mapping.values()), "letters": list(mapping)}
            )
    inputs = (
        [{"messages": messages, "letters": [chr(65 + i) for i in range(len(item.question.labels))]}]
        if len(item.question.labels) <= 26
        else messages
    )
    token_inputs = [reader.describe(value["messages"], value["letters"]) for value in inputs]
    runtime = {
        "backend": getattr(reader, "backend", type(reader).__name__),
        "readout": "grouped_approx"
        if len(item.question.labels) > 26
        else getattr(reader, "readout", "undeclared"),
        "single_pass_readout": getattr(reader, "readout", "undeclared"),
        "prompt_variant": prompt_variant,
        "state_format": state_format,
        "group_size": group_size,
        "logprobs_mode": getattr(reader, "logprobs_mode", "undeclared"),
        "chat_template_kwargs": getattr(reader, "chat_template_kwargs", {}),
        "dtype": getattr(reader, "dtype", None),
        "device": getattr(reader, "device", None),
        "tokenizer_model": getattr(reader, "tokenizer_model", model),
        "adapter_sha256": getattr(reader, "adapter_sha256", None),
        "implementation_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("readers.py", "prompt.py", "grouping.py")
        },
    }
    payload = {
        "contract": CONTRACT,
        "question_id": item.id,
        "split": item.split,
        "public": item.public,
        "case_id": item.case_id or item.cluster_id or item.id,
        "cluster_id": item.cluster_id or item.case_id or item.id,
        "lineage_ids": list(item.lineage_ids),
        "model": model,
        "revision": revision,
        "tokenizer_revision": getattr(reader, "tokenizer_revision", None) or revision,
        "runtime": runtime,
        "runtime_sha256": fingerprint(runtime),
        "readout": runtime["readout"],
        "prompt_variant": prompt_variant,
        "diagnostic": diagnostic,
        "token_inputs": token_inputs,
        "canonical_token_ids_sha256": fingerprint([v["canonical_token_ids"] for v in token_inputs]),
        "rendered_input_sha256": fingerprint(messages),
        "state": item.state,
        "question": asdict(item.question),
        "messages": messages,
        "labels": item.question.labels,
        "gold": item.gold,
        "gold_distribution": item.gold_distribution,
        "source": item.source,
        "tier": item.tier,
        "adapter": item.adapter,
    }
    return {**payload, "binding_sha256": fingerprint(payload)}


class SwiftReadIndex:
    """Require exact bindings before any inference; refuse legacy/corrupt reads."""

    def __init__(self, rows=()):
        self.rows = {}
        for row in rows:
            binding = row.get("binding", {})
            payload = {key: value for key, value in binding.items() if key != "binding_sha256"}
            if binding.get("contract") != CONTRACT or binding.get("binding_sha256") != fingerprint(
                payload
            ):
                raise ValueError(
                    "existing reads lack a valid Swift binding; recollect legacy reads"
                )
            if row.get("record_sha256") != fingerprint(
                {key: value for key, value in row.items() if key != "record_sha256"}
            ):
                raise ValueError("cached read record fingerprint mismatch")
            if (
                not row.get("id")
                or row.get("id") != binding.get("question_id")
                or row["id"] in self.rows
            ):
                raise ValueError("duplicate or inconsistent question ids in read artifact")
            for key in (
                "model",
                "revision",
                "tokenizer_revision",
                "runtime",
                "readout",
                "prompt_variant",
            ):
                if not binding.get(key) or (
                    key != "runtime"
                    and (not isinstance(binding[key], str) or not binding[key].strip())
                ):
                    raise ValueError(f"bound reads require non-empty {key}")
            runtime = binding["runtime"]
            if not isinstance(runtime, dict) or binding.get("runtime_sha256") != fingerprint(
                runtime
            ):
                raise ValueError("runtime recipe fingerprint mismatch")
            for key in (
                "model",
                "revision",
                "tokenizer_revision",
                "prompt_variant",
                "readout",
                "split",
                "public",
                "source",
                "case_id",
                "cluster_id",
                "lineage_ids",
                "labels",
                "gold",
                "gold_distribution",
                "diagnostic",
            ):
                if row.get(key) != binding.get(key):
                    raise ValueError(f"read row differs from binding: {key}")
            if row.get("binding_sha256") != binding["binding_sha256"]:
                raise ValueError("row binding hash mismatch")
            if (
                runtime.get("readout") != binding["readout"]
                or runtime.get("prompt_variant") != binding["prompt_variant"]
            ):
                raise ValueError("readout/prompt_variant differs from runtime recipe")
            if binding.get("rendered_input_sha256") != fingerprint(binding.get("messages")):
                raise ValueError("rendered input fingerprint mismatch")
            inputs = binding.get("token_inputs")
            if not isinstance(inputs, list) or not inputs:
                raise ValueError("bound reads require actual token inputs")
            for value in inputs:
                prefix, ids = value.get("input_token_ids"), value.get("canonical_token_ids")
                if (
                    not isinstance(prefix, list)
                    or not prefix
                    or any(type(i) is not int or i < 0 for i in prefix)
                ):
                    raise ValueError("invalid prompt token ids")
                if (
                    not isinstance(ids, dict)
                    or not ids
                    or any(
                        not isinstance(v, list) or len(v) != 1 or type(v[0]) is not int or v[0] < 0
                        for v in ids.values()
                    )
                    or len({v[0] for v in ids.values()}) != len(ids)
                ):
                    raise ValueError("invalid canonical token id map")
                if value.get("input_token_ids_sha256") != fingerprint(prefix) or value.get(
                    "canonical_token_ids_sha256"
                ) != fingerprint(ids):
                    raise ValueError("token input fingerprint mismatch")
            if binding.get("canonical_token_ids_sha256") != fingerprint(
                [v["canonical_token_ids"] for v in inputs]
            ):
                raise ValueError("canonical id map fingerprint mismatch")
            if binding["readout"] == READOUT:
                if len(inputs) != 1 or set(inputs[0]["canonical_token_ids"]) != {
                    chr(65 + i) for i in range(len(binding["labels"]))
                }:
                    raise ValueError("canonical id map must cover every letter exactly")
                passes = row.get("pass_bindings")
                if not isinstance(passes, list) or len(passes) != 1:
                    raise ValueError("canonical raw reads require one complete gathered pass")
                gathered = passes[0]
                for key in ("input_token_ids", "canonical_token_ids"):
                    if gathered.get(key) != inputs[0][key]:
                        raise ValueError("gathered input/token ids differ from predeclared binding")
                logits = gathered.get("token_logits", {})
                required = {str(v[0]) for v in inputs[0]["canonical_token_ids"].values()}
                if set(logits) != required or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in logits.values()
                ):
                    raise ValueError("complete finite gathered raw logits required")
                masses = row.get("candidate_log_masses")
                if not isinstance(masses, dict) or set(masses) != set(binding["labels"]):
                    raise ValueError("complete raw log masses required in bound reads")
                for letter, label in zip(
                    inputs[0]["canonical_token_ids"], binding["labels"], strict=True
                ):
                    if masses[label] != logits[str(inputs[0]["canonical_token_ids"][letter][0])]:
                        raise ValueError("raw log masses differ from canonical gathered logits")
                expected = logmass_probs(masses)
                probs = row.get("raw_probs", {})
                if set(probs) != set(expected) or any(
                    type(probs[k]) not in (float, int)
                    or not math.isfinite(probs[k])
                    or not math.isclose(probs[k], p, abs_tol=1e-12, rel_tol=1e-9)
                    for k, p in expected.items()
                ):
                    raise ValueError("probabilities disagree with bound raw log masses")
            self.rows[row["id"]] = row

    def get(self, binding):
        row = self.rows.get(binding["question_id"])
        if row is not None and row["binding"] != binding:
            raise ValueError(
                "cached read differs in model or revision, prompt_variant, recipe, input, target or split"
            )
        return row


def validate_bound_reads(rows):
    """Strict fitting/selection entry; diagnostic and legacy reads need exploration."""
    SwiftReadIndex(rows)
    if any(
        row.get("readout") != READOUT
        or row.get("diagnostic")
        or row.get("comparison_valid") is False
        or row.get("promotable") is False
        or row["binding"]["runtime"].get("backend") not in ("hf", "vllm")
        or row["binding"]["runtime"].get("logprobs_mode") != "raw_logits"
        or row["binding"]["runtime"].get("single_pass_readout") != READOUT
        for row in rows
    ):
        raise ValueError("diagnostic reads require --exploratory")
