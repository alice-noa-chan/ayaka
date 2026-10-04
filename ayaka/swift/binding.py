"""Exact Swift cache bindings, distinct from the native raw-token contract.

read_artifact.make_binding/ReadIndex require raw native logits, actual input
token ids, and tokenizer fingerprints. The vLLM processed-logprob HTTP API
does not expose those. Grouped reads also exceed that contract's 26 labels.
This version therefore binds Swift's declared recipe and rendered messages
using the shared fingerprint helper, without claiming a native raw receipt.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from ayaka.eval.read_artifact import fingerprint

from .grouping import option_groups
from .prompt import render_options, render_question


def make_swift_binding(item, reader, *, model, revision, prompt_variant, state_format, group_size):
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
            )
            messages.append(
                {"messages": rendered, "labels": list(mapping.values()), "letters": list(mapping)}
            )
    runtime = {
        "backend": getattr(reader, "backend", type(reader).__name__),
        "readout": "grouped_approx"
        if len(item.question.labels) > 26
        else getattr(reader, "readout", "canonical_letter"),
        "single_pass_readout": getattr(reader, "readout", "canonical_letter"),
        "prompt_variant": prompt_variant,
        "state_format": state_format,
        "group_size": group_size,
        "logprobs_mode": getattr(reader, "logprobs_mode", "undeclared"),
        "chat_template_kwargs": getattr(reader, "chat_template_kwargs", {}),
        "dtype": getattr(reader, "dtype", None),
        "device": getattr(reader, "device", None),
        "implementation_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("readers.py", "prompt.py", "grouping.py")
        },
    }
    payload = {
        "contract": "swift_letter_messages_v1",
        "question_id": item.id,
        "split": item.split,
        "public": item.public,
        "case_id": item.case_id,
        "cluster_id": item.cluster_id,
        "lineage_ids": list(item.lineage_ids),
        "model": model,
        "revision": revision,
        "runtime": runtime,
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
            if binding.get("contract") != "swift_letter_messages_v1" or binding.get(
                "binding_sha256"
            ) != fingerprint(payload):
                raise ValueError(
                    "existing reads lack a valid Swift binding; recollect legacy reads"
                )
            if row.get("record_sha256") != fingerprint(
                {key: value for key, value in row.items() if key != "record_sha256"}
            ):
                raise ValueError("cached read record fingerprint mismatch")
            if row["id"] != binding["question_id"] or row["id"] in self.rows:
                raise ValueError("duplicate or inconsistent question ids in read artifact")
            self.rows[row["id"]] = row

    def get(self, binding):
        row = self.rows.get(binding["question_id"])
        if row is not None and row["binding"] != binding:
            raise ValueError(
                "cached read differs in model or revision, prompt_variant, recipe, input, target or split"
            )
        return row
