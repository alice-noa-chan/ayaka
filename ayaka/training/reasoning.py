"""Prepare verified traces with the exact v2 generation/readout layout."""

from dataclasses import replace

import torch
import torch.nn.functional as F

from ..collate import EncodedQuestion
from ..evidence_generation import chat_ids
from ..model.electra import PRIMITIVE_INDEX
from ..primitives import QuestionSpec
from ..reasoning_pipeline import readout_suffix, trace_messages
from .batching import _noul_canonical, sample_to_items


def reasoning_items(sample, tok, cfg, traces, *, include_direct=True):
    """Traces are supplied by the deterministic validation data builder.

    Refuse unknown, empty, overbudget, or truncated traces. Callers must keep
    the validator provenance in the sample manifest; arbitrary model output
    is not a verified trace and must not be passed here.
    """
    direct = sample_to_items(sample, tok, cfg)
    items = list(direct) if include_direct else []
    for question, original in zip(sample.questions, direct, strict=True):
        q = _noul_canonical(question)
        if q.id not in traces:
            continue
        notes = traces[q.id]
        if not isinstance(notes, str) or not notes.strip():
            raise ValueError("verified reasoning must be non-empty")
        spec = QuestionSpec(
            q.type,
            q.instruction,
            [c.description for c in q.candidates],
            [c.ordinal for c in q.candidates] if q.type == "score" else None,
        )
        prompt = chat_ids(tok, trace_messages(sample.state, spec))
        tokens = tok.encode(notes)
        hf = getattr(tok, "hf", None)
        eos = getattr(hf, "eos_token_id", getattr(tok, "eos_id", None))
        if eos is not None:
            tokens.append(eos)
        suffix = readout_suffix(tok, spec, cfg.max_label_candidates)
        if (
            len(tokens) > 1024
            or len(prompt) + len(tokens) + len(suffix.suffix_ids) > cfg.max_seq_len
        ):
            raise ValueError("complete verified reasoning does not fit; do not truncate it")
        enc = EncodedQuestion(prompt + tokens, suffix, PRIMITIVE_INDEX[q.type])
        items.append(
            replace(
                original,
                enc=enc,
                reasoning_positions=list(range(len(prompt) - 1, len(prompt) + len(tokens) - 1)),
                reasoning_labels=tokens,
            )
        )
    return items


def trace_ce(model, hidden, items, chunk_tokens=32):
    """Question-averaged trace CE, materializing only small vocabulary chunks."""
    losses = []
    for row, item in enumerate(items):
        if not item.reasoning_labels:
            continue
        positions = torch.tensor(item.reasoning_positions, device=hidden.device)
        labels = torch.tensor(item.reasoning_labels, device=hidden.device)
        loss = hidden.sum() * 0
        for start in range(0, len(labels), chunk_tokens):
            logits = model.lm_logits(hidden[row, positions[start : start + chunk_tokens]])
            loss = loss + F.cross_entropy(
                logits.float(), labels[start : start + chunk_tokens], reduction="sum"
            )
        losses.append(loss / len(labels))
    return sum(losses, hidden.sum() * 0) / len(items)
