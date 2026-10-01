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


def supervised_forward(model, input_ids, attention_mask, items, answers=None, *, prune=True):
    """Keep only answer/CE query positions in Gemma's independent KV-shared layers."""
    if not prune or not model.prune_shared_positions:
        output = model.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=model.span_layer is not None,
        )
        hidden = output.last_hidden_state
        spans = output.hidden_states[model.span_layer] if model.span_layer is not None else hidden
        return hidden, spans, items
    from ..model.fastpath import forward_kept

    positions, proxies = [], []
    for row, item in enumerate(items):
        wanted = ([answers[row]] if answers is not None else []) + (item.reasoning_positions or [])
        if not wanted:
            wanted = [len(item.enc.prefix_ids) + len(item.enc.rendered.suffix_ids) - 1]
        positions.append(wanted)
        offset = int(answers is not None)
        proxies.append(
            replace(
                item,
                reasoning_positions=list(range(offset, offset + len(item.reasoning_labels or []))),
            )
        )
    width = max(map(len, positions))
    keep = torch.tensor(
        [row + [row[-1]] * (width - len(row)) for row in positions], device=input_ids.device
    )
    hidden, spans = forward_kept(
        model.text_model(), input_ids, keep, attention_mask=attention_mask, normalize_spans=False
    )
    return hidden, spans, proxies


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


def trace_ce(model, hidden, items, chunk_tokens=128):
    """Batch supervised positions, retaining the original per-question weighting."""
    if chunk_tokens < 1:
        raise ValueError("CE chunk size must be positive")
    states, targets, weights = [], [], []
    for row, item in enumerate(items):
        if not item.reasoning_labels:
            continue
        positions = torch.tensor(item.reasoning_positions, device=hidden.device)
        labels = torch.tensor(item.reasoning_labels, device=hidden.device)
        states.append(hidden[row, positions])
        targets.append(labels)
        weights.append(torch.full_like(labels, 1 / (len(labels) * len(items)), dtype=torch.float32))
    loss = hidden[0, 0, 0] * 0
    if not states:
        return loss
    states, targets, weights = torch.cat(states), torch.cat(targets), torch.cat(weights)
    for start in range(0, len(targets), chunk_tokens):
        logits = model.lm_logits(states[start : start + chunk_tokens])
        token_loss = F.cross_entropy(
            logits.float(), targets[start : start + chunk_tokens], reduction="none"
        )
        loss = loss + (token_loss * weights[start : start + chunk_tokens]).sum()
    return loss
