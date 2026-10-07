"""Rendered questions -> DecisionBatch tensors.

Two layouts share one batch type:

- ``full_rows``: every question is its own row ``prefix + suffix``
  (training, calibration, teacher labeling — simple and exact).
- ``suffix_rows``: suffixes only, attending to a prefix KV cache that
  was encoded once and repeated per question (serving path). Isolation
  is identical: a suffix sees the prefix and itself, never a sibling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model.decision import PRIMITIVE_INDEX, DecisionBatch
from .prompt import QuestionView, RenderedQuestion, render_prefix, render_question
from .tokenization import Tokenizer


@dataclass
class EncodedQuestion:
    prefix_ids: list[int]
    rendered: RenderedQuestion
    primitive: int


def encode_decision(
    state,
    questions: list[QuestionView],
    tok: Tokenizer,
    max_seq_len: int,
    max_label_candidates: int = 26,
) -> tuple[list[int], list[EncodedQuestion]]:
    """Render one state + its questions; the state is truncated so the
    longest question still fits in ``max_seq_len``."""
    rendered = [render_question(q, tok, max_label_candidates) for q in questions]
    longest = max(len(r.suffix_ids) for r in rendered)
    overhead = len(render_prefix("", tok))
    budget = max(64, max_seq_len - longest - overhead)
    prefix = render_prefix(state, tok, max_state_tokens=budget)
    enc = [
        EncodedQuestion(prefix, r, PRIMITIVE_INDEX[q.type])
        for q, r in zip(questions, rendered, strict=True)
    ]
    return prefix, enc


def _cand_fields(items: list[EncodedQuestion], offsets: list[int]):
    cu, cq, spans, labels, has_label = [0], [], [], [], []
    tokens, token_offsets = [], [0]
    for qi, (it, off) in enumerate(zip(items, offsets, strict=True)):
        r = it.rendered
        n = len(r.option_spans)
        cu.append(cu[-1] + n)
        cq.extend([qi] * n)
        spans.extend([(s + off, e + off) for s, e in r.option_spans])
        for start, end in r.option_spans:
            tokens.extend((qi, pos + off) for pos in range(start, end))
            token_offsets.append(len(tokens))
        labels.extend(r.label_ids if r.label_ids is not None else [0] * n)
        has_label.append(r.label_ids is not None)
    return (
        torch.tensor(cu, dtype=torch.long),
        torch.tensor(cq, dtype=torch.long),
        torch.tensor(spans, dtype=torch.long).view(-1, 2),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(has_label, dtype=torch.bool),
        torch.tensor(tokens, dtype=torch.long).view(-1, 2),
        torch.tensor(token_offsets, dtype=torch.long),
    )


def full_rows(items: list[EncodedQuestion], pad_id: int) -> DecisionBatch:
    rows = [it.prefix_ids + it.rendered.suffix_ids for it in items]
    length = max(len(r) for r in rows)
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    cu, cq, spans, labels, has_label, tokens, token_offsets = _cand_fields(
        items, [len(it.prefix_ids) for it in items]
    )
    return DecisionBatch(
        input_ids=ids,
        attention_mask=mask,
        answer_pos=torch.tensor([len(r) - 1 for r in rows], dtype=torch.long),
        cand_cu=cu,
        cand_question=cq,
        cand_spans=spans,
        label_ids=labels,
        has_label=has_label,
        primitive=torch.tensor([it.primitive for it in items], dtype=torch.long),
        seq_len=torch.tensor([len(r) for r in rows], dtype=torch.long),
        cand_tokens=tokens,
        cand_token_offsets=token_offsets,
    )


def suffix_rows(items: list[EncodedQuestion], prefix_len: int, pad_id: int) -> DecisionBatch:
    sufs = [it.rendered.suffix_ids for it in items]
    length = max(len(s) for s in sufs)
    ids = torch.full((len(sufs), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sufs), prefix_len + length), dtype=torch.long)
    mask[:, :prefix_len] = 1
    for i, s in enumerate(sufs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, prefix_len : prefix_len + len(s)] = 1
    cu, cq, spans, labels, has_label, tokens, token_offsets = _cand_fields(items, [0] * len(items))
    return DecisionBatch(
        input_ids=ids,
        attention_mask=mask,
        answer_pos=torch.tensor([len(s) - 1 for s in sufs], dtype=torch.long),
        cand_cu=cu,
        cand_question=cq,
        cand_spans=spans,
        label_ids=labels,
        has_label=has_label,
        primitive=torch.tensor([it.primitive for it in items], dtype=torch.long),
        position_ids=(prefix_len + torch.arange(length)).unsqueeze(0).expand(len(sufs), -1),
        seq_len=torch.tensor([prefix_len + len(x) for x in sufs], dtype=torch.long),
        cand_tokens=tokens,
        cand_token_offsets=token_offsets,
    )
