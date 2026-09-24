"""Samples -> question-level training items -> token-budgeted batches.

Each question becomes one row (``prefix + suffix``); rows are grouped by
length under a token budget (rows x longest row) to keep padding low.
Teacher distributions for distillation ride along in sample metadata as
``teacher_probs: {question_id: [p...]}`` aligned with candidate order.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from ..collate import EncodedQuestion, encode_decision, full_rows
from ..config import ElectraConfig
from ..data.schema import Question, Sample
from ..model.electra import DecisionBatch
from ..prompt import QuestionView
from ..tokenization import Tokenizer

FLAGGED_EVIDENCE = {"deleted", "partial", "contradictory"}


@dataclass
class TrainItem:
    enc: EncodedQuestion
    target: list[float]  # aligned with candidates (input order)
    ordinals: list[int]  # score ordinals, else candidate index
    type: str
    flagged: bool
    teacher: list[float] | None
    sample_id: str

    @property
    def length(self) -> int:
        return len(self.enc.prefix_ids) + len(self.enc.rendered.suffix_ids)


def _noul_canonical(q: Question) -> Question:
    """Noul candidates must be [false, true] (rendering convention)."""
    if q.type != "noul" or [c.id for c in q.candidates] == ["false", "true"]:
        return q
    by_id = {c.id: c for c in q.candidates}
    if set(by_id) != {"false", "true"}:
        raise ValueError(f"noul question {q.id} needs candidates 'false'/'true'")
    return Question(
        q.id, q.type, q.instruction, [by_id["false"], by_id["true"]], q.target_distribution
    )


def question_view(q: Question) -> QuestionView:
    ords = [c.ordinal for c in q.candidates] if q.type == "score" else None
    return QuestionView(q.type, q.instruction, [c.description for c in q.candidates], ords)


def sample_to_items(sample: Sample, tok: Tokenizer, cfg: ElectraConfig) -> list[TrainItem]:
    qs = [_noul_canonical(q) for q in sample.questions]
    _, encs = encode_decision(
        sample.state, [question_view(q) for q in qs], tok, cfg.max_seq_len, cfg.max_label_candidates
    )
    teacher = sample.metadata.get("teacher_probs") or {}
    flagged = sample.metadata.get("evidence_state", "intact") in FLAGGED_EVIDENCE
    sid = str(sample.metadata.get("source_example_id", ""))
    items = []
    for q, enc in zip(qs, encs, strict=True):
        t = teacher.get(q.id)
        items.append(
            TrainItem(
                enc=enc,
                target=[q.target_distribution.get(c.id, 0.0) for c in q.candidates],
                ordinals=[
                    c.ordinal if c.ordinal is not None else i for i, c in enumerate(q.candidates)
                ],
                type=q.type,
                flagged=flagged,
                teacher=list(t) if t is not None and len(t) == len(q.candidates) else None,
                sample_id=sid,
            )
        )
    return items


def budget_batches(
    items: list[TrainItem], max_tokens: int, max_rows: int = 64, shuffle_seed: int | None = 0
) -> list[list[TrainItem]]:
    """Length-sorted groups with rows * longest <= max_tokens."""
    order = sorted(items, key=lambda it: it.length)
    batches: list[list[TrainItem]] = []
    cur: list[TrainItem] = []
    for it in order:
        longest = max([it.length] + [c.length for c in cur])
        if cur and (longest * (len(cur) + 1) > max_tokens or len(cur) >= max_rows):
            batches.append(cur)
            cur = []
        cur.append(it)
    if cur:
        batches.append(cur)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(batches)
    return batches


@dataclass
class TrainTensors:
    batch: DecisionBatch
    targets: torch.Tensor  # [n_c]
    ordinals: torch.Tensor  # [n_c]
    flagged: torch.Tensor  # [n_q] bool
    teacher: torch.Tensor | None  # [n_c]
    teacher_mask: torch.Tensor | None  # [n_q] bool

    def to(self, device) -> TrainTensors:
        mv = lambda t: t.to(device) if t is not None else None  # noqa: E731
        return TrainTensors(
            self.batch.to(device),
            mv(self.targets),
            mv(self.ordinals),
            mv(self.flagged),
            mv(self.teacher),
            mv(self.teacher_mask),
        )


def collate_items(items: list[TrainItem], pad_id: int) -> TrainTensors:
    batch = full_rows([it.enc for it in items], pad_id)
    targets = torch.tensor([p for it in items for p in it.target], dtype=torch.float32)
    ordinals = torch.tensor([o for it in items for o in it.ordinals], dtype=torch.long)
    flagged = torch.tensor([it.flagged for it in items], dtype=torch.bool)
    teacher = teacher_mask = None
    if any(it.teacher is not None for it in items):
        teacher = torch.tensor(
            [p for it in items for p in (it.teacher if it.teacher is not None else it.target)],
            dtype=torch.float32,
        )
        teacher_mask = torch.tensor([it.teacher is not None for it in items], dtype=torch.bool)
    return TrainTensors(batch, targets, ordinals, flagged, teacher, teacher_mask)
