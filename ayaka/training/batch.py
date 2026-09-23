"""Training batch construction: canonical Samples -> ragged tensors.

Produces the flat [n_c] target distributions, per-candidate ordinals,
score/missing masks, primitive indices, and optional router block
supervision aligned with the model's forward contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..collate import encode_candidate, encode_question, encode_state
from ..data.schema import Sample
from ..model.model import CHOICE, NOUL, SCORE
from ..tokenizer import Tokenizer

_PRIM = {"noul": NOUL, "choice": CHOICE, "score": SCORE}


@dataclass
class TrainBatch:
    inputs: dict  # kwargs for ElectraDecisionModel.forward
    targets: torch.Tensor  # [n_c]
    cand_ordinals: torch.Tensor  # [n_c]
    score_question_mask: torch.Tensor  # [n_q] bool
    missing_mask: torch.Tensor  # [n_q] bool
    block_target: torch.Tensor | None  # [n_c, n_blocks]
    block_mask: torch.Tensor | None  # [n_c, n_blocks]
    n_questions: int


def _blocks_per_state(state_len: int, block_size: int) -> int:
    return max(1, math.ceil(state_len / block_size))


def build_train_batch(
    samples: list[Sample],
    tokenizer: Tokenizer,
    block_size: int,
    device: torch.device | str = "cpu",
) -> TrainBatch:
    state_ids: list[int] = []
    state_cu = [0]
    question_ids: list[int] = []
    question_cu = [0]
    question_state_index: list[int] = []
    candidate_ids: list[int] = []
    candidate_cu = [0]
    candidate_question_index: list[int] = []
    primitive_index: list[int] = []
    targets: list[float] = []
    cand_ordinals: list[int] = []
    score_mask: list[bool] = []
    missing_mask: list[bool] = []
    # router supervision: (question_global_idx, [block ids in its state])
    q_block_targets: list[tuple[int, list[int]]] = []
    state_block_offsets = [0]
    n_states_blocks: list[int] = []

    qg = 0
    for si, sample in enumerate(samples):
        sids = encode_state(sample.state, tokenizer)
        state_ids += sids
        state_cu.append(len(state_ids))
        nb = _blocks_per_state(len(sids), block_size)
        n_states_blocks.append(nb)
        state_block_offsets.append(state_block_offsets[-1] + nb)
        ev_blocks = sample.metadata.get("evidence_blocks")
        flagged = sample.evidence_state in ("deleted", "partial", "contradictory")
        for q in sample.questions:
            question_ids += encode_question(q.instruction, tokenizer)
            question_cu.append(len(question_ids))
            question_state_index.append(si)
            primitive_index.append(_PRIM[q.type])
            score_mask.append(q.type == "score")
            missing_mask.append(flagged)
            if ev_blocks:
                q_block_targets.append((qg, list(ev_blocks)))
            for c in q.candidates:
                candidate_ids += encode_candidate(c.description, tokenizer)
                candidate_cu.append(len(candidate_ids))
                candidate_question_index.append(qg)
                targets.append(q.target_distribution.get(c.id, 0.0))
                cand_ordinals.append(c.ordinal if c.ordinal is not None else -1)
            qg += 1

    def t(x, dt=torch.long):
        return torch.tensor(x, dtype=dt, device=device)

    inputs = {
        "state_ids": t(state_ids),
        "state_cu": t(state_cu),
        "question_ids": t(question_ids),
        "question_cu": t(question_cu),
        "question_state_index": t(question_state_index),
        "candidate_ids": t(candidate_ids),
        "candidate_cu": t(candidate_cu),
        "candidate_question_index": t(candidate_question_index),
        "primitive_index": t(primitive_index),
        "apply_temperature": False,
    }
    n_c = len(targets)
    n_blocks_total = state_block_offsets[-1]
    block_target = block_mask = None
    if q_block_targets:
        block_target = torch.zeros(n_c, n_blocks_total, device=device)
        block_mask = torch.zeros(n_c, n_blocks_total, dtype=torch.bool, device=device)
        for qi, blocks in q_block_targets:
            si = question_state_index[qi]
            base = state_block_offsets[si]
            for c in range(n_c):
                if candidate_question_index[c] == qi:
                    for b in blocks:
                        block_target[c, base + b] = 1.0
                    block_mask[c, base : base + n_states_blocks[si]] = True
    return TrainBatch(
        inputs=inputs,
        targets=torch.tensor(targets, dtype=torch.float, device=device),
        cand_ordinals=t(cand_ordinals),
        score_question_mask=torch.tensor(score_mask, dtype=torch.bool, device=device),
        missing_mask=torch.tensor(missing_mask, dtype=torch.bool, device=device),
        block_target=block_target,
        block_mask=block_mask,
        n_questions=qg,
    )
