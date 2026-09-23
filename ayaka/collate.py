"""Ragged batch construction for decision inputs (addendum A6).

One sample = one state + questions[]. Many samples pack into one
padding-free batch: every axis is a flat token buffer plus cu_seqlens
and integer index maps — the BatchDescriptor contract of section 46.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model.model import CHOICE, NOUL, SCORE
from .serialization import serialize_state
from .tokenizer import Tokenizer

_PRIMITIVE_INDEX = {"noul": NOUL, "choice": CHOICE, "score": SCORE}


@dataclass
class DecisionInputs:
    state_ids: torch.Tensor
    state_cu: torch.Tensor
    question_ids: torch.Tensor
    question_cu: torch.Tensor
    question_state_index: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_cu: torch.Tensor
    candidate_question_index: torch.Tensor
    primitive_index: torch.Tensor  # [n_q]


def encode_state(state, tokenizer: Tokenizer) -> list[int]:
    return tokenizer.encode(serialize_state(state))


def encode_question(instruction: str, tokenizer: Tokenizer) -> list[int]:
    return tokenizer.encode(f"<q> <instruction> {instruction} </instruction> </q>")


def encode_candidate(description: str, tokenizer: Tokenizer) -> list[int]:
    return tokenizer.encode(f"<candidate> {description} </candidate>")


def build_decision_inputs(
    samples: list[dict],
    tokenizer: Tokenizer,
    device: torch.device | str = "cpu",
) -> DecisionInputs:
    """samples: [{"state": ..., "questions": [{"type":..., "instruction":...,
    "candidates":[...], "ordinals":[...]?}, ...]}]
    """
    state_ids: list[int] = []
    state_cu = [0]
    question_ids: list[int] = []
    question_cu = [0]
    question_state_index: list[int] = []
    candidate_ids: list[int] = []
    candidate_cu = [0]
    candidate_question_index: list[int] = []
    primitive_index: list[int] = []

    q_global = 0
    for state_idx, sample in enumerate(samples):
        state_ids += encode_state(sample["state"], tokenizer)
        state_cu.append(len(state_ids))
        for q in sample["questions"]:
            question_ids += encode_question(q["instruction"], tokenizer)
            question_cu.append(len(question_ids))
            question_state_index.append(state_idx)
            primitive_index.append(_PRIMITIVE_INDEX[q["type"]])
            for cand in q["candidates"]:
                candidate_ids += encode_candidate(cand, tokenizer)
                candidate_cu.append(len(candidate_ids))
                candidate_question_index.append(q_global)
            q_global += 1

    def t(x):
        return torch.tensor(x, dtype=torch.long, device=device)

    return DecisionInputs(
        state_ids=t(state_ids),
        state_cu=t(state_cu),
        question_ids=t(question_ids),
        question_cu=t(question_cu),
        question_state_index=t(question_state_index),
        candidate_ids=t(candidate_ids),
        candidate_cu=t(candidate_cu),
        candidate_question_index=t(candidate_question_index),
        primitive_index=t(primitive_index),
    )
