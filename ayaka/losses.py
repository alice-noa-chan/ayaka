"""Proper-scoring + distillation losses (docs.md section 16/40, A4).

Candidate-level targets are flat [n_c], aligned with the batch's
candidate order; per-question reduction uses ``cand_cu``. Score
questions carry per-candidate ordinals (addendum A3).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model.electra import SCORE, DecisionOutput
from .model.ragged import per_question_sum, ragged_max


def nll_loss(log_probs: torch.Tensor, targets: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    """Cross-entropy to the target distribution (log score)."""
    return (-per_question_sum(targets * log_probs, cu)).mean()


def brier_loss(probs: torch.Tensor, targets: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    return per_question_sum((probs - targets) ** 2, cu).mean()


def rps_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    cu: torch.Tensor,
    ordinals: torch.Tensor,
    score_mask: torch.Tensor,
) -> torch.Tensor:
    """Ranked Probability Score over Score questions (ordinal CDFs)."""
    losses = []
    cul = cu.tolist()
    for qi in torch.nonzero(score_mask).flatten().tolist():
        s, e = cul[qi], cul[qi + 1]
        k = e - s
        if k < 2:
            continue
        order = torch.argsort(ordinals[s:e])
        cp = probs[s:e][order].cumsum(0)[:-1]
        cy = targets[s:e][order].cumsum(0)[:-1]
        losses.append(((cp - cy) ** 2).sum() / (k - 1))
    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


def missing_evidence_loss(
    log_probs: torch.Tensor, cu: torch.Tensor, flagged: torch.Tensor, tau: float = 0.8
) -> torch.Tensor:
    """relu(p_max - tau)^2 on insufficient/conflicting-evidence questions (A4)."""
    if not flagged.any():
        return log_probs.sum() * 0.0
    p_max = ragged_max(log_probs, cu).exp()
    return (torch.relu(p_max - tau) ** 2)[flagged].mean()


def kl_distill(
    teacher_probs: torch.Tensor, student_log_probs: torch.Tensor, cu: torch.Tensor
) -> torch.Tensor:
    """KL(p_teacher || p_student) per question."""
    t = teacher_probs.clamp(min=1e-8)
    return per_question_sum(t * (t.log() - student_log_probs), cu).mean()


@dataclass
class LossWeights:
    nll: float = 1.0
    brier: float = 0.5
    rps: float = 0.35
    missing: float = 0.25
    pointer_aux: float = 0.3  # pointer-only NLL keeps the pointer usable alone (large sets)
    distill: float = 1.0  # KL to teacher, when teacher targets exist


def decision_loss(
    out: DecisionOutput,
    targets: torch.Tensor,
    *,
    ordinals: torch.Tensor | None = None,
    missing_mask: torch.Tensor | None = None,
    teacher_probs: torch.Tensor | None = None,
    teacher_mask: torch.Tensor | None = None,
    weights: LossWeights | None = None,
    missing_tau: float = 0.8,
) -> dict[str, torch.Tensor]:
    """Composite decision loss. With teacher targets (distillation),
    questions with teacher_mask use KL to the teacher in place of the
    gold NLL term; gold Brier/RPS still anchor them."""
    w = weights or LossWeights()
    cu = out.cand_cu
    logp = out.log_probs()
    p = logp.exp()
    parts: dict[str, torch.Tensor] = {}

    if teacher_probs is not None and teacher_mask is not None and teacher_mask.any():
        per_q_gold = -per_question_sum(targets * logp, cu)
        t = teacher_probs.clamp(min=1e-8)
        per_q_kl = per_question_sum(t * (t.log() - logp), cu)
        per_q = torch.where(teacher_mask, w.distill * per_q_kl, w.nll * per_q_gold)
        main = per_q.mean()
        parts["kl"] = per_q_kl[teacher_mask].mean().detach()
    else:
        main = w.nll * nll_loss(logp, targets, cu)
    parts["nll"] = nll_loss(logp, targets, cu).detach()
    total = main

    br = brier_loss(p, targets, cu)
    total = total + w.brier * br
    parts["brier"] = br.detach()

    if ordinals is not None:
        score_mask = out.primitive == SCORE
        if score_mask.any():
            rps = rps_loss(p, targets, cu, ordinals, score_mask)
            total = total + w.rps * rps
            parts["rps"] = rps.detach()

    if missing_mask is not None and missing_mask.any():
        miss = missing_evidence_loss(logp, cu, missing_mask, missing_tau)
        total = total + w.missing * miss
        parts["missing"] = miss.detach()

    if w.pointer_aux:
        aux = nll_loss(out.pointer_log_probs(), targets, cu)
        total = total + w.pointer_aux * aux
        parts["pointer_nll"] = aux.detach()

    parts["total"] = total
    return parts
