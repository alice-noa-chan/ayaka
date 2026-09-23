"""Proper-scoring + distillation losses (docs.md section 16/40, A4/A9).

All losses are ragged-aware: candidate-level targets are flat [n_c]
aligned with the batch's candidate order; per-question reduction uses
cand_cu boundaries. Score questions additionally carry per-candidate
ordinals (addendum A3).
"""

from __future__ import annotations

import torch

from .model.model import DecisionOutput
from .model.pointer import ragged_max


def _seg_ids(cu: torch.Tensor) -> torch.Tensor:
    lens = (cu[1:] - cu[:-1]).long()
    return torch.repeat_interleave(torch.arange(lens.numel(), device=cu.device), lens)


def _per_question_sum(x: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(cu.numel() - 1, dtype=x.dtype, device=x.device)
    out.index_add_(0, _seg_ids(cu), x)
    return out


def nll_loss(out: DecisionOutput, targets: torch.Tensor) -> torch.Tensor:
    """L_log = mean_q -Σ_i y_i log p_i (log score / cross-entropy to dist)."""
    logp = out.log_probs()
    return (-_per_question_sum(targets * logp, out.cand_cu)).mean()


def brier_loss(out: DecisionOutput, targets: torch.Tensor) -> torch.Tensor:
    """L_brier = mean_q Σ_i (p_i - y_i)^2."""
    p = out.probs()
    return _per_question_sum((p - targets) ** 2, out.cand_cu).mean()


def rps_loss(
    out: DecisionOutput,
    targets: torch.Tensor,
    cand_ordinals: torch.Tensor,
    score_question_mask: torch.Tensor,
) -> torch.Tensor:
    """Ranked Probability Score over Score questions only.

    cand_ordinals: [n_c] ordinal level per candidate (A3).
    score_question_mask: [n_q] bool — which questions are scored.
    """
    p = out.probs()
    cu = out.cand_cu
    losses = []
    for qi in torch.nonzero(score_question_mask).squeeze(1).tolist():
        s, e = int(cu[qi]), int(cu[qi + 1])
        k = e - s
        if k < 2:
            continue
        order = torch.argsort(cand_ordinals[s:e])
        cp = p[s:e][order].cumsum(0)[:-1]
        cy = targets[s:e][order].cumsum(0)[:-1]
        losses.append(((cp - cy) ** 2).sum() / (k - 1))
    if not losses:
        return p.sum() * 0.0
    return torch.stack(losses).mean()


def permutation_consistency(
    log_probs1: torch.Tensor,
    log_probs2_aligned: torch.Tensor,
    cand_cu: torch.Tensor,
) -> torch.Tensor:
    """Symmetric KL between two aligned runs of the same semantic set.

    log_probs2_aligned must already be reordered into run1's candidate
    order by the caller (via its permutation map).
    """
    p1, p2 = log_probs1.exp(), log_probs2_aligned.exp()
    kl = _per_question_sum(p1 * (log_probs1 - log_probs2_aligned), cand_cu)
    kl += _per_question_sum(p2 * (log_probs2_aligned - log_probs1), cand_cu)
    return (kl * 0.5).mean()


def missing_evidence_loss(
    out: DecisionOutput, flagged_mask: torch.Tensor, tau: float = 0.8
) -> torch.Tensor:
    """Overconfidence penalty on insufficient/conflicting evidence (A4).

    flagged_mask: [n_q] bool for samples with deleted/partial/
    contradictory evidence. Penalizes relu(p_max - tau)^2.
    """
    if not flagged_mask.any():
        return out.logits.sum() * 0.0
    p_max = ragged_max(out.log_probs(), out.cand_cu).exp()
    pen = torch.relu(p_max - tau) ** 2
    return pen[flagged_mask].mean()


def kl_distill(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    cand_cu: torch.Tensor,
) -> torch.Tensor:
    """KL(p_teacher || p_student) per question, aligned flat layout."""
    t = teacher_log_probs.exp()
    per_q = _per_question_sum(t * (teacher_log_probs - student_log_probs), cand_cu)
    return per_q.mean()


def router_bce(
    route_probs: torch.Tensor,
    block_target: torch.Tensor,
    block_mask: torch.Tensor,
) -> torch.Tensor:
    """Supervised router loss (A9): BCE on per-candidate block scores.

    route_probs:   [n_c, n_blocks]
    block_target:  [n_c, n_blocks] float in {0,1} — evidence blocks
    block_mask:    [n_c, n_blocks] bool — own-state blocks (valid range)
    """
    eps = 1e-7
    p = route_probs.clamp(eps, 1 - eps)
    bce = -(block_target * p.log() + (1 - block_target) * (1 - p).log())
    bce = bce * block_mask
    denom = block_mask.sum().clamp(min=1)
    return bce.sum() / denom


def router_kl(
    teacher_route_logp: torch.Tensor,
    student_route_logp: torch.Tensor,
    block_mask: torch.Tensor,
) -> torch.Tensor:
    """Router distillation KL(route_T || route_S) over own-state blocks."""
    t = teacher_route_logp.exp()
    kl = (t * (teacher_route_logp - student_route_logp)) * block_mask
    return kl.sum() / block_mask.sum().clamp(min=1)


def state_relation_loss(hs_a: torch.Tensor, hs_b: torch.Tensor) -> torch.Tensor:
    """Match normalized token-similarity Gram matrices (sec 28.4).

    hs_a, hs_b: [S, D] aligned state memories of teacher/student —
    widths may differ, token geometry must not.
    """
    ga = hs_a / hs_a.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gb = hs_b / hs_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return ((ga @ ga.T) - (gb @ gb.T)).pow(2).mean()


def candidate_relation_loss(r_a: torch.Tensor, r_b: torch.Tensor) -> torch.Tensor:
    """Match candidate-set geometry between teacher and student."""
    ga = r_a / r_a.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gb = r_b / r_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return ((ga @ ga.T) - (gb @ gb.T)).pow(2).mean()


class DecisionLossWeights:
    """Loss weights from docs.md section 40.2 (Large) — defaults."""

    def __init__(
        self,
        nll: float = 1.0,
        brier: float = 0.5,
        rps: float = 0.35,
        consistency: float = 0.15,
        router: float = 0.25,
        missing: float = 0.25,
    ):
        self.nll = nll
        self.brier = brier
        self.rps = rps
        self.consistency = consistency
        self.router = router
        self.missing = missing


def decision_loss(
    out: DecisionOutput,
    targets: torch.Tensor,
    *,
    cand_ordinals: torch.Tensor | None = None,
    score_question_mask: torch.Tensor | None = None,
    missing_mask: torch.Tensor | None = None,
    block_target: torch.Tensor | None = None,
    block_mask: torch.Tensor | None = None,
    weights: DecisionLossWeights | None = None,
    missing_tau: float = 0.8,
) -> dict[str, torch.Tensor]:
    """Composite Large-stage decision loss (sec 40.2)."""
    w = weights or DecisionLossWeights()
    total = w.nll * nll_loss(out, targets) + w.brier * brier_loss(out, targets)
    parts = {"nll": nll_loss(out, targets).detach(), "brier": brier_loss(out, targets).detach()}
    if cand_ordinals is not None and score_question_mask is not None:
        rps = rps_loss(out, targets, cand_ordinals, score_question_mask)
        total = total + w.rps * rps
        parts["rps"] = rps.detach()
    if missing_mask is not None:
        miss = missing_evidence_loss(out, missing_mask, missing_tau)
        total = total + w.missing * miss
        parts["missing"] = miss.detach()
    if block_target is not None and block_mask is not None:
        rtr = router_bce(out.route_probs, block_target, block_mask)
        total = total + w.router * rtr
        parts["router"] = rtr.detach()
    parts["total"] = total
    return parts
