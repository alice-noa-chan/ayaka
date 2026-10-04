"""Gold-anchored supervised ablation for the experimental evidence head.

Reference KL is additive, never a replacement for gold CE. The reference must
be frozen and bound to the same examples, recipe and candidate order by the
caller. This tensor objective neither establishes that binding nor implements
Clef's unpublished RL training algorithm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch

from ayaka.model.evidence import EvidenceOutput


@dataclass(frozen=True)
class EvidenceLossWeights:
    nll: float = 1.0
    brier: float = 0.5
    rps: float = 0.35
    reference: float = 0.0  # explicit ablation; enabling it requires reference logits

    def validate(self) -> None:
        values = (self.nll, self.brier, self.rps, self.reference)
        if any(
            not isinstance(v, Real) or isinstance(v, bool) or not math.isfinite(v) or v < 0
            for v in values
        ):
            raise ValueError("loss weights must be finite nonnegative numbers")
        if self.nll <= 0:
            raise ValueError("gold nll must remain enabled")


def evidence_loss(
    out: EvidenceOutput,
    targets: torch.Tensor,
    primitive: torch.Tensor,
    *,
    ordinals: torch.Tensor | None = None,
    reference_logits: torch.Tensor | None = None,
    weights: EvidenceLossWeights | None = None,
    label_smoothing: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Mean gold CE+Brier, Score RPS and optional KL(reference || student).

    Soft gold is retained. Smoothing is opt-in and affects only the CE target;
    Brier/RPS always use original gold. RPS sorts explicit ordinal values and
    uses rank-normalized CDF error, matching the existing benchmark loss (not
    a value-distance metric for nonuniform levels). Padding has zero mass.
    Noul is exactly two candidates; Score needs at least two distinct levels.
    Diagnostics are detached, while ``total`` retains student gradients.
    """
    w = weights or EvidenceLossWeights()
    w.validate()
    if (
        not isinstance(label_smoothing, Real)
        or isinstance(label_smoothing, bool)
        or not math.isfinite(label_smoothing)
        or not 0 <= label_smoothing < 1
    ):
        raise ValueError("label_smoothing must be in [0, 1)")
    logits, mask, active = out.logits, out.candidate_mask, out.question_mask
    if logits.ndim != 3 or not logits.is_floating_point() or min(logits.shape) <= 0:
        raise ValueError("logits must be nonempty floating [record, question, candidate]")
    expected = [
        (mask, logits.shape),
        (active, logits.shape[:2]),
        (targets, logits.shape),
        (primitive, logits.shape[:2]),
    ]
    if any(t.shape != shape or t.device != logits.device for t, shape in expected):
        raise ValueError("loss shape/device mismatch")
    if (
        mask.dtype != torch.bool
        or active.dtype != torch.bool
        or not torch.equal(mask.any(-1), active)
        or not active.any()
    ):
        raise ValueError("loss requires consistent boolean masks and active questions")
    if not torch.isfinite(logits[mask]).all():
        raise ValueError("valid logits must be finite")
    if (
        primitive.dtype not in (torch.int32, torch.int64)
        or ((primitive[active] < 0) | (primitive[active] > 2)).any()
    ):
        raise ValueError("primitive must contain integer types 0, 1, 2")
    if (
        not targets.is_floating_point()
        or not torch.isfinite(targets).all()
        or (targets < 0).any()
        or (targets[~mask] != 0).any()
    ):
        raise ValueError("targets must be finite nonnegative distributions with zero padding")
    if not torch.allclose(
        targets.sum(-1)[active], torch.ones_like(targets.sum(-1)[active]), atol=1e-6, rtol=1e-6
    ):
        raise ValueError("active targets must sum to one")
    counts = mask.sum(-1)
    if ((primitive == 0) & active & (counts != 2)).any():
        raise ValueError("Noul requires exactly two candidates")
    score_questions = (primitive == 2) & active
    if (score_questions & (counts < 2)).any():
        raise ValueError("Score requires at least two candidates")
    if score_questions.any() and ordinals is None:
        raise ValueError("Score requires explicit ordinal levels")
    if ordinals is not None and (
        ordinals.shape != logits.shape
        or ordinals.device != logits.device
        or not ordinals.is_floating_point()
    ):
        raise ValueError("ordinal shape/device/dtype mismatch")
    if reference_logits is None and w.reference > 0:
        raise ValueError("reference weight requires bound reference logits")
    if reference_logits is not None and (
        reference_logits.shape != logits.shape
        or reference_logits.device != logits.device
        or not reference_logits.is_floating_point()
        or not torch.isfinite(reference_logits[mask]).all()
    ):
        raise ValueError("reference shape/device/dtype/finite mismatch")

    normalized = EvidenceOutput(logits.masked_fill(~mask, -torch.inf), out.correction, mask, active)
    logp = normalized.log_probs().masked_fill(~mask, 0)
    logp = logp.to(torch.promote_types(logp.dtype, targets.dtype))
    if not torch.isfinite(logp[mask]).all():
        raise ValueError("student log probabilities overflow")
    p = logp.exp().masked_fill(~mask, 0)
    gold = targets.detach().to(logp.dtype)
    ce_gold = (1 - label_smoothing) * gold + label_smoothing * mask / counts.clamp_min(1)[..., None]
    nll = -(ce_gold * logp).sum(-1)[active].mean()
    brier = ((p - gold) ** 2).sum(-1)[active].mean()
    total = w.nll * nll + w.brier * brier
    parts = {"nll": nll.detach(), "brier": brier.detach()}

    if score_questions.any():
        levels = ordinals[score_questions]
        score_mask = mask[score_questions]
        if not torch.isfinite(levels[score_mask]).all():
            raise ValueError("valid Score ordinals must be finite")
        order = levels.masked_fill(~score_mask, torch.inf).argsort(-1)
        sorted_levels = levels.gather(-1, order)
        thresholds = (
            torch.arange(logits.shape[-1], device=logits.device)[None]
            < (counts[score_questions] - 1)[:, None]
        )
        if ((sorted_levels[:, 1:] == sorted_levels[:, :-1]) & thresholds[:, :-1]).any():
            raise ValueError("Score ordinal levels must be distinct")
        cp = p[score_questions].gather(-1, order).cumsum(-1)
        cy = gold[score_questions].gather(-1, order).cumsum(-1)
        rps = (
            ((cp - cy) ** 2).masked_fill(~thresholds, 0).sum(-1) / (counts[score_questions] - 1)
        ).mean()
        parts["rps"] = rps.detach()
        total = total + w.rps * rps

    if reference_logits is not None:
        # Preserve fp64 tiny-tail references when supplied, even for fp32 students.
        dtype = torch.promote_types(logp.dtype, reference_logits.dtype)
        reference = EvidenceOutput(
            reference_logits.detach().to(dtype).masked_fill(~mask, -torch.inf),
            out.correction,
            mask,
            active,
        )
        ref_logp = reference.log_probs().masked_fill(~mask, 0)
        if not torch.isfinite(ref_logp[mask]).all():
            raise ValueError("reference log probabilities overflow")
        ref_p = ref_logp.exp().masked_fill(~mask, 0)
        kl = (ref_p * (ref_logp - logp)).sum(-1)[active].mean()
        parts["reference_kl"] = kl.detach()
        total = total + w.reference * kl
    parts["total"] = total
    return parts
