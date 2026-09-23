"""Pointer Distribution Head (docs.md section 14).

No fixed class head: the question's pooled decision representation
points at candidate representations, so the same head serves K=2 or
K=255 and never-before-seen candidate ontologies.

    z_j = (Wq d)^T (Wk R_j) / sqrt(Dp);  p = softmax(z)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ElectraConfig


def _seg_ids(cu: torch.Tensor) -> torch.Tensor:
    lens = (cu[1:] - cu[:-1]).long()
    return torch.repeat_interleave(torch.arange(lens.numel(), device=cu.device), lens)


def ragged_log_softmax(logits: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    """log_softmax within each ragged segment (per question's set)."""
    seg = _seg_ids(cu)
    n = cu.numel() - 1
    m = torch.full((n,), float("-inf"), dtype=logits.dtype, device=logits.device)
    m.index_reduce_(0, seg, logits, "amax")
    z = logits - m[seg]
    e = z.exp()
    denom = torch.zeros(n, dtype=logits.dtype, device=logits.device)
    denom.index_add_(0, seg, e)
    return z - denom[seg].log()


def ragged_softmax(logits: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    return ragged_log_softmax(logits, cu).exp()


def ragged_max(logits: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    """Max logit per segment (for missing-evidence penalty)."""
    seg = _seg_ids(cu)
    out = torch.full((cu.numel() - 1,), float("-inf"), dtype=logits.dtype, device=logits.device)
    out.index_reduce_(0, seg, logits, "amax")
    return out


class PointerHead(nn.Module):
    def __init__(self, cfg: ElectraConfig):
        super().__init__()
        d, dp = cfg.hidden, cfg.pointer_dim
        self.w_q = nn.Linear(d, dp, bias=False)
        self.w_k = nn.Linear(d, dp, bias=False)
        self.scale = dp**-0.5

    def forward(
        self,
        question_repr: torch.Tensor,  # [n_q, D]
        candidate_repr: torch.Tensor,  # [n_c, D]
        cand_question_index: torch.Tensor,  # [n_c]
    ) -> torch.Tensor:
        """Per-candidate logits [n_c]; softmax happens per question set."""
        q = self.w_q(question_repr)  # [n_q, Dp]
        k = self.w_k(candidate_repr)  # [n_c, Dp]
        return (q[cand_question_index] * k).sum(-1) * self.scale
