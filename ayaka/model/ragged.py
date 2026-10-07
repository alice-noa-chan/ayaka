"""Ragged candidate-set ops.

Candidates of all questions in a batch live in one flat tensor; the
boundary tensor ``cu`` ([n_q + 1], cumulative counts) delimits each
question's set. Softmax/max/sum are taken within a set only.
"""

from __future__ import annotations

import torch


def seg_ids(cu: torch.Tensor, *, output_size: int | None = None) -> torch.Tensor:
    """Expand question IDs; a known candidate count avoids a device synchronization."""
    lens = (cu[1:] - cu[:-1]).long()
    return torch.repeat_interleave(
        torch.arange(lens.numel(), device=cu.device), lens, output_size=output_size
    )


def ragged_log_softmax(logits: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    seg = seg_ids(cu, output_size=logits.shape[0])
    n = cu.numel() - 1
    m = logits.new_full((n,), float("-inf")).scatter_reduce(0, seg, logits.detach(), "amax")
    z = logits - m[seg]
    denom = torch.zeros(n, dtype=logits.dtype, device=logits.device).index_add(0, seg, z.exp())
    return z - denom[seg].log()


def ragged_softmax(logits: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    return ragged_log_softmax(logits, cu).exp()


def ragged_max(x: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    seg = seg_ids(cu, output_size=x.shape[0])
    return x.new_full((cu.numel() - 1,), float("-inf")).scatter_reduce(0, seg, x, "amax")


def per_question_sum(x: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(cu.numel() - 1, dtype=x.dtype, device=x.device)
    return out.index_add(0, seg_ids(cu, output_size=x.shape[0]), x)


def to_padded(x: torch.Tensor, cu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[n_c, D] flat -> ([n_q, K, D] padded, [n_q, K] bool valid mask)."""
    lens = (cu[1:] - cu[:-1]).long()
    n_q, k = lens.numel(), int(lens.max()) if lens.numel() else 0
    seg = seg_ids(cu, output_size=x.shape[0])
    slot = torch.arange(x.shape[0], device=x.device) - cu[:-1].long()[seg]
    out = x.new_zeros(n_q, k, *x.shape[1:])
    out[seg, slot] = x
    mask = torch.zeros(n_q, k, dtype=torch.bool, device=x.device)
    mask[seg, slot] = True
    return out, mask


def from_padded(x: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    seg = seg_ids(cu)
    slot = torch.arange(seg.numel(), device=x.device) - cu[:-1].long()[seg]
    return x[seg, slot]
