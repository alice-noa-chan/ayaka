"""Candidate Set Mixer + Pointer Head (docs.md sections 13-14).

Each candidate is represented by the mean backbone state over its
option span, shifted by its question's answer-position state. The Set
Mixer is self-attention over one question's candidates with no
positional or index features, so it is permutation-equivariant; the
pointer scores each mixed candidate against the question query.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ragged import from_padded, to_padded


class SetMixerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.ln2 = nn.RMSNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, k, d = x.shape
        q, kk, v = self.qkv(self.ln1(x)).view(b, k, 3, self.heads, d // self.heads).unbind(2)
        attn_mask = mask[:, None, None, :]  # keys: valid candidates only
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), kk.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask
        )
        x = x + self.drop(self.proj(out.transpose(1, 2).reshape(b, k, d)))
        return x + self.drop(self.ffn(self.ln2(x)))


class PointerHead(nn.Module):
    """r_i (span states) + q (answer state) -> per-candidate logits."""

    def __init__(self, hidden: int, dim: int, layers: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.in_r = nn.Linear(hidden, dim, bias=False)
        self.in_q = nn.Linear(hidden, dim, bias=False)
        self.blocks = nn.ModuleList(SetMixerBlock(dim, heads, dropout) for _ in range(layers))
        self.ln_out = nn.RMSNorm(dim)
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        cand_repr: torch.Tensor,  # [n_c, H] mean span states
        query: torch.Tensor,  # [n_q, H] answer-position states
        cand_cu: torch.Tensor,
        cand_question: torch.Tensor,  # [n_c] question index
    ) -> torch.Tensor:
        q = self.in_q(query)
        x = self.in_r(cand_repr) + q[cand_question]
        xp, mask = to_padded(x, cand_cu)
        for blk in self.blocks:
            xp = blk(xp, mask)
        r = self.ln_out(from_padded(xp, cand_cu))
        return (self.w_q(q)[cand_question] * self.w_k(r)).sum(-1) / math.sqrt(self.dim)
