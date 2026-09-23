"""Permutation-equivariant Candidate Set Mixer (docs.md section 13).

Self-attention over the pooled candidate vectors of one question's
candidate set. There are NO positional embeddings and no candidate
index features anywhere — candidates form an unordered semantic set,
so F(state, q, pi(C)) = pi(F(state, q, C)) holds structurally.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ElectraConfig
from .attention import varlen_attention
from .blocks import SwiGLU
from .norm import RMSNorm


class SetSelfAttention(nn.Module):
    """Self-attention without any positional encoding."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        t, d = x.shape
        qkv = self.qkv(x).view(t, 3, self.heads, self.head_dim)
        out = varlen_attention(
            qkv[:, 0], qkv[:, 1], qkv[:, 2], cu_seqlens, cu_seqlens, self.dropout
        )
        return self.proj(out.reshape(t, d))


class SetMixerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: int, resid_dropout: float):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = SetSelfAttention(dim, heads)
        self.ln2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn)
        self.drop = nn.Dropout(resid_dropout)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), cu_seqlens))
        return x + self.drop(self.ffn(self.ln2(x)))


class SetMixer(nn.Module):
    """Listwise interaction within each question's candidate set."""

    def __init__(self, cfg: ElectraConfig):
        super().__init__()
        d = cfg.hidden
        self.layers = nn.ModuleList(
            SetMixerBlock(d, cfg.heads, cfg.ffn, cfg.resid_dropout)
            for _ in range(cfg.set_mixer_layers)
        )
        self.ln_out = RMSNorm(d)

    def forward(self, r: torch.Tensor, cand_cu: torch.Tensor) -> torch.Tensor:
        """r: [total_cand, D]; cand_cu: per-question set boundaries."""
        for layer in self.layers:
            r = layer(r, cand_cu)
        return self.ln_out(r)
