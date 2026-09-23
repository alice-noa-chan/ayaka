"""Transformer blocks for Electra (docs.md section 39 common block).

Pre-Norm RMSNorm + bidirectional varlen attention (RoPE) + SwiGLU FFN.
Inputs are flat ragged token buffers [T, D] with cu_seqlens metadata —
no padding, no dense attention masks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import varlen_attention
from .norm import RMSNorm
from .rope import RotaryEmbedding, rope_positions


class SwiGLU(nn.Module):
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.w3 = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class SelfAttention(nn.Module):
    """Multi-head self-attention over flat ragged tokens."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope: RotaryEmbedding,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t, d = x.shape
        qkv = self.qkv(x).view(t, 3, self.heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        if positions is None:
            positions = rope_positions(cu_seqlens, device=x.device)
        cos, sin = rope.cos_sin(positions, x.dtype)
        q = rope.apply(q, cos.unsqueeze(-2), sin.unsqueeze(-2))
        k = rope.apply(k, cos.unsqueeze(-2), sin.unsqueeze(-2))
        out = varlen_attention(q, k, v, cu_seqlens, cu_seqlens, self.dropout)
        return self.proj(out.reshape(t, d))


class CrossAttention(nn.Module):
    """Cross-attention: ragged queries attend to ragged memory segments."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> torch.Tensor:
        t, d = x.shape
        q = self.q_proj(x).view(t, self.heads, self.head_dim)
        kv = self.kv_proj(memory).view(-1, 2, self.heads, self.head_dim)
        k, v = kv[:, 0], kv[:, 1]
        out = varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, self.dropout)
        return self.proj(out.reshape(t, d))


class TransformerBlock(nn.Module):
    """Pre-norm encoder block: self-attn + SwiGLU."""

    def __init__(self, dim: int, heads: int, ffn: int, resid_dropout: float = 0.05):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.ln2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn)
        self.drop = nn.Dropout(resid_dropout)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope: RotaryEmbedding,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), cu_seqlens, rope, positions))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x
