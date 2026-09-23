"""Latent-set pooling: one learned query summarizes M latents -> [D]."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import CrossAttention
from .norm import RMSNorm


class LatentPool(nn.Module):
    """Pools [n_items, M, D] -> [n_items, D] via a learned query."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.ln_q = RMSNorm(dim)
        self.ln_kv = RMSNorm(dim)
        self.attn = CrossAttention(dim, heads)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        n_items, m, d = latents.shape
        q = self.ln_q(self.query).expand(n_items, d)
        cu_q = torch.arange(0, n_items + 1, dtype=torch.long, device=latents.device)
        cu_k = torch.arange(0, n_items * m + 1, m, dtype=torch.long, device=latents.device)
        return self.attn(q, self.ln_kv(latents.reshape(n_items * m, d)), cu_q, cu_k)
