"""Candidate <-> state evidence cross-attention (docs.md section 12/22).

Each query segment's latents attend to its matching memory segment —
the full Hs for short states or the router-selected block tokens for
long ones. Query segments are defined by the caller so the same code
serves per-state grouping (shared-memory path, sec 21.2) and
per-candidate grouping (routed path, sec 19).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ElectraConfig
from .blocks import CrossAttention, SwiGLU
from .norm import RMSNorm


class EvidenceBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: int, resid_dropout: float):
        super().__init__()
        self.ln_q = RMSNorm(dim)
        self.ln_kv = RMSNorm(dim)
        self.attn = CrossAttention(dim, heads)
        self.ln_f = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn)
        self.drop = nn.Dropout(resid_dropout)

    def forward(
        self,
        latents: torch.Tensor,  # [T_lat, D]
        memory: torch.Tensor,  # [Tm, D]
        item_cu: torch.Tensor,  # query segment boundaries
        mem_cu: torch.Tensor,  # memory segment boundaries
    ) -> torch.Tensor:
        latents = latents + self.drop(
            self.attn(self.ln_q(latents), self.ln_kv(memory), item_cu, mem_cu)
        )
        return latents + self.drop(self.ffn(self.ln_f(latents)))


class EvidenceCrossAttention(nn.Module):
    """Stack of evidence blocks producing evidence-aware latents Ec."""

    def __init__(self, cfg: ElectraConfig):
        super().__init__()
        d = cfg.hidden
        self.blocks = nn.ModuleList(
            EvidenceBlock(d, cfg.heads, cfg.ffn, cfg.resid_dropout)
            for _ in range(cfg.cross_attn_blocks)
        )
        self.ln_out = RMSNorm(d)

    def forward(
        self,
        flat_latents: torch.Tensor,  # [T_lat, D]
        item_cu: torch.Tensor,  # [n_items + 1]
        memory: torch.Tensor,  # [Tm, D]
        mem_cu: torch.Tensor,  # [n_items + 1]
    ) -> torch.Tensor:
        for block in self.blocks:
            flat_latents = block(flat_latents, memory, item_cu, mem_cu)
        return self.ln_out(flat_latents)
