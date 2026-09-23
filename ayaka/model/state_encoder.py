"""Shared State Encoder (docs.md sections 9, 19, 39.1).

Encodes each state exactly once into Full Token Memory ``Hs [T, D]``
plus Block Memory ``Bs [n_blocks, D]``:

- states shorter than ``short_context_threshold`` take the full
  bidirectional attention path (no long-context penalty)
- longer states take the hierarchical path: block-local attention
  (1024-token blocks) -> per-block summary latents -> global summary
  attention -> token <- block-summary integration

Both paths share the same parameters; the short path is recovered
automatically because chunking leaves short segments whole.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import ElectraConfig
from .blocks import CrossAttention, SwiGLU, TransformerBlock
from .norm import RMSNorm
from .ragged import block_cu_seqlens, chunk_cu_seqlens, state_block_ranges
from .rope import RotaryEmbedding


@dataclass
class StateMemory:
    """Output of the shared encoder consumed by decision branches."""

    hs: torch.Tensor  # [T, D] full token memory, flat ragged
    bs: torch.Tensor  # [n_blocks, D] block memory
    state_cu: torch.Tensor  # [n_states + 1] token boundaries per state
    blk_cu: torch.Tensor  # [n_blocks + 1] token boundaries per block
    state_blk_cu: torch.Tensor  # [n_states + 1] block ranges per state
    block_state_index: torch.Tensor  # [n_blocks] owning state
    route_tokens: bool  # True when any state took the hierarchical path

    @property
    def state_lens(self) -> torch.Tensor:
        return self.state_cu[1:] - self.state_cu[:-1]


class BlockSummaryPool(nn.Module):
    """One learned latent per block pools its tokens into Bs."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(dim) * 0.02)
        self.ln_q = RMSNorm(dim)
        self.ln_kv = RMSNorm(dim)
        self.attn = CrossAttention(dim, heads)

    def forward(self, hs: torch.Tensor, blk_cu: torch.Tensor, n_blocks: int) -> torch.Tensor:
        q = self.ln_q(self.latent).expand(n_blocks, -1)
        cu_q = torch.arange(0, n_blocks + 1, dtype=blk_cu.dtype, device=hs.device)
        return self.attn(q, self.ln_kv(hs), cu_q, blk_cu)


class EvidenceIntegration(nn.Module):
    """Tokens absorb global context from their state's block summaries."""

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
        hs: torch.Tensor,
        bs: torch.Tensor,
        state_cu: torch.Tensor,
        state_blk_cu: torch.Tensor,
    ) -> torch.Tensor:
        hs = hs + self.drop(self.attn(self.ln_q(hs), self.ln_kv(bs), state_cu, state_blk_cu))
        return hs + self.drop(self.ffn(self.ln_f(hs)))


class StateEncoder(nn.Module):
    def __init__(self, cfg: ElectraConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.max_state)

        n_local = max(1, cfg.state_layers - 2)
        n_global = 1 if cfg.state_layers >= 3 else 0
        n_integrate = cfg.state_layers - n_local - n_global

        self.local_layers = nn.ModuleList(
            TransformerBlock(d, cfg.heads, cfg.ffn, cfg.resid_dropout) for _ in range(n_local)
        )
        self.global_layers = nn.ModuleList(
            TransformerBlock(d, cfg.heads, cfg.ffn, cfg.resid_dropout) for _ in range(n_global)
        )
        self.integrate_layers = nn.ModuleList(
            EvidenceIntegration(d, cfg.heads, cfg.ffn, cfg.resid_dropout)
            for _ in range(n_integrate)
        )
        self.block_pool = BlockSummaryPool(d, cfg.heads)
        self.ln_out = RMSNorm(d)
        self.ln_bs = RMSNorm(d)

    def forward(self, token_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> StateMemory:
        """token_ids: [T] flat ragged; cu_seqlens: [n_states+1]."""
        x = self.embed(token_ids)
        blk_cu = block_cu_seqlens(cu_seqlens, self.cfg.block_size)
        state_blk_cu, block_state_index = state_block_ranges(cu_seqlens, blk_cu)
        any_long = bool(
            ((cu_seqlens[1:] - cu_seqlens[:-1]) > self.cfg.short_context_threshold).any()
        )
        attn_cu = (
            chunk_cu_seqlens(cu_seqlens, self.cfg.block_size, self.cfg.short_context_threshold)
            if any_long
            else cu_seqlens
        )

        for layer in self.local_layers:
            x = layer(x, attn_cu, self.rope)

        n_blocks = blk_cu.numel() - 1
        bs = self.block_pool(x, blk_cu, n_blocks)
        if self.global_layers:
            # blocks only attend within their owning state
            for layer in self.global_layers:
                bs = layer(bs, state_blk_cu, self.rope)
        for layer in self.integrate_layers:
            x = layer(x, bs, cu_seqlens, state_blk_cu)

        return StateMemory(
            hs=self.ln_out(x),
            bs=self.ln_bs(bs),
            state_cu=cu_seqlens,
            blk_cu=blk_cu,
            state_blk_cu=state_blk_cu,
            block_state_index=block_state_index,
            route_tokens=any_long,
        )
