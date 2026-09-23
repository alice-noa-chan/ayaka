"""Shared Question/Candidate Text Encoder (docs.md sections 11, 12, 38.1).

One encoder (shared weights, shared token embedding) compresses each
instruction and each candidate description into a small latent set:
question -> Rq [Mq, D], candidate -> Rc [Mc, D]. Every candidate goes
through the same encoder independently, and candidate count K stays a
runtime-dynamic axis.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ElectraConfig
from .blocks import CrossAttention, SwiGLU, TransformerBlock
from .norm import RMSNorm
from .rope import RotaryEmbedding


class LatentReadout(nn.Module):
    """M learned latents cross-attend to an item's encoded tokens."""

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
        latents: torch.Tensor,
        tokens: torch.Tensor,
        item_cu: torch.Tensor,
        tok_cu: torch.Tensor,
    ) -> torch.Tensor:
        """latents: [n_items*M, D]; tokens: [T, D] ragged."""
        h = latents + self.drop(self.attn(self.ln_q(latents), self.ln_kv(tokens), item_cu, tok_cu))
        return h + self.drop(self.ffn(self.ln_f(h)))


class TextEncoder(nn.Module):
    """Encodes question instructions or candidate text into latent sets."""

    def __init__(self, cfg: ElectraConfig, embed: nn.Embedding):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden
        self.embed = embed  # shared table owned by the state encoder
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        self.layers = nn.ModuleList(
            TransformerBlock(d, cfg.heads, cfg.ffn, cfg.resid_dropout)
            for _ in range(cfg.text_encoder_layers)
        )
        self.q_latents = nn.Parameter(torch.randn(cfg.question_latents, d) * 0.02)
        self.c_latents = nn.Parameter(torch.randn(cfg.candidate_latents, d) * 0.02)
        self.q_readout = LatentReadout(d, cfg.heads, cfg.ffn, cfg.resid_dropout)
        self.c_readout = LatentReadout(d, cfg.heads, cfg.ffn, cfg.resid_dropout)
        self.ln_out = RMSNorm(d)

    def _encode_tokens(self, token_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)
        for layer in self.layers:
            x = layer(x, cu_seqlens, self.rope)
        return x

    def forward(
        self,
        token_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        kind: str,
    ) -> torch.Tensor:
        """Return latents [n_items, M, D] for kind in {"q", "c"}."""
        tokens = self._encode_tokens(token_ids, cu_seqlens)
        n_items = cu_seqlens.numel() - 1
        if kind == "q":
            m, readout = self.cfg.question_latents, self.q_readout
            base = self.q_latents
        else:
            m, readout = self.cfg.candidate_latents, self.c_readout
            base = self.c_latents
        latents = base.unsqueeze(0).expand(n_items, m, d := base.shape[-1])
        latents = latents.reshape(n_items * m, d)
        item_cu = torch.arange(0, n_items * m + 1, m, dtype=cu_seqlens.dtype, device=tokens.device)
        out = readout(latents, tokens, item_cu, cu_seqlens)
        return self.ln_out(out).view(n_items, m, d)
