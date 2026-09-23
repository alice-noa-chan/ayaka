"""Rotary position embeddings (docs.md section 9.2 / 39: RoPE).

Positions are computed per ragged segment — each sequence restarts at
position 0 — so they carry no cross-segment or cross-candidate meaning.
"""

from __future__ import annotations

import torch


def rope_positions(cu_seqlens: torch.Tensor, device=None) -> torch.Tensor:
    """Per-token position indices within each segment, for flat [T] input."""
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    total = int(cu_seqlens[-1].item())
    pos = torch.arange(total, device=device or cu_seqlens.device)
    starts = torch.repeat_interleave(cu_seqlens[:-1], lens)
    return pos - starts


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim: int, theta: float = 10_000.0, max_seq: int = 65_536):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = head_dim
        self.theta = theta
        self.max_seq = max_seq
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.float()[..., None] * self.inv_freq.to(positions.device)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [..., head_dim]; cos/sin broadcast over all but last dim
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        rot = torch.cat([-x2, x1], dim=-1)
        return x * cos + rot * sin
