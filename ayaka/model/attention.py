"""Windowed SDPA: exact sliding-window attention without the full S x S.

Gemma 4 sliding layers attend to the last ``sliding_window`` keys, but
SDPA with an explicit mask still computes every query x key score and
masks most of them away. For long inputs (JevBench hard averages ~1.2K
tokens, window 512) this splits the queries into blocks and gives each
block only the key range its mask can allow.

Exactness guard: before using a narrowed key range, the block's mask is
checked to be all-False outside it; if not (an unexpected cache layout),
that block falls back to the full key range. The softmax over the kept
range is then identical to the softmax over all keys.
"""

from __future__ import annotations

import torch

NAME = "ayaka_sdpa"
QUERY_BLOCK = 256

_registered = False


def windowed_sdpa(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    **kwargs,
):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def full(q, k, v, m):
        return sdpa_attention_forward(
            module,
            q,
            k,
            v,
            m,
            dropout=dropout,
            scaling=scaling,
            sliding_window=sliding_window,
            **kwargs,
        )

    q_len, kv_len = query.shape[2], key.shape[2]
    if (
        not sliding_window
        or attention_mask is None
        or attention_mask.dtype != torch.bool
        or attention_mask.dim() != 4
        or attention_mask.shape[-2] != q_len
        or attention_mask.shape[-1] != kv_len
        or q_len <= QUERY_BLOCK
        or kv_len <= sliding_window + QUERY_BLOCK
    ):
        return full(query, key, value, attention_mask)

    offset = kv_len - q_len  # queries are the newest positions of the key sequence
    outs = []
    for qs in range(0, q_len, QUERY_BLOCK):
        qe = min(qs + QUERY_BLOCK, q_len)
        ks = max(0, offset + qs - sliding_window + 1)
        ke = offset + qe
        m = attention_mask[:, :, qs:qe]
        if m[..., :ks].any() or m[..., ke:].any():  # guard: never drop an allowed key
            ks, ke = 0, kv_len
        o, _ = full(query[:, :, qs:qe], key[:, :, ks:ke], value[:, :, ks:ke], m[..., ks:ke])
        outs.append(o)
    return torch.cat(outs, dim=1), None


def enable_windowed_attention(config) -> None:
    """Route a text config's attention through ``windowed_sdpa``."""
    global _registered
    if not _registered:
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

        AttentionInterface.register(NAME, windowed_sdpa)
        AttentionMaskInterface.register(NAME, sdpa_mask)
        _registered = True
    config._attn_implementation_internal = NAME
