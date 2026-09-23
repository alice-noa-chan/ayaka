"""Variable-length (padding-free) bidirectional attention.

Training path uses FlashAttention-2's varlen kernel via
``cu_seqlens`` — no dense [B, S, S] mask is ever materialized
(docs.md section 45.2). The fallback path packs segments into a
padded buffer with an O(B*S) key-padding mask, which also never
builds a dense pairwise mask and works on CPU for tests.

All attention here is non-causal: the evidence encoder and decision
branches are bidirectional by design (sections 9.2, 23).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # training path (GPU image on beam.cloud installs flash-attn)
    from flash_attn import flash_attn_varlen_func

    _HAS_FLASH = True
except ImportError:  # pragma: no cover - depends on env
    flash_attn_varlen_func = None
    _HAS_FLASH = False


def _seg_maxlen(cu_seqlens: torch.Tensor) -> int:
    return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())


def _flat_index(cu_seqlens: torch.Tensor, seg_len: int) -> torch.Tensor:
    """Flat-buffer position of each ragged token inside a [B*seg_len] buffer."""
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    total = int(cu_seqlens[-1].item())
    seg_ids = torch.repeat_interleave(torch.arange(lens.numel(), device=cu_seqlens.device), lens)
    within = torch.arange(total, device=cu_seqlens.device) - torch.repeat_interleave(
        cu_seqlens[:-1], lens
    )
    return seg_ids * seg_len + within


def _pack(x: torch.Tensor, cu_seqlens: torch.Tensor, seg_len: int) -> torch.Tensor:
    """[T, H, Dh] ragged -> [B, seg_len, H, Dh] zero-padded."""
    b = cu_seqlens.numel() - 1
    t, h, dh = x.shape
    buf = x.new_zeros((b * seg_len, h, dh))
    buf[_flat_index(cu_seqlens, seg_len)] = x
    return buf.view(b, seg_len, h, dh)


def _key_padding_mask(cu_seqlens: torch.Tensor, seg_len: int) -> torch.Tensor:
    """[B, 1, 1, seg_len] bool mask, True on real tokens."""
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    positions = torch.arange(seg_len, device=cu_seqlens.device)
    keep = positions[None, :] < lens[:, None]
    return keep[:, None, None, :]


def _sdpa_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    sq, sk = _seg_maxlen(cu_q), _seg_maxlen(cu_k)
    b = cu_q.numel() - 1
    qp = _pack(q, cu_q, sq).transpose(1, 2)  # [B, H, S, Dh]
    kp = _pack(k, cu_k, sk).transpose(1, 2)
    vp = _pack(v, cu_k, sk).transpose(1, 2)
    mask = _key_padding_mask(cu_k, sk)
    out = F.scaled_dot_product_attention(
        qp, kp, vp, attn_mask=mask, dropout_p=dropout_p, is_causal=False
    )
    flat = out.transpose(1, 2).reshape(b * sq, q.shape[1], q.shape[2])
    return flat[_flat_index(cu_q, sq)]


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Bidirectional attention over ragged segments.

    q: [Tq, H, Dh]; k, v: [Tk, H, Dh]. Returns [Tq, H, Dh].
    Segment i of q attends only to segment i of k/v.
    """
    if _HAS_FLASH and q.is_cuda:
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            _seg_maxlen(cu_seqlens_q),
            _seg_maxlen(cu_seqlens_k),
            dropout_p=dropout_p,
            causal=False,
        )
    return _sdpa_fallback(q, k, v, cu_seqlens_q, cu_seqlens_k, dropout_p)
