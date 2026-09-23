"""Ragged/padding-free metadata helpers.

These functions operate on cu_seqlens-style boundary tensors only —
they are batch-descriptor (collation) work, not per-token compute, so
they run eagerly on the metadata path (docs.md section 46).
"""

from __future__ import annotations

import torch


def chunk_cu_seqlens(
    cu_seqlens: torch.Tensor, chunk_size: int, short_threshold: int
) -> torch.Tensor:
    """Split each segment into attention chunks.

    Segments of length <= short_threshold stay whole (full exact
    attention); longer segments are split into chunks of at most
    chunk_size (block-local attention). Returns new cu_seqlens.
    """
    bounds = [0]
    offset = 0
    for i in range(cu_seqlens.numel() - 1):
        s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        seg_len = e - s
        if seg_len <= short_threshold:
            bounds.append(offset + seg_len)
        else:
            pos = 0
            while pos < seg_len:
                step = min(chunk_size, seg_len - pos)
                bounds.append(offset + pos + step)
                pos += step
        offset += seg_len
    return torch.tensor(bounds, dtype=cu_seqlens.dtype, device=cu_seqlens.device)


def block_cu_seqlens(cu_seqlens: torch.Tensor, block_size: int) -> torch.Tensor:
    """Per-segment fixed-size block boundaries (for Block Memory Bs).

    Unlike chunk_cu_seqlens this always uses block_size — block
    summaries exist for the router regardless of the attention path.
    """
    bounds = [0]
    for i in range(cu_seqlens.numel() - 1):
        s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        pos = s
        while pos < e:
            bounds.append(min(pos + block_size, e))
            pos += block_size
    return torch.tensor(bounds, dtype=cu_seqlens.dtype, device=cu_seqlens.device)


def state_block_ranges(
    cu_seqlens: torch.Tensor, blk_cu: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map blocks back to their owning state.

    Returns (state_blk_cu [n_states+1], block_state_index [n_blocks]).
    """
    n_states = cu_seqlens.numel() - 1
    state_blk_cu = torch.zeros(n_states + 1, dtype=blk_cu.dtype, device=blk_cu.device)
    block_state_index = torch.zeros(blk_cu.numel() - 1, dtype=torch.long, device=blk_cu.device)
    cursor = 0
    for i in range(n_states):
        s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        n = 0
        while cursor < blk_cu.numel() - 1 and int(blk_cu[cursor]) < e:
            if int(blk_cu[cursor]) >= s:
                block_state_index[cursor] = i
                n += 1
                cursor += 1
            else:
                break
        state_blk_cu[i + 1] = cursor
    return state_blk_cu, block_state_index


def segment_mean(x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Mean-pool each ragged segment: [T, D] -> [n_seg, D]."""
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(x.dtype)
    summed = torch.zeros(cu_seqlens.numel() - 1, x.shape[-1], dtype=x.dtype, device=x.device)
    seg_ids = torch.repeat_interleave(torch.arange(lens.numel(), device=x.device), lens.long())
    summed.index_add_(0, seg_ids, x)
    return summed / lens.clamp(min=1).unsqueeze(-1)


def gather_segments(
    x: torch.Tensor, cu_seqlens: torch.Tensor, segment_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather whole segments by id (vectorized).

    Returns (tokens [T_sel, D], cu [len(segment_ids)+1]) in the order
    the ids were given.
    """
    src_start = cu_seqlens[:-1][segment_ids]
    sel_lens = cu_seqlens[1:][segment_ids] - src_start
    cu = torch.zeros(segment_ids.numel() + 1, dtype=cu_seqlens.dtype, device=x.device)
    cu[1:] = torch.cumsum(sel_lens, 0)
    cum = torch.cumsum(sel_lens, 0)
    dst_off = cum - sel_lens
    total = int(cum[-1].item())
    tok = torch.arange(total, device=x.device)
    seg_of_tok = torch.searchsorted(cum, tok, right=True)
    src_idx = src_start[seg_of_tok] + tok - dst_off[seg_of_tok]
    return x[src_idx], cu


def gather_block_tokens(
    hs: torch.Tensor,
    blk_cu: torch.Tensor,
    flat_sel: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather token memory for selected blocks (vectorized).

    flat_sel: block ids grouped by item (e.g. from flatten_selection);
    counts:   [n_items] how many ids each item contributes.
    Returns (tokens [T_sel, D], cu [n_items + 1]) — ragged.
    """
    src_start = blk_cu[:-1][flat_sel]
    sel_lens = blk_cu[1:][flat_sel] - src_start
    n_items = counts.numel()
    cu = torch.zeros(n_items + 1, dtype=blk_cu.dtype, device=hs.device)
    cu[1:] = torch.cumsum(
        torch.zeros(n_items, dtype=sel_lens.dtype, device=hs.device).index_add_(
            0,
            torch.repeat_interleave(torch.arange(n_items, device=hs.device), counts),
            sel_lens,
        ),
        0,
    )

    cum = torch.cumsum(sel_lens, 0)
    dst_off = cum - sel_lens
    total = int(cum[-1].item())
    tok = torch.arange(total, device=hs.device)
    blk_of_tok = torch.searchsorted(cum, tok, right=True)
    src_idx = src_start[blk_of_tok] + tok - dst_off[blk_of_tok]
    return hs[src_idx], cu
