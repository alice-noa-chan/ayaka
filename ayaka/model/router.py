"""Block Router for long-context evidence selection (sec 19/39.1/A9).

Each decision item (candidate latent set, pooled to one query) scores
the block summaries of its owning state; for long states only the
top-k blocks' tokens become cross-attention memory. Short states
select all of their blocks, which reduces to exact full attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ElectraConfig


class BlockRouter(nn.Module):
    def __init__(self, cfg: ElectraConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.scale = d**-0.5

    def forward(
        self,
        item_queries: torch.Tensor,  # [n_items, D] pooled query per item
        bs: torch.Tensor,  # [n_blocks, D] block memory
        item_state_index: torch.Tensor,  # [n_items] owning state
        block_state_index: torch.Tensor,  # [n_blocks] owning state
    ) -> dict[str, torch.Tensor]:
        """Return route distributions + selected block sets per item.

        Only items whose state exceeded the short-context threshold are
        routed; the caller handles short states with the full-memory
        path. route_probs is still emitted for every item so router
        supervision/distillation works batch-wide.

        route_probs: [n_items, n_blocks] dense (0 outside owning state)
        selected:    [n_items, k] global block indices
        valid:       [n_items, k] bool — False for pad picks when the
                     owning state has fewer blocks than k
        """
        n_blocks = bs.shape[0]
        q = self.q_proj(item_queries)  # [n_items, D]
        k = self.k_proj(bs)  # [n_blocks, D]
        scores = (q @ k.T) * self.scale  # [n_items, n_blocks]
        own = block_state_index.unsqueeze(0) == item_state_index.unsqueeze(1)
        scores = scores.masked_fill(~own, float("-inf"))
        route_probs = torch.softmax(scores, dim=-1)

        blocks_per_item = own.sum(dim=1)  # [n_items]
        k_sel = min(self.cfg.block_topk, n_blocks)
        selected = route_probs.topk(k_sel, dim=1).indices
        pos = torch.arange(k_sel, device=bs.device).unsqueeze(0)
        valid = pos < torch.clamp(blocks_per_item, max=k_sel).unsqueeze(1)
        selected = selected.masked_fill(~valid, 0)
        return {"route_probs": route_probs, "selected": selected, "valid": valid}


def flatten_selection(
    selected: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """[n_items, k] + mask -> (flat block ids grouped by item, counts)."""
    flat = selected.masked_select(valid)
    counts = valid.sum(dim=1)
    return flat, counts
