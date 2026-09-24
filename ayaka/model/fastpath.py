"""Exact position pruning in Gemma 4's KV-shared layers.

Gemma 4 E2B/E4B end with ``num_kv_shared_layers`` layers that compute no
keys/values of their own: they attend with the K/V stored by the last
non-shared layer of the same attention type. So from the first shared
layer on, a token's hidden state influences nothing except that token's
own output. Electra reads only a few positions (the answer position and
the option spans), so every other position can be dropped there without
changing a single output value. Option spans are read from the states
just below the shared layers (``span_layer``), so in the shared layers
only one position per question — the answer position — is computed.

On E2B the 20 shared layers (of 35) also have double-wide MLPs, about 70%
of per-token FLOPs; on E4B, 18 of 42 layers. 12B has no KV sharing, so
this path does not apply to it (``kv_shared_start`` returns None).

The prefix + shared-layer cache path is exact too: the last non-shared
layer stores its full-length K/V (cached prefix included) for the shared
layers to read, independently of which query rows run afterwards.
"""

from __future__ import annotations

import torch


def kv_shared_start(config) -> int | None:
    n = config.num_hidden_layers
    shared = getattr(config, "num_kv_shared_layers", 0) or 0
    first = n - shared
    return first if shared > 0 and 0 < first < n else None


def _rows(t: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """t: [B or 1, S, ...] -> [B, K, ...] gathered along dim 1."""
    t = t.expand(keep.shape[0], *t.shape[1:])
    idx = keep.view(*keep.shape, *([1] * (t.dim() - 2))).expand(*keep.shape, *t.shape[2:])
    return torch.gather(t, 1, idx)


def forward_kept(
    text,
    input_ids: torch.Tensor,
    keep: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (final normed states at ``keep`` [B, K, D],
           normed states of every position after the last non-shared layer [B, S, D]).

    keep: [B, K] indices into the current ``input_ids`` rows.
    """
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    cfg = text.config
    first_shared = kv_shared_start(cfg)
    if first_shared is None:
        raise ValueError("backbone has no KV-shared layers; use the regular forward")

    inputs_embeds = text.embed_tokens(input_ids)
    per_layer_inputs = None
    if text.hidden_size_per_layer_input:
        per_layer_inputs = text.get_per_layer_inputs(input_ids, inputs_embeds)
        per_layer_inputs = text.project_per_layer_inputs(inputs_embeds, per_layer_inputs)
    if position_ids is None:
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = (torch.arange(input_ids.shape[1], device=input_ids.device) + past).unsqueeze(
            0
        )

    mask_kwargs = {
        "config": cfg,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "allow_is_causal_skip": False,  # explicit masks: query rows get sliced below
    }
    masks = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }
    hidden = inputs_embeds
    pos_emb = {t: text.rotary_emb(hidden, position_ids, t) for t in text.unique_layer_types}
    shared_kv: dict = {}

    def run(i, h, pli, pe, mask, pos):
        return text.layers[i](
            h,
            pli,
            shared_kv_states=shared_kv,
            position_embeddings=pe,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=past_key_values,
        )

    for i in range(first_shared):
        lt = cfg.layer_types[i]
        pli = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
        hidden = run(i, hidden, pli, pos_emb[lt], masks[lt], position_ids)
    below_shared = text.norm(hidden)

    # ---- shared layers: only the kept query rows
    hidden = _rows(hidden, keep)
    kept_pos = _rows(position_ids, keep)
    kept_pe = {t: tuple(_rows(x, keep) for x in pe) for t, pe in pos_emb.items()}
    kept_masks = {t: _rows(m.transpose(1, 2), keep).transpose(1, 2) for t, m in masks.items()}
    for i in range(first_shared, cfg.num_hidden_layers):
        lt = cfg.layer_types[i]
        pli = _rows(per_layer_inputs[:, :, i, :], keep) if per_layer_inputs is not None else None
        hidden = run(i, hidden, pli, kept_pe[lt], kept_masks[lt], kept_pos)
    return text.norm(hidden), below_shared


def prefix_cache(text, prefix_ids: torch.Tensor):
    """Encode a shared prefix for later suffix branches, stopping before
    the KV-shared layers: their outputs at prefix positions are never
    read (suffix queries use the non-shared layers' cached K/V), so the
    remaining layers are skipped entirely. Returns the filled cache."""
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    cfg = text.config
    first_shared = kv_shared_start(cfg)
    cache = DynamicCache(config=cfg)
    inputs_embeds = text.embed_tokens(prefix_ids)
    per_layer_inputs = None
    if text.hidden_size_per_layer_input:
        per_layer_inputs = text.get_per_layer_inputs(prefix_ids, inputs_embeds)
        per_layer_inputs = text.project_per_layer_inputs(inputs_embeds, per_layer_inputs)
    position_ids = torch.arange(prefix_ids.shape[1], device=prefix_ids.device).unsqueeze(0)
    mask_kwargs = {
        "config": cfg,
        "inputs_embeds": inputs_embeds,
        "attention_mask": None,
        "past_key_values": cache,
        "position_ids": position_ids,
    }
    masks = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }
    hidden = inputs_embeds
    pos_emb = {t: text.rotary_emb(hidden, position_ids, t) for t in text.unique_layer_types}
    shared_kv: dict = {}
    for i in range(first_shared):
        lt = cfg.layer_types[i]
        pli = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
        hidden = text.layers[i](
            hidden,
            pli,
            shared_kv_states=shared_kv,
            position_embeddings=pos_emb[lt],
            attention_mask=masks[lt],
            position_ids=position_ids,
            past_key_values=cache,
        )
    return cache
