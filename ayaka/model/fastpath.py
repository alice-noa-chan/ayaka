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


def _masks(cfg, inputs_embeds, attention_mask, past_key_values, position_ids, explicit: bool):
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    kw = {
        "config": cfg,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        # explicit=False lets SDPA take its is_causal fast path (flash on
        # GPU) whenever no mask is needed, e.g. unpadded rows shorter than
        # the sliding window
        "allow_is_causal_skip": not explicit,
    }
    return {
        "full_attention": create_causal_mask(**kw),
        "sliding_attention": create_sliding_window_causal_mask(**kw),
    }


def _embed(text, input_ids):
    inputs_embeds = text.embed_tokens(input_ids)
    per_layer_inputs = None
    if text.hidden_size_per_layer_input:
        per_layer_inputs = text.get_per_layer_inputs(input_ids, inputs_embeds)
        per_layer_inputs = text.project_per_layer_inputs(inputs_embeds, per_layer_inputs)
    return inputs_embeds, per_layer_inputs


def _positions(input_ids, past_key_values):
    past = past_key_values.get_seq_length() if past_key_values is not None else 0
    return (torch.arange(input_ids.shape[1], device=input_ids.device) + past).unsqueeze(0)


def _run_layers(
    text, layers, hidden, per_layer_inputs, pos_emb, masks, position_ids, cache, shared_kv
):
    cfg = text.config
    for i in layers:
        lt = cfg.layer_types[i]
        hidden = text.layers[i](
            hidden,
            per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None,
            shared_kv_states=shared_kv,
            position_embeddings=pos_emb[lt],
            attention_mask=masks[lt],
            position_ids=position_ids,
            past_key_values=cache,
        )
    return hidden


def forward_kept(
    text,
    input_ids: torch.Tensor,
    keep: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values=None,
    normalize_spans: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (final normed states at ``keep`` [B, K, D],
           states after the last non-shared layer [B, S, D]).

    normalize_spans=False defers normalization to gathered option tokens;
    the answer states are always normalized. The default keeps the original
    full normalized span-state API.

    keep: [B, K] indices into the current ``input_ids`` rows.
    """
    cfg = text.config
    first_shared = kv_shared_start(cfg)
    if first_shared is None:
        raise ValueError("backbone has no KV-shared layers; use the regular forward")
    inputs_embeds, per_layer_inputs = _embed(text, input_ids)
    if position_ids is None:
        position_ids = _positions(input_ids, past_key_values)
    mask_args = (cfg, inputs_embeds, attention_mask, past_key_values, position_ids)
    masks = _masks(*mask_args, explicit=False)
    sliced_masks = _masks(*mask_args, explicit=True)  # rows get gathered below
    pos_emb = {t: text.rotary_emb(inputs_embeds, position_ids, t) for t in text.unique_layer_types}
    shared_kv: dict = {}

    hidden = _run_layers(
        text,
        range(first_shared),
        inputs_embeds,
        per_layer_inputs,
        pos_emb,
        masks,
        position_ids,
        past_key_values,
        shared_kv,
    )
    below_shared = text.norm(hidden) if normalize_spans else hidden

    # ---- shared layers: only the kept query rows
    kept_pli = _rows(per_layer_inputs, keep) if per_layer_inputs is not None else None
    kept_pe = {t: tuple(_rows(x, keep) for x in pe) for t, pe in pos_emb.items()}
    kept_masks = {
        t: _rows(m.transpose(1, 2), keep).transpose(1, 2) for t, m in sliced_masks.items()
    }
    hidden = _run_layers(
        text,
        range(first_shared, cfg.num_hidden_layers),
        _rows(hidden, keep),
        kept_pli,
        kept_pe,
        kept_masks,
        _rows(position_ids, keep),
        past_key_values,
        shared_kv,
    )
    return text.norm(hidden), below_shared


def prefix_cache(text, prefix_ids: torch.Tensor, cache=None):
    """Encode a shared prefix for later suffix branches, stopping before
    the KV-shared layers: their outputs at prefix positions are never
    read (suffix queries use the non-shared layers' cached K/V), so the
    remaining layers are skipped entirely. ``cache`` continues an
    already-encoded head (e.g. the constant system prompt). Returns the
    filled cache."""
    from transformers.cache_utils import DynamicCache

    cfg = text.config
    cache = cache if cache is not None else DynamicCache(config=cfg)
    inputs_embeds, per_layer_inputs = _embed(text, prefix_ids)
    position_ids = _positions(prefix_ids, cache)
    masks = _masks(cfg, inputs_embeds, None, cache, position_ids, explicit=False)
    pos_emb = {t: text.rotary_emb(inputs_embeds, position_ids, t) for t in text.unique_layer_types}
    _run_layers(
        text,
        range(kv_shared_start(cfg)),
        inputs_embeds,
        per_layer_inputs,
        pos_emb,
        masks,
        position_ids,
        cache,
        {},
    )
    return cache
