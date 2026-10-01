"""Gemma 4 text backbone loading + restricted label readout.

Gemma 4 checkpoints are multimodal (``Gemma4ForConditionalGeneration``);
only the text stack is loaded — vision/audio towers never reach memory.
Readout touches just the label-token rows of the tied embedding, so the
262K-vocab LM head is never materialized:

    logit(label) = softcap(h · E[label]),  softcap(x) = c · tanh(x / c)
"""

from __future__ import annotations

import torch

TEXT_KEY_MAPPING = {r"^model\.language_model\.": "model."}


def tiny_text_config(vocab_size: int = 512):
    from transformers import Gemma4TextConfig

    return Gemma4TextConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=vocab_size,
        num_kv_shared_layers=2,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
        sliding_window=16,
        max_position_embeddings=4096,
        pad_token_id=0,
        final_logit_softcapping=30.0,
    )


def load_text_backbone(
    repo: str,
    dtype: torch.dtype = torch.bfloat16,
    device="cpu",
    seed: int = 0,
    revision: str | None = None,
):
    """Return (Gemma4TextModel, text_config).

    ``repo == "tiny"`` builds a random CPU test stack. A multimodal Gemma 4
    checkpoint (hub repo) has its text weights remapped out; a text-only
    directory (an Electra export) loads directly.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, Gemma4ForCausalLM

    if repo == "tiny":
        torch.manual_seed(seed)
        cfg = tiny_text_config()
        lm = Gemma4ForCausalLM(cfg).to(dtype)
    else:
        any_cfg = AutoConfig.from_pretrained(repo, revision=revision)
        cfg = getattr(any_cfg, "text_config", any_cfg)
        gemma = cfg.model_type == "gemma4_text"
        mapping = TEXT_KEY_MAPPING if gemma else {r"^model\.language_model\.": "model."}
        kwargs = {} if cfg is any_cfg else {"key_mapping": mapping}
        # load straight onto the target device: no full CPU copy of the weights
        dev = torch.device(device)
        loader = Gemma4ForCausalLM if gemma else AutoModelForCausalLM
        lm = loader.from_pretrained(
            repo,
            config=cfg,
            dtype=dtype,
            attn_implementation="sdpa",
            revision=revision,
            device_map={"": dev.index or 0} if dev.type == "cuda" else None,
            **kwargs,
        )
    text = detach_text_backbone(lm)
    del lm
    return text.to(device), cfg


def detach_text_backbone(lm):
    """Keep an untied native output head; never substitute input embeddings."""
    text = lm.base_model
    head = lm.get_output_embeddings()
    embedding = text.get_input_embeddings()
    if head.weight is not embedding.weight:
        text.add_module("_ayaka_lm_head", head)
    return text


def output_rows(text, ids):
    head = getattr(text, "_ayaka_lm_head", None)
    return embedding_rows(head if head is not None else text.get_input_embeddings(), ids)


def native_logits(text, hidden, ids=None):
    """Restricted or full logits including native bias, scaling and softcap."""
    head = getattr(text, "_ayaka_lm_head", None)
    if ids is None:
        embed = text.get_input_embeddings()
        logits = (
            head(hidden.to(head.weight.dtype))
            if head is not None
            else torch.nn.functional.linear(hidden.to(embed.weight.dtype), embed.weight)
        )
    else:
        rows = output_rows(text, ids)
        logits = (hidden.float() * rows.float()).sum(-1)
        if head is not None and getattr(head, "bias", None) is not None:
            logits = logits + head.bias[ids]
    logits = logits / getattr(text.config, "logits_scaling", 1.0)
    return softcap(logits, getattr(text.config, "final_logit_softcapping", None))


def softcap(logits: torch.Tensor, cap: float | None) -> torch.Tensor:
    if not cap:
        return logits
    return torch.tanh(logits / cap) * cap


def embedding_rows(embed: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """Unscaled embedding rows == tied LM-head rows (int8 embeddings too)."""
    if hasattr(embed, "rows"):
        return embed.rows(ids)
    return embed.weight[ids]


def label_logits(hidden: torch.Tensor, rows: torch.Tensor, cap: float | None) -> torch.Tensor:
    """hidden: [n, D] answer-position states (one per candidate row);
    rows: [n, D] tied LM-head rows of each candidate's readout token.
    Returns [n] fp32 logits."""
    dt = torch.promote_types(hidden.dtype, torch.float32)
    return softcap((hidden.to(dt) * rows.to(dt)).sum(-1), cap)
