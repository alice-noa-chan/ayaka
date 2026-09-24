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


def load_text_backbone(repo: str, dtype: torch.dtype = torch.bfloat16, device="cpu", seed: int = 0):
    """Return (Gemma4TextModel, text_config).

    ``repo == "tiny"`` builds a random CPU test stack. A multimodal Gemma 4
    checkpoint (hub repo) has its text weights remapped out; a text-only
    directory (an Electra export) loads directly.
    """
    from transformers import AutoConfig, Gemma4ForCausalLM

    if repo == "tiny":
        torch.manual_seed(seed)
        cfg = tiny_text_config()
        lm = Gemma4ForCausalLM(cfg).to(dtype)
    else:
        any_cfg = AutoConfig.from_pretrained(repo)
        text_only = any_cfg.model_type == "gemma4_text"
        cfg = any_cfg if text_only else any_cfg.text_config
        kwargs = {} if text_only else {"key_mapping": TEXT_KEY_MAPPING}
        # load straight onto the target device: no full CPU copy of the weights
        dev = torch.device(device)
        lm = Gemma4ForCausalLM.from_pretrained(
            repo,
            config=cfg,
            dtype=dtype,
            attn_implementation="sdpa",
            device_map={"": dev.index or 0} if dev.type == "cuda" else None,
            **kwargs,
        )
    text = lm.model  # Gemma4TextModel; lm_head is tied to text.embed_tokens
    del lm
    return text.to(device), cfg


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
    return softcap((hidden.float() * rows.float()).sum(-1), cap)
