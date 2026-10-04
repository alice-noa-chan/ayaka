"""Inspect native LM/LoRA architecture on meta without loading any model weights."""

from __future__ import annotations

from collections import Counter

import torch

from ..backbone import detach_text_backbone, tiny_text_config
from ..checkpoint import apply_lora
from ..model.electra import ElectraDecisionModel
from .optimization import OptimizationConfig, optimization_plan


def inspect_direct_model(cfg, *, official_weight_elements=None, optimizations=None):
    """Load cached configuration only; instantiate every parameter on meta.

    The official full-checkpoint element count is used conservatively, even
    when only the text stack will be loaded. This verifies architecture and
    LoRA targets, not actual weight bytes, runtime parity or VRAM throughput.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    if cfg.readout != "lm":
        raise ValueError("direct preflight requires the native LM readout")
    if type(cfg.lora_r) is not int or cfg.lora_r < 1 or not cfg.lora_targets:
        raise ValueError("direct preflight requires positive LoRA rank and explicit targets")
    if cfg.backbone == "tiny":
        text_config = tiny_text_config()
    else:
        revision = cfg.backbone_revision
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)
        ):
            raise ValueError("cached native configuration requires an immutable revision")
        native = AutoConfig.from_pretrained(
            cfg.backbone, revision=revision, local_files_only=True, trust_remote_code=False
        )
        text_config = getattr(native, "text_config", native)
    context = getattr(text_config, "max_position_embeddings", None)
    if type(context) is not int or max(cfg.max_seq_len, cfg.serve_max_seq_len) > context:
        raise ValueError("requested context exceeds the actual native configuration")
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        lm = AutoModelForCausalLM.from_config(text_config, trust_remote_code=False)
        base_parameters = sum(p.numel() for p in lm.parameters())
        head, embedding = lm.get_output_embeddings(), lm.get_input_embeddings()
        tied = head.weight is embedding.weight
        output_shape = list(head.weight.shape)
        has_bias = getattr(head, "bias", None) is not None
        backbone = detach_text_backbone(lm)
        model = ElectraDecisionModel(cfg, backbone, text_config)
        model.backbone.requires_grad_(False)
        apply_lora(model)
        kernel_plan = optimization_plan(model, optimizations or OptimizationConfig())
    parameters = list(model.parameters())
    if not parameters or any(not p.is_meta for p in parameters):
        raise ValueError("native architecture inspection unexpectedly materialized parameters")
    total = sum(p.numel() for p in parameters)
    extra = total - base_parameters
    if extra < 0:
        raise ValueError("native output parameters were lost while wrapping the text backbone")
    if official_weight_elements is not None and (
        type(official_weight_elements) is not int or official_weight_elements < base_parameters
    ):
        raise ValueError("official full-weight count cannot be below native text parameters")
    conservative = (official_weight_elements or base_parameters) + extra
    if conservative > 14_000_000_000:
        raise ValueError("full checkpoint plus actual LoRA and decision modules exceeds 14B")
    targets = Counter(
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and len(module.lora_A)
    )
    if set(targets) != set(cfg.lora_targets):
        raise ValueError("actual LoRA placements do not cover the declared targets exactly")
    return {
        "repo": cfg.backbone,
        "revision": cfg.backbone_revision,
        "architecture": text_config.model_type,
        "native_text_parameters": base_parameters,
        "official_full_weight_elements": official_weight_elements,
        "adapter_and_decision_parameters": extra,
        "conservative_total_parameters": conservative,
        "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
        "lora_rank": cfg.lora_r,
        "lora_targets": dict(targets),
        "native_context": context,
        "training_context": cfg.max_seq_len,
        "serving_context": max(cfg.max_seq_len, cfg.serve_max_seq_len),
        "configured_serving_context": cfg.serve_max_seq_len,
        "native_output_shape": output_shape,
        "output_tied_to_input": tied,
        "output_has_bias": has_bias,
        "parameter_storage": "meta",
        "materialized_parameter_bytes": 0,
        "weights_loaded": False,
        "kernel_plan": kernel_plan,
        "forward_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "scope": "cached configuration and architecture only; actual weights/runtime remain unverified",
    }
