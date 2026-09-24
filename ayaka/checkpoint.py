"""LoRA wrapping + checkpoint save/load.

A checkpoint directory holds only what training changed:

    electra_config.json   ElectraConfig (backbone repo, head sizes, ...)
    adapter/              PEFT LoRA adapter for the Gemma 4 text stack
    head.pt               pointer head + gate + per-primitive temperatures
    meta.json             run metadata (steps, metrics, teacher lineage)

Loading merges the adapter into the backbone for plain-speed inference.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import torch

from .config import ElectraConfig
from .model.electra import ElectraDecisionModel


def apply_lora(model: ElectraDecisionModel) -> ElectraDecisionModel:
    from peft import LoraConfig, get_peft_model

    cfg = model.cfg
    lcfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets),
        bias="none",
    )
    model.backbone = get_peft_model(model.backbone, lcfg)
    return model


def save_checkpoint(model: ElectraDecisionModel, path: str, meta: dict | None = None) -> str:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "electra_config.json"), "w") as f:
        json.dump(asdict(model.cfg), f, indent=2)
    if hasattr(model.backbone, "save_pretrained") and hasattr(model.backbone, "peft_config"):
        model.backbone.save_pretrained(os.path.join(path, "adapter"))
    torch.save(model.head_state_dict(), os.path.join(path, "head.pt"))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta or {}, f, indent=2, default=str)
    return path


def load_config(path: str) -> ElectraConfig:
    with open(os.path.join(path, "electra_config.json")) as f:
        d = json.load(f)
    d["lora_targets"] = tuple(d.get("lora_targets", ()))
    return ElectraConfig(**d)


def load_checkpoint(
    path: str,
    device="cpu",
    dtype: torch.dtype = torch.bfloat16,
    merge: bool = True,
    trainable: bool = False,
) -> ElectraDecisionModel:
    """Rebuild a model from a checkpoint dir. ``trainable`` keeps the
    adapter unmerged and trainable (resume / continue training)."""
    cfg = load_config(path)
    model = ElectraDecisionModel.from_config(cfg, dtype=dtype, device=device)
    adapter = os.path.join(path, "adapter")
    if os.path.isdir(adapter):
        from peft import PeftModel

        model.backbone = PeftModel.from_pretrained(model.backbone, adapter, is_trainable=trainable)
        if merge and not trainable:
            model.backbone = model.backbone.merge_and_unload()
    model.load_head_state_dict(torch.load(os.path.join(path, "head.pt"), map_location=device))
    return model.to(device)
