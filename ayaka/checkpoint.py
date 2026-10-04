"""LoRA wrapping + checkpoint save/load.

A checkpoint directory holds only what training changed:

    ayaka_config.json     model config (backbone repo + pinned revision, head sizes, ...)
    electra_config.json   the same config under its legacy name ("Electra" was the
                          project's pre-Gemma model name); kept for older loaders
    adapter/              PEFT LoRA adapter for the Gemma 4 text stack
    head.safetensors      pointer head + gate + per-primitive temperatures
    head.pt               the same tensors as a legacy torch pickle (training only)
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


CONFIG_FILES = ("ayaka_config.json", "electra_config.json")  # preferred first


def config_path(path: str) -> str:
    for name in CONFIG_FILES:
        if os.path.exists(os.path.join(path, name)):
            return os.path.join(path, name)
    raise FileNotFoundError(f"no {' or '.join(CONFIG_FILES)} in {path}")


def write_config(cfg_dict: dict, path: str) -> None:
    for name in CONFIG_FILES:
        with open(os.path.join(path, name), "w") as f:
            json.dump(cfg_dict, f, indent=2)


def save_head(head_sd: dict, path: str, pickle: bool = True) -> None:
    """head.safetensors (flat: head.<name>, gate, temperature) and, for older
    loaders, the same nested dict as head.pt."""
    from safetensors.torch import save_file

    flat = {f"head.{k}": v.detach().cpu().contiguous() for k, v in head_sd["head"].items()}
    flat["gate"] = head_sd["gate"].detach().cpu().contiguous()
    flat["temperature"] = head_sd["temperature"].detach().cpu().contiguous()
    save_file(flat, os.path.join(path, "head.safetensors"), metadata={"format": "pt"})
    if pickle:
        torch.save(head_sd, os.path.join(path, "head.pt"))


def load_head(path: str, device="cpu") -> dict:
    """Prefer head.safetensors; fall back to the legacy head.pt pickle."""
    st = os.path.join(path, "head.safetensors")
    if os.path.exists(st):
        from safetensors.torch import load_file

        flat = load_file(st, device=str(device))
        head = {k[len("head.") :]: v for k, v in flat.items() if k.startswith("head.")}
        return {"head": head, "gate": flat["gate"], "temperature": flat["temperature"]}
    return torch.load(os.path.join(path, "head.pt"), map_location=device, weights_only=True)


def save_checkpoint(model: ElectraDecisionModel, path: str, meta: dict | None = None) -> str:
    os.makedirs(path, exist_ok=True)
    write_config(asdict(model.cfg), path)
    if hasattr(model.backbone, "save_pretrained") and hasattr(model.backbone, "peft_config"):
        model.backbone.save_pretrained(os.path.join(path, "adapter"))
    save_head(model.head_state_dict(), path)
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta or {}, f, indent=2, default=str)
    return path


def resolve_checkpoint(path_or_repo: str, revision: str | None = None) -> str:
    """A local checkpoint dir, or ``<user>/<repo>`` downloaded from the Hub.

    Only checkpoint files are fetched (config, head, adapter, metadata); the
    base model comes from its own repo at the config's pinned revision.
    """
    if os.path.isdir(path_or_repo):
        return path_or_repo
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=path_or_repo, revision=revision)


def load_config(path: str) -> ElectraConfig:
    with open(config_path(path)) as f:
        d = json.load(f)
    d["lora_targets"] = tuple(d.get("lora_targets", ()))
    return ElectraConfig(**d)


def load_checkpoint(
    path: str,
    device="cpu",
    dtype: torch.dtype = torch.bfloat16,
    merge: bool = True,
    trainable: bool = False,
    *,
    backbone_path: str | None = None,
    local_files_only: bool = False,
    strict_loading: bool = False,
) -> ElectraDecisionModel:
    """Rebuild a model from a checkpoint dir. ``trainable`` keeps the
    adapter unmerged and trainable (resume / continue training)."""
    cfg = load_config(path)
    model = ElectraDecisionModel.from_config(
        cfg,
        dtype=dtype,
        device=device,
        backbone_path=backbone_path,
        local_files_only=local_files_only,
        strict_loading=strict_loading,
    )
    adapter = os.path.join(path, "adapter")
    if os.path.isdir(adapter):
        from peft import PeftModel

        model.backbone = PeftModel.from_pretrained(model.backbone, adapter, is_trainable=trainable)
        if merge and not trainable:
            model.backbone = model.backbone.merge_and_unload()
    model.load_head_state_dict(load_head(path, device))
    return model.to(device)


def compact_checkpoint(
    src: str,
    dst: str,
    dtype: torch.dtype = torch.bfloat16,
    backbone_revision: str | None = None,
) -> dict:
    """Copy a checkpoint with its LoRA tensors stored in ``dtype``.

    Training keeps adapter weights in fp32 (1.05 GB for large); general-purpose
    compression saves only 7-10% of them. A bf16 copy halves the size but is an
    approximation, not a lossless copy: PEFT keeps adapter weights in fp32 even
    on a bf16 backbone, so rounding them changes merged weights slightly (on
    the tiny test model, decision probabilities moved by ~0.003). Check parity
    on a real checkpoint before serving a compacted copy. The head, config and
    metadata are copied unchanged; meta.json records the dtype.
    """
    import shutil

    from safetensors.torch import load_file, save_file

    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError("compact into a new directory")
    shutil.copytree(src, dst)
    weights = os.path.join(dst, "adapter", "adapter_model.safetensors")
    before = os.path.getsize(weights)
    tensors = {
        k: v.to(dtype) if v.is_floating_point() else v for k, v in load_file(weights).items()
    }
    save_file(tensors, weights, metadata={"format": "pt"})
    with open(config_path(dst)) as f:
        cfg = json.load(f)
    if backbone_revision:
        cfg["backbone_revision"] = backbone_revision
    write_config(cfg, dst)  # both names, identical content
    # release copies carry the head as safetensors only (no pickle to load)
    head_sd = load_head(dst)
    save_head(head_sd, dst, pickle=False)
    legacy = os.path.join(dst, "head.pt")
    if os.path.exists(legacy):
        os.remove(legacy)
    meta_path = os.path.join(dst, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["adapter_dtype"] = str(dtype).removeprefix("torch.")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return {"adapter_bytes_before": before, "adapter_bytes_after": os.path.getsize(weights)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="checkpoint utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compact", help="copy a checkpoint with a bf16 LoRA adapter")
    c.add_argument("src")
    c.add_argument("dst")
    c.add_argument("--backbone-revision", default=None, help="pin the base model revision")
    args = ap.parse_args()
    print(
        json.dumps(compact_checkpoint(args.src, args.dst, backbone_revision=args.backbone_revision))
    )
