"""Offline detached-text LoRA -> full native causal LM adapter export.

PEFT's missing-key warnings cannot establish that trained weights loaded. This
export derives full module names from the native architecture on meta, scopes
targets to the text stack, and requires every expected A/B tensor and shape.
Only the small adapter is materialized. Native weights are never downloaded.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from .checkpoint import load_config, load_head
from .eval.read_artifact import fingerprint
from .input_contract import read_contract

VERSION = "ayaka-native-lm-adapter-export-1"
RECEIPT = "ayaka_adapter_export.json"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_layout(cfg, adapter, native_path):
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from peft.tuners.tuners_utils import check_target_module_exists
    from transformers import AutoConfig, AutoModelForCausalLM

    native_cfg = AutoConfig.from_pretrained(
        native_path,
        revision=cfg.backbone_revision if native_path == cfg.backbone else None,
        local_files_only=True,
        trust_remote_code=False,
    )
    lora = LoraConfig.from_pretrained(str(adapter), local_files_only=True)
    if (
        lora.bias != "none"
        or lora.modules_to_save
        or lora.use_dora
        or lora.target_parameters
        or lora.rank_pattern
        or lora.alpha_pattern
        or lora.use_rslora
        or lora.r != cfg.lora_r
        or lora.lora_alpha != cfg.lora_alpha
    ):
        raise ValueError("portable direct adapters require complete standard unbiased A/B LoRA")
    if lora.task_type is not None:
        raise ValueError("expected the detached-text Ayaka LoRA wrapper")
    with torch.device("meta"):
        native = AutoModelForCausalLM.from_config(native_cfg, trust_remote_code=False)
        embedding = native.get_input_embeddings()
        stacks = []
        for name, module in native.named_modules():
            if (
                not name
                or not hasattr(module, "layers")
                or not callable(getattr(module, "get_input_embeddings", None))
            ):
                continue
            try:
                current = module.get_input_embeddings()
            except (NotImplementedError, AttributeError):
                continue  # audio/vision PreTrainedModels may inherit the unimplemented method
            if current is embedding:
                stacks.append((name, module))
        if len(stacks) != 1:
            raise ValueError("native causal LM must expose one identifiable decoder text stack")
        prefix, text = stacks[0]
        targets = [
            f"{prefix}.{name}"
            for name, _ in text.named_modules()
            if name and check_target_module_exists(lora, name)
        ]
        if not targets:
            raise ValueError("detached-text LoRA targets do not exist in the native text stack")
        scoped = copy.deepcopy(lora)
        scoped.target_modules = set(targets)
        # Targets are now full module names; selection restrictions have
        # already been applied against the original detached-text names.
        scoped.layers_to_transform = scoped.layers_pattern = None
        scoped.exclude_modules = None
        scoped.base_model_name_or_path = cfg.backbone
        scoped.revision = cfg.backbone_revision
        scoped.auto_mapping = {
            "base_model_class": type(native).__name__,
            "parent_library": type(native).__module__,
        }
        wrapped = get_peft_model(native, scoped)
        expected = get_peft_model_state_dict(wrapped, save_embedding_layers=False)
        shapes = {name: tuple(tensor.shape) for name, tensor in expected.items()}
    config = scoped.to_dict()
    # get_peft_model may record an in-memory config's empty _name_or_path.
    config.update(base_model_name_or_path=cfg.backbone, revision=cfg.backbone_revision)
    return prefix, shapes, config, fingerprint(native_cfg.to_dict())


def export_hf_adapter(checkpoint, output, *, backbone_path=None):
    """Export a new directory, rejecting omissions instead of accepting PEFT warnings.

    This is a raw native-LM readout artifact. Use the source Ayaka checkpoint
    for calibrated typed serving; ordinary HF readers do not load its head/T.
    """
    from safetensors.torch import load_file, save_file

    source, destination = Path(checkpoint).resolve(), Path(output).resolve()
    if destination.exists():
        raise ValueError("portable adapter output must be a new directory")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("portable adapter output must be outside the source checkpoint")
    tracked = [
        source / name
        for name in (
            "ayaka_config.json",
            "electra_config.json",
            "meta.json",
            "adapter/adapter_model.safetensors",
            "adapter/adapter_config.json",
            "head.safetensors",
            "head.pt",
        )
    ]

    def capture():
        return {str(path): _sha(path) if path.is_file() else None for path in tracked}

    before = capture()
    cfg = load_config(str(source))
    contract = read_contract(source, cfg)
    if (
        cfg.readout != "lm"
        or contract is None
        or contract["input_encoding"]["encoder"] != "swift_canonical"
    ):
        raise ValueError("portable native adapters require a recipe-bound Swift LM checkpoint")
    if backbone_path is not None and (not str(backbone_path) or not Path(backbone_path).is_dir()):
        raise ValueError("explicit backbone_path must be an existing local native LM directory")
    native_path = str(backbone_path) if backbone_path is not None else cfg.backbone
    if native_path == "tiny":
        raise ValueError("random tiny checkpoints need an explicit saved native backbone path")
    if not Path(native_path).is_dir() and not cfg.backbone_revision:
        raise ValueError("remote native architecture requires the pinned checkpoint revision")
    adapter = source / "adapter"
    weights_path, config_path = (
        adapter / "adapter_model.safetensors",
        adapter / "adapter_config.json",
    )
    prefix, expected, config, native_config_sha = _native_layout(cfg, adapter, native_path)
    if backbone_path is not None:
        config.update(base_model_name_or_path=str(Path(native_path).resolve()), revision=None)
    weights = load_file(str(weights_path))
    converted, mapping = {}, {}
    for name, tensor in weights.items():
        old_prefix = "base_model.model."
        if not name.startswith(old_prefix) or not name.endswith(
            (".lora_A.weight", ".lora_B.weight")
        ):
            raise ValueError(f"unexpected detached LoRA tensor: {name}")
        mapped = old_prefix + prefix + "." + name[len(old_prefix) :]
        if mapped not in expected or tuple(tensor.shape) != expected[mapped]:
            raise ValueError(f"native LoRA key or shape differs: {name}")
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"LoRA tensor must contain finite floating weights: {name}")
        converted[mapped], mapping[name] = tensor.contiguous(), mapped
    if set(converted) != set(expected):
        missing = sorted(set(expected) - set(converted))
        raise ValueError(
            f"detached adapter is missing {len(missing)} native LoRA tensors: {missing[:4]}"
        )
    head = load_head(str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".ayaka-native-adapter-"
    ) as temp:
        stage = Path(temp) / "adapter"
        stage.mkdir()
        config["inference_mode"] = True
        # PEFT's writer serializes set-valued target_modules in its official format.
        from peft import LoraConfig

        LoraConfig(**config).save_pretrained(str(stage))
        portable_config = json.loads((stage / "adapter_config.json").read_text(encoding="utf-8"))
        portable_config["target_modules"] = sorted(portable_config["target_modules"])
        (stage / "adapter_config.json").write_text(
            json.dumps(portable_config, indent=2, sort_keys=True), encoding="utf-8"
        )
        save_file(converted, str(stage / "adapter_model.safetensors"), metadata={"format": "pt"})
        receipt = {
            "version": VERSION,
            "source_config_sha256": before[str(source / "ayaka_config.json")]
            or before[str(source / "electra_config.json")],
            "source_metadata_sha256": before[str(source / "meta.json")],
            "source_adapter_config_sha256": before[str(config_path)],
            "source_adapter_weights_sha256": before[str(weights_path)],
            "source_head_sha256": before[str(source / "head.safetensors")]
            or before[str(source / "head.pt")],
            "adapter_config_sha256": _sha(stage / "adapter_config.json"),
            "adapter_weights_sha256": _sha(stage / "adapter_model.safetensors"),
            "native_model": cfg.backbone,
            "native_revision": cfg.backbone_revision,
            "resolved_base_model": config["base_model_name_or_path"],
            "resolved_base_revision": config["revision"],
            "native_config_sha256": native_config_sha,
            "text_module": prefix,
            "tensor_count": len(converted),
            "key_mapping": mapping,
            **contract,
            "readout": "canonical_letter_raw",
            "calibration_applied": False,
            "source_temperature": head["temperature"].tolist(),
            "calibrated_serving": "use the original Ayaka checkpoint, not a raw HF reader",
            "native_weights_loaded": False,
            "gpu_seconds": 0,
        }
        (stage / RECEIPT).write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (stage / "README.md").write_text(
            "# Ayaka native LM adapter\n\n"
            "Offline conversion of detached-text LoRA to a full native causal LM layout. "
            "Every expected adapter tensor is retained. This artifact provides raw canonical "
            "letter logits/probabilities; it does not apply the checkpoint's typed/length "
            "temperatures. Use the original Ayaka checkpoint for calibrated serving. "
            "Architecture/shape validation is not model accuracy or loaded-weight attestation.\n",
            encoding="utf-8",
        )
        if capture() != before:
            raise ValueError("source checkpoint changed during adapter conversion; discard export")
        if destination.exists():
            raise ValueError("portable adapter output appeared during conversion")
        os.replace(stage, destination)
    return receipt


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone-path", help="local saved native LM for offline mechanics tests")
    args = parser.parse_args(argv)
    result = export_hf_adapter(args.checkpoint, args.out, backbone_path=args.backbone_path)
    print(json.dumps({k: v for k, v in result.items() if k != "key_mapping"}, indent=2))
    return result


if __name__ == "__main__":
    main()
