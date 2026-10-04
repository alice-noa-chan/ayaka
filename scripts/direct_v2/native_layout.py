"""Verify pinned native text tensor coverage on CPU before paid model loading.

This checks headers against the actual installed HF loader's meta architecture.
It does not load pretrained tensors or prove CUDA kernels or decision quality.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from ayaka.backbone import TEXT_KEY_MAPPING
from ayaka.eval.read_artifact import fingerprint
from ayaka.training.direct_state import file_digest
from ayaka.training.native_snapshot import verify_snapshot
from ayaka.training.prepare_v2 import canonical

VERSION = "ayaka-native-text-layout-1"


def audit_layout(record, native_path):
    """Verify bytes, names, shapes and genuine tied aliases without tensor reads.

    Reject converters that need tensor transformations: a shape-only inspector
    cannot certify their result. Extra tensors must belong to explicitly known
    Gemma multimodal components; unexpected text parameters always fail.
    The separately trusted snapshot record supplies provenance, not this report.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, Gemma4ForCausalLM
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

    root = verify_snapshot(record, path=native_path)
    config = AutoConfig.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    text_config = getattr(config, "text_config", config)
    mapping = TEXT_KEY_MAPPING if config is not text_config else None
    loader = Gemma4ForCausalLM if text_config.model_type == "gemma4_text" else AutoModelForCausalLM
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        lm = (
            loader.from_config(text_config, trust_remote_code=False)
            if loader is AutoModelForCausalLM
            else loader(text_config)
        )
    expected = lm.state_dict(keep_vars=True)
    if not expected or any(not value.is_meta for value in expected.values()):
        raise ValueError("native layout inspection unexpectedly allocated tensor storage")
    transforms = get_model_conversion_mapping(lm, mapping)
    if any(not isinstance(entry, (WeightRenaming, WeightConverter)) for entry in transforms):
        raise ValueError("unsupported native loader transformation in header-only inspection")
    renamings = [entry for entry in transforms if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in transforms if isinstance(entry, WeightConverter)]
    aliases = defaultdict(list)
    for key, tensor in expected.items():
        aliases[id(tensor)].append(key)
    tensors, mapped, ignored = {}, {}, []
    for name in sorted(record["files"]):
        if not name.endswith(".safetensors"):
            continue
        # Header inspection must not reserve writable TorchStorage for an
        # entire pretrained shard. No NumPy tensor values are materialized.
        with safe_open(root / name, framework="np", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 -- safe_open is not a mapping
                view = handle.get_slice(key)
                shape = view.get_shape()
                tensors[key] = {"shape": shape, "dtype": view.get_dtype(), "file": name}
                target, conversion = rename_source_key(
                    key, renamings, converters, lm.base_model_prefix, expected
                )
                if target not in expected and key in expected:
                    target, conversion = rename_source_key(
                        key, [], [], lm.base_model_prefix, expected
                    )
                if target not in expected:
                    if not _auxiliary_key(config, key):
                        raise ValueError(f"unexpected native text tensor: {key}")
                    ignored.append(key)
                    continue
                if conversion is not None:
                    raise ValueError(f"native layout requires a runtime tensor conversion: {key}")
                if target in mapped:
                    raise ValueError(
                        f"native keys collide after loader mapping: {mapped[target]} / {key}"
                    )
                if shape != list(expected[target].shape):
                    raise ValueError(f"native text tensor shape mismatch: {key} -> {target}")
                mapped[target] = key
    if fingerprint(tensors) != record["tensor_schema_sha256"]:
        raise ValueError("native tensor headers changed during layout inspection")
    missing_groups = [keys for keys in aliases.values() if not set(keys).intersection(mapped)]
    if missing_groups:
        raise ValueError(
            f"native text parameters or persistent buffers are missing: {missing_groups}"
        )
    if any(len(set(keys).intersection(mapped)) > 1 for keys in aliases.values()):
        raise ValueError(
            "multiple serialized tied aliases require runtime tensor equality verification"
        )
    if file_digest(root / "config.json") != record["files"]["config.json"]["sha256"]:
        raise ValueError("native configuration changed during layout inspection")
    tied_aliases = {
        key: next(name for name in keys if name in mapped)
        for keys in aliases.values()
        for key in keys
        if key not in mapped
    }
    # Meta construction/header reads leave time for data bytes to change while
    # retaining identical shapes. Bind all current loader bytes again at exit.
    verify_snapshot(record, path=root)
    return {
        "version": VERSION,
        "inspector_sha256": file_digest(__file__),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "safetensors")
        },
        "snapshot_sha256": record["snapshot_sha256"],
        "tensor_schema_sha256": record["tensor_schema_sha256"],
        "text_architecture": text_config.model_type,
        "text_state_entries": len(expected),
        "matched_text_tensors": len(mapped),
        "tied_aliases": tied_aliases,
        "ignored_auxiliary_tensors": sorted(ignored),
        "text_dtype_entries": dict(Counter(tensors[source]["dtype"] for source in mapped.values())),
        "text_mapping_sha256": fingerprint(mapped),
        "parameter_storage": "meta",
        "materialized_parameter_bytes": 0,
        "pretrained_tensors_loaded": False,
        "gpu_allocated": False,
        "network_downloads": 0,
        "optimizer_steps": 0,
        "scope": "verified local bytes and text name/shape coverage; no runtime/kernel/quality proof",
    }


def _auxiliary_key(config, key):
    # Narrow to the native full-checkpoint families supported by our text loader.
    return (
        hasattr(config, "text_config")
        and config.model_type in {"gemma4", "gemma4_unified"}
        and key.startswith(
            (
                "model.vision_tower.",
                "model.audio_tower.",
                "model.embed_vision.",
                "model.embed_audio.",
                "model.vision_embedder.",
            )
        )
    )


def read_record(path, expected_sha256):
    raw = Path(path).read_bytes()
    from ayaka.training.prepare_v2 import sha256

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha256)
        or sha256(raw) != expected_sha256
    ):
        raise ValueError("native snapshot record differs from its external SHA256")
    return json.loads(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-record", type=Path, required=True)
    parser.add_argument("--expected-snapshot-record-sha256", required=True)
    parser.add_argument("--snapshot-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    destination = args.out.resolve()
    root = args.snapshot_path.resolve()
    if destination.exists() or destination == root or root in destination.parents:
        raise ValueError("layout report must be a new file outside the native directory")
    report = audit_layout(
        read_record(args.snapshot_record, args.expected_snapshot_record_sha256), root
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(canonical(report) + b"\n")
    print(json.dumps({"report_sha256": file_digest(destination), **report}, indent=2))


if __name__ == "__main__":
    main()
