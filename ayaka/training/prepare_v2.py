"""Create and audit an immutable v2 bundle without loading weights or training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from ..checkpoint import load_config
from ..data.candidate_v2 import candidate_curriculum, partition_diagnostics
from ..data.language_v2 import language_curriculum
from ..data.multimodal_v2 import image_curriculum
from ..data.reasoning_v2 import SPLITS, curriculum
from ..data.schema import Sample

VERSION = "ayaka-v2-pretraining-1"
LINEAGE_KEYS = ("source_lineage", "generator_template_id", "rule_combination", "document_voice")


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def prepared_items(sample, tok, cfg, backend):
    from ..prompt import render_prefix, render_question
    from .batching import question_view, sample_to_items
    from .candidates import proposal_items
    from .multimodal import image_items
    from .reasoning import reasoning_items

    # The legacy direct encoder truncates long states; supervised preparation
    # must reject overflow before that encoder can discard oracle evidence.
    if "media" not in sample.metadata:
        prefix = render_prefix(sample.state, tok)
        if any(
            len(prefix)
            + len(render_question(question_view(q), tok, cfg.max_label_candidates).suffix_ids)
            > cfg.max_seq_len
            for q in sample.questions
        ):
            raise ValueError("complete direct training row does not fit; do not truncate it")
    if "media" in sample.metadata:
        return image_items(sample, backend, sample.metadata.get("verified_traces"))
    if "proposal_supervision" in sample.metadata:
        return proposal_items(sample, tok, cfg)
    if sample.metadata.get("verified_traces"):
        return reasoning_items(sample, tok, cfg, sample.metadata["verified_traces"])
    return sample_to_items(sample, tok, cfg)


def cached_weights_ready(cfg):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = Path(
            hf_hub_download(
                cfg.backbone,
                "model.safetensors.index.json",
                revision=cfg.backbone_revision,
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError:
        try:
            hf_hub_download(
                cfg.backbone,
                "model.safetensors",
                revision=cfg.backbone_revision,
                local_files_only=True,
            )
            return True
        except LocalEntryNotFoundError:
            return False
    shards = set(json.loads(path.read_text(encoding="utf-8"))["weight_map"].values())
    return bool(shards) and all((path.parent / shard).is_file() for shard in shards)


def audit_splits(splits, *, development_only=False):
    required = tuple(s for s in SPLITS if s != "test") if development_only else SPLITS
    if (
        type(development_only) is not bool
        or set(splits) != set(required)
        or any(not splits[s] for s in required)
    ):
        raise ValueError("all required nonempty independent splits are required")
    seen = {key: {} for key in (*LINEAGE_KEYS, "content")}
    counts = {}
    for split, samples in splits.items():
        typed, languages, modalities, families = Counter(), Counter(), Counter(), Counter()
        for sample in samples:
            m = sample.metadata
            natural = m.get("data_kind") == "natural"
            from ..data.natural_training_v2 import valid_provenance

            if m.get("split") != split or (
                not valid_provenance(m) if natural else m.get("license") != "MIT"
            ):
                raise ValueError("split/provenance license mismatch")
            if not sample.questions or len({q.id for q in sample.questions}) != len(
                sample.questions
            ):
                raise ValueError("missing or duplicate question ids")
            serialized = sample.to_json()
            Sample.from_json(serialized)
            for question in sample.questions:
                if len(question.candidates) < 2 or not question.target_distribution:
                    raise ValueError("supervised questions require candidates and targets")
                typed[question.type] += 1
            languages[m.get("language", "unknown")] += 1
            modalities[m.get("modality", "text")] += 1
            families[m.get("task_family", "unknown")] += 1
            values = {key: m.get(key) for key in (("source_lineage",) if natural else LINEAGE_KEYS)}
            # Exclude gold targets: changed labels cannot conceal evidence leakage.
            values["content"] = sha256(
                canonical(
                    {
                        "state": sample.state,
                        "questions": [
                            {k: v for k, v in q.items() if k not in {"target_distribution", "id"}}
                            for q in serialized["questions"]
                        ],
                        "media": m.get("media"),
                    }
                )
            )
            for key, value in values.items():
                if not isinstance(value, str) or not value:
                    raise ValueError(f"missing provenance: {key}")
                previous = seen[key].get(value)
                if previous and previous != split:
                    raise ValueError(f"cross-split leakage in {key}: {previous}/{split}")
                seen[key][value] = split
        counts[split] = {
            "samples": len(samples),
            "questions": sum(typed.values()),
            "types": dict(typed),
            "languages": dict(languages),
            "modalities": dict(modalities),
            "families": dict(families),
        }
    return counts


def build_splits(per_type=32, image_cases=32, candidate_cases=32):
    if any(
        type(n) is not int or not 1 <= n <= 256 for n in (per_type, image_cases, candidate_cases)
    ):
        raise ValueError("curriculum counts must be integers in 1–256")
    splits = {}
    for split in SPLITS:
        samples = []
        for sample, traces in curriculum(split, per_type):
            sample.metadata.update(
                source_lineage=sample.metadata["case_facts_sha256"],
                modality="text",
                verified_traces=traces,
            )
            samples.append(sample)
        samples += image_curriculum(split, image_cases)
        samples += language_curriculum(split, per_type)
        candidates = candidate_curriculum(split, candidate_cases)
        samples += candidates + [partition_diagnostics(s) for s in candidates]
        splits[split] = samples
    return splits


def validate_bundle(path):
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError("unsupported preparation manifest")
    splits = {}
    expected = {f"{split}.jsonl" for split in SPLITS} | {
        "training_config.json",
        "model_preflight.json",
    }
    if set(manifest["files"]) != expected:
        raise ValueError("manifest must cover exactly the data, model audit and training config")
    for name, digest in manifest["files"].items():
        if sha256((root / name).read_bytes()) != digest:
            raise ValueError(f"bundle checksum mismatch: {name}")
    for split in SPLITS:
        splits[split] = [
            Sample.from_json(json.loads(line))
            for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    counts = audit_splits(splits)
    if counts != manifest["counts"]:
        raise ValueError("bundle count audit mismatch")
    return manifest, splits


def inspect_model(cfg, *, offline=True):
    """Count actual native architecture plus adapter/head on meta, without weights."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    from ..checkpoint import apply_lora
    from ..model.electra import ElectraDecisionModel
    from ..tokenization import HFTokenizer

    if (
        not cfg.backbone_revision
        or len(cfg.backbone_revision) != 40
        or any(c not in "0123456789abcdef" for c in cfg.backbone_revision)
    ):
        raise ValueError("pretraining requires an immutable 40-character backbone revision")
    config = AutoConfig.from_pretrained(
        cfg.backbone, revision=cfg.backbone_revision, local_files_only=offline
    )
    if config.model_type not in {"gemma4", "gemma4_unified"} or config.vision_config is None:
        raise ValueError("pretraining requires a supported native image architecture")
    processor = AutoProcessor.from_pretrained(
        cfg.backbone, revision=cfg.backbone_revision, local_files_only=offline
    )
    tok = HFTokenizer(processor.tokenizer, cfg.backbone)
    with torch.device("meta"):
        native_lm = AutoModelForImageTextToText.from_config(config)
        native = native_lm.model
        text = native.language_model
        if native_lm.get_output_embeddings().weight is not text.get_input_embeddings().weight:
            text.add_module("_ayaka_lm_head", native_lm.get_output_embeddings())
        model = ElectraDecisionModel(cfg, text, config.text_config)
        base_count = sum(p.numel() for p in native_lm.parameters())
        model.backbone.requires_grad_(False)
        apply_lora(model)
        native.language_model = model.backbone
    parameters = {id(p): p for p in [*native_lm.parameters(), *model.parameters()]}
    count = sum(p.numel() for p in parameters.values())
    if count > 14_000_000_000:
        raise ValueError("native model plus adapter and head exceed 14B parameters")
    from ..multimodal import ImageBackend

    backend = ImageBackend(native, processor, model, tok, processing_device="cpu")
    return (
        {
            "repo": cfg.backbone,
            "revision": cfg.backbone_revision,
            "architecture": config.model_type,
            "base_parameters": base_count,
            "total_parameters": count,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "native_context": config.text_config.max_position_embeddings,
            "training_context": cfg.max_seq_len,
            "vision_policy": "frozen",
            "weight_loading": "not_loaded_meta_only",
            "local_weights_ready": cached_weights_ready(cfg),
            "tokenizer": tok.name,
            "processor": type(processor).__name__,
            "parameter_limit": 14_000_000_000,
        },
        tok,
        backend,
    )


def prepare_bundle(
    out,
    cfg,
    *,
    offline=True,
    per_type=32,
    image_cases=32,
    candidate_cases=32,
    natural_data=False,
    reserved_evaluation=None,
):
    survey_path = (
        Path(__file__).resolve().parents[2] / "docs" / "experiments" / "v2_candidates.json"
    )
    survey = json.loads(survey_path.read_text(encoding="utf-8"))
    candidate = next(
        (
            c
            for c in survey["candidates"]
            if c["repo"] == cfg.backbone and c["revision"] == cfg.backbone_revision
        ),
        None,
    )
    if candidate is None or candidate["license"] not in {"apache-2.0", "mit"}:
        raise ValueError("model revision is missing from the approved license survey")
    root = Path(out)
    if root.exists():
        raise ValueError("prepare into a new directory; never overwrite an audited bundle")
    splits = build_splits(per_type, image_cases, candidate_cases)
    audit, tok, backend = inspect_model(cfg, offline=offline)
    if natural_data:
        from ..data.natural_training_v2 import local_sources, partition_sources

        if reserved_evaluation is None:
            raise ValueError("natural rehearsal requires the previous reserved evaluation file")
        reserved_raw = Path(reserved_evaluation).read_bytes()
        reserved = [Sample.from_json(row) for row in json.loads(reserved_raw)]
        if not reserved:
            raise ValueError("reserved natural evaluation must not be empty")

        def fits(sample):
            try:
                return all(
                    item.length <= cfg.max_seq_len
                    for item in prepared_items(sample, tok, cfg, backend)
                )
            except ValueError as exc:
                if "fit" in str(exc) or "overflow" in str(exc) or "truncate" in str(exc):
                    return False
                raise

        sources, provenance = local_sources()
        additions, natural_audit = partition_sources(sources, reserved, fits=fits)
        for split in SPLITS:
            splits[split].extend(additions[split])
        audit["natural_data"] = {
            **natural_audit,
            "sources": provenance,
            "reserved_evaluation_sha256": sha256(reserved_raw),
        }
    counts = audit_splits(splits)
    conservative = max(
        audit["total_parameters"],
        candidate["total_parameters"] + audit["total_parameters"] - audit["base_parameters"],
    )
    if conservative > 14_000_000_000:
        raise ValueError("official weight elements plus adapter/head exceed 14B")
    audit.update(
        official_weight_elements=candidate["total_parameters"],
        conservative_total_elements=conservative,
    )
    from .workload import describe_rows

    context, train_inventory = {}, []
    for split, samples in splits.items():
        max_length, questions = 0, 0
        for sample in samples:
            items = prepared_items(sample, tok, cfg, backend)
            if split == "train":
                train_inventory.append(describe_rows(sample, items))
            max_length = max(max_length, *(it.length for it in items))
            questions += len(items)
        if max_length > min(cfg.max_seq_len, audit["native_context"]):
            raise ValueError(f"{split} contains an overflowing training row")
        context[split] = {"prepared_rows": questions, "max_tokens": max_length}
        print(f"[prepare] {split}: {questions} rows, max {max_length} tokens", flush=True)
    audit.update(
        context_audit=context,
        license=candidate["license"],
        license_source=candidate["card_url"],
        train_inventory=train_inventory,
    )
    root.mkdir(parents=True, exist_ok=False)
    for split, samples in splits.items():
        (root / f"{split}.jsonl").write_bytes(
            b"\n".join(canonical(s.to_json()) for s in samples) + b"\n"
        )
    recipe = {
        "version": VERSION,
        "model": asdict(cfg),
        "initialization": "fresh_lora_from_pinned_base",
        "vision_policy": "frozen",
        "optimizer_steps_executed": 0,
        "language_sampling": {"en": 0.6, "ko": 0.2, "ja": 0.2},
        "training": {
            "lr": 0.0001,
            "head_lr": 0.0005,
            "reasoning_ce_weight": 0.3,
            "proposal_ce_weight": 0.2,
            "bf16": True,
            "grad_checkpointing": False,
            "micro_batch_tokens": 8192,
            "questions_per_step": 64,
            "ce_chunk_tokens": 128,
            "image_batch_rows": 4,
            "image_feature_cache_bytes": 128 * 1024 * 1024,
            "prune_supervised_positions": True,
            "seed": 20261001,
        },
        "prepared_cache_bytes": 256 * 1024 * 1024,
        "completion_target_seconds": 14400,
        "execution_requires": [
            "explicit optimizer step count",
            "explicit GPU time budget",
            "pinned pretrained weights available",
            "GPU forward/backward preflight without optimizer update",
        ],
        "promotion_requires": [
            "independent natural EN/KO/JA regression",
            "image perception vs reasoning paired evaluation",
            "candidate semantic and coverage evaluation",
            "modality/partition-scoped calibration",
            "dev-only router selection",
            "single final independent test",
        ],
    }
    (root / "training_config.json").write_bytes(canonical(recipe) + b"\n")
    (root / "model_preflight.json").write_bytes(canonical(audit) + b"\n")
    files = [*(f"{s}.jsonl" for s in SPLITS), "training_config.json", "model_preflight.json"]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    manifest = {
        "version": VERSION,
        "code_revision": revision,
        "source_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[1])).replace("\\", "/"): sha256(
                p.read_bytes()
            )
            for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
        },
        "counts": counts,
        "files": {name: sha256((root / name).read_bytes()) for name in files},
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "pillow", "torchvision", "safetensors")
        },
        "status": "data_and_meta_preflight_complete_no_training",
        "optimizer_steps_executed": 0,
        "data_scope": (
            "authored mechanics plus pinned human-labelled natural rehearsal; quality unmeasured"
            if natural_data
            else "repository-authored mechanics curriculum; not natural-data performance evidence"
        ),
    }
    (root / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    validate_bundle(root)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="config source only; experimental adapter weights are not reused",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow-metadata-downloads",
        action="store_true",
        help="allow pinned config/tokenizer/processor downloads; never pretrained weights",
    )
    parser.add_argument("--per-type", type=int, default=32)
    parser.add_argument("--image-cases", type=int, default=32)
    parser.add_argument("--candidate-cases", type=int, default=32)
    parser.add_argument(
        "--natural-data", action="store_true", help="include pinned cached human-labelled sources"
    )
    parser.add_argument(
        "--reserved-evaluation",
        help="previous evaluation-only natural JSON; never used for training",
    )
    args = parser.parse_args(argv)
    cfg = replace(load_config(args.checkpoint), name="ayaka-v2-pretraining", version=2)
    manifest = prepare_bundle(
        args.out,
        cfg,
        offline=not args.allow_metadata_downloads,
        per_type=args.per_type,
        image_cases=args.image_cases,
        candidate_cases=args.candidate_cases,
        natural_data=args.natural_data,
        reserved_evaluation=args.reserved_evaluation,
    )
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve()),
                "status": manifest["status"],
                "counts": manifest["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
