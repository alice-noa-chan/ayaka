"""Immutable direct-student preparation and audit, entirely CPU/offline by default.

No optimizer, model weights or paid execution is available through this CLI.
Prepared tokens are regenerated during audit rather than trusted from disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from ..config import ElectraConfig
from ..data.direct_verification import VERSION as VERIFIER_VERSION
from ..data.direct_verification import verify_authored_gold
from ..data.reasoning_v2 import SPLITS
from ..data.schema import Sample
from ..eval.read_artifact import fingerprint
from ..losses import LossWeights
from ..tokenization import HFTokenizer, ToyTokenizer
from .direct_distillation import prepare_direct_distillation
from .prepare_v2 import canonical, sha256
from .workload import describe_rows, finite_workload, scheduled_batches

VERSION = "ayaka-direct-bundle-1"
STATUS = "cpu_prepared_no_model_or_optimizer_execution"
FILES = {f"{split}.jsonl" for split in SPLITS} | {
    "teacher_reads.json",
    "recipe.json",
    "preparation.json",
    "train_items.jsonl",
}


def _source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {
        p.relative_to(root).as_posix(): sha256(p.read_bytes()) for p in sorted(root.rglob("*.py"))
    }


def _json_safe(value):
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("prepared ordinals must be finite")
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _item_bytes(items):
    return b"\n".join(canonical(_json_safe(asdict(item))) for item in items) + b"\n"


def _model_policy(cfg, *, allow_tiny):
    if cfg.readout != "lm":
        raise ValueError("this direct-student arm requires the native LM readout")
    if cfg.backbone == "tiny":
        if not allow_tiny:
            raise ValueError("tiny random models require explicit mechanics-only mode")
        return {"scope": "random CPU mechanics only", "production_model_verified": False}
    revision = cfg.backbone_revision
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
    ):
        raise ValueError("direct preparation requires an immutable backbone revision")
    survey_path = Path(__file__).resolve().parents[2] / "docs/experiments/v2_candidates.json"
    survey = json.loads(survey_path.read_bytes())
    candidate = next(
        (
            c
            for c in survey["candidates"]
            if c["repo"] == cfg.backbone and c["revision"] == revision
        ),
        None,
    )
    if (
        candidate is None
        or candidate["license"] not in {"apache-2.0", "mit"}
        or candidate["support"] != "builtin"
        or candidate["total_parameters"] > 14_000_000_000
    ):
        raise ValueError("backbone must be pinned in the approved size/license/support survey")
    return {
        "scope": "pinned survey; actual architecture and adapter count still require preflight",
        "production_model_verified": False,
        "survey_sha256": sha256(survey_path.read_bytes()),
        "backbone_weight_elements": candidate["total_parameters"],
        "license": candidate["license"],
    }


def local_tokenizer(cfg, *, allow_tiny=False):
    _model_policy(cfg, allow_tiny=allow_tiny)
    if cfg.backbone == "tiny":
        return ToyTokenizer()
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(
        cfg.backbone, revision=cfg.backbone_revision, local_files_only=True, trust_remote_code=False
    )
    return HFTokenizer(hf, cfg.backbone)


def _tokenizer_identity(tok, cfg):
    if cfg.backbone == "tiny" and type(tok) is ToyTokenizer:
        return fingerprint(
            {
                "name": tok.name,
                "vocab_size": tok.vocab_size,
                "special": tok.SPECIAL,
                "bos": tok.bos_id,
                "pad": tok.pad_id,
            }
        )
    if cfg.backbone == "tiny" or not isinstance(tok, HFTokenizer) or tok.name != cfg.backbone:
        raise ValueError("tokenizer must match the declared model")
    backend = getattr(tok.hf, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("exact fast-tokenizer serialization is required")
    return fingerprint(
        {
            "backend_sha256": hashlib.sha256(backend.to_str().encode()).hexdigest(),
            "chat_template": tok.hf.chat_template,
            "special_tokens": tok.hf.special_tokens_map,
            "decision_chat": tok.decision_chat,
            "bos": tok.bos_id,
            "pad": tok.pad_id,
            "model_revision": cfg.backbone_revision,
        }
    )


def _prepare(splits, tok, cfg, teacher_reads, weights, schedule):
    # Verify every reserved split too; no holdout answer enters training or teacher selection.
    for samples in splits.values():
        for sample in samples:
            for q in sample.questions:
                gold = verify_authored_gold(sample, q)
                if gold != {c.id: q.target_distribution.get(c.id, 0) for c in q.candidates}:
                    raise ValueError(
                        "independently verified reserved gold disagrees with stored targets"
                    )
    items, report = prepare_direct_distillation(
        splits, tok, cfg, teacher_reads, verify_authored_gold, weights=weights
    )
    inventory, groups, offset = [], [], 0
    for sample in splits["train"]:
        group = items[offset : offset + len(sample.questions)]
        inventory.append(describe_rows(sample, group))
        groups.append(group)
        offset += len(group)
    report.update(verifier=VERIFIER_VERSION, workload=finite_workload(inventory, **schedule))
    return items, report, inventory, groups


def _validate_schedule(schedule):
    if not isinstance(schedule, dict) or set(schedule) != {"steps", "rows_per_step", "seed"}:
        raise ValueError("exact fixed steps, rows_per_step and seed are required")
    if (
        any(
            type(schedule[key]) is not int or schedule[key] < 1
            for key in ("steps", "rows_per_step")
        )
        or type(schedule["seed"]) is not int
        or schedule["seed"] < 0
    ):
        raise ValueError("fixed schedule counts must be positive integers and seed nonnegative")


def prepare_bundle(
    out,
    splits,
    tok,
    cfg,
    teacher_reads,
    *,
    steps,
    rows_per_step,
    seed=20261004,
    weights=None,
    allow_tiny=False,
):
    """Validate all data/tokens/settings before creating a fresh output directory."""
    root = Path(out)
    if root.exists():
        raise ValueError("direct bundle output must be a new directory")
    policy = _model_policy(cfg, allow_tiny=allow_tiny)
    if type(allow_tiny) is not bool:
        raise ValueError("mechanics-only mode must be boolean")
    weights = weights or LossWeights(gold_nll_with_teacher=True, distill=0.2, pointer_aux=0)
    if weights.pointer_aux != 0:
        raise ValueError("native direct-student arm does not train a pointer auxiliary loss")
    if type(seed) is not int or seed < 0:
        raise ValueError("schedule seed must be a nonnegative integer")
    schedule = {"steps": steps, "rows_per_step": rows_per_step, "seed": seed}
    _validate_schedule(schedule)
    tokenizer_sha256 = _tokenizer_identity(tok, cfg)
    items, report, _, _ = _prepare(splits, tok, cfg, teacher_reads, weights, schedule)
    recipe = {
        "version": VERSION,
        "model": asdict(cfg),
        "loss_weights": asdict(weights),
        "schedule": schedule,
        "tokenizer_sha256": tokenizer_sha256,
        "allow_tiny": allow_tiny,
        "verifier": VERIFIER_VERSION,
        "model_policy": policy,
        "inference_mode": "off",
        "reasoning_training_tokens": 0,
        "initialization": "fresh_lora_from_pinned_native_base",
    }
    payloads = {
        f"{split}.jsonl": b"\n".join(canonical(s.to_json()) for s in splits[split]) + b"\n"
        for split in SPLITS
    }
    payloads.update(
        {
            "teacher_reads.json": canonical(teacher_reads) + b"\n",
            "recipe.json": canonical(recipe) + b"\n",
            "preparation.json": canonical(report) + b"\n",
            "train_items.jsonl": _item_bytes(items),
        }
    )
    manifest = {
        "version": VERSION,
        "status": STATUS,
        "files": {name: sha256(raw) for name, raw in payloads.items()},
        "source_sha256": _source_hashes(),
        "promotable": False,
        "execution_attested": False,
        "optimizer_steps_executed": 0,
        "data_scope": "versioned authored mechanics; not natural-data or JevBench quality evidence",
        "pending": [
            "independent review",
            "actual native weights/adapter parameter count",
            "GPU parity and throughput",
            "whole-workload credit admission",
            "trained calibration/dev/frozen independent test",
        ],
    }
    root.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        (root / name).write_bytes(raw)
    (root / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    return manifest


def audit_bundle(path, tok=None, *, allow_tiny=False):
    """Recompute corpus gold, original tokens, teacher filtering and the entire schedule."""
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_bytes())
    if (
        manifest.get("version") != VERSION
        or manifest.get("status") != STATUS
        or manifest.get("promotable") is not False
        or manifest.get("execution_attested") is not False
        or type(manifest.get("optimizer_steps_executed")) is not int
        or manifest["optimizer_steps_executed"] != 0
    ):
        raise ValueError("unsupported or falsely promoted direct bundle")
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != FILES:
        raise ValueError("manifest must cover exactly the known direct bundle files")
    for name, digest in manifest["files"].items():
        if sha256((root / name).read_bytes()) != digest:
            raise ValueError(f"direct bundle checksum mismatch: {name}")
    if manifest.get("source_sha256") != _source_hashes():
        raise ValueError("training source snapshot changed; rebuild the CPU bundle")
    recipe = json.loads((root / "recipe.json").read_bytes())
    if (
        recipe.get("version") != VERSION
        or recipe.get("verifier") != VERIFIER_VERSION
        or recipe.get("inference_mode") != "off"
        or type(recipe.get("reasoning_training_tokens")) is not int
        or recipe["reasoning_training_tokens"] != 0
        or recipe.get("initialization") != "fresh_lora_from_pinned_native_base"
    ):
        raise ValueError("direct recipe must retain the original-input off objective")
    if type(recipe.get("allow_tiny")) is not bool or recipe["allow_tiny"] and not allow_tiny:
        raise ValueError("auditing a tiny bundle requires explicit mechanics-only mode")
    cfg = ElectraConfig(**recipe["model"])
    _validate_schedule(recipe.get("schedule"))
    if recipe.get("model_policy") != _model_policy(cfg, allow_tiny=allow_tiny):
        raise ValueError("declared model policy changed")
    tok = tok if tok is not None else local_tokenizer(cfg, allow_tiny=allow_tiny)
    if _tokenizer_identity(tok, cfg) != recipe.get("tokenizer_sha256"):
        raise ValueError("tokenizer or chat template changed")
    weights = LossWeights(**recipe["loss_weights"])
    if weights.pointer_aux != 0:
        raise ValueError("native direct-student arm does not train a pointer auxiliary loss")
    splits = {
        split: [
            Sample.from_json(json.loads(line))
            for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for split in SPLITS
    }
    teachers = json.loads((root / "teacher_reads.json").read_bytes())
    items, report, inventory, groups = _prepare(
        splits, tok, cfg, teachers, weights, recipe["schedule"]
    )
    if (
        canonical(report) + b"\n" != (root / "preparation.json").read_bytes()
        or _item_bytes(items) != (root / "train_items.jsonl").read_bytes()
    ):
        raise ValueError("regenerated direct items or workload disagree with saved preparation")
    return manifest, recipe, items, inventory, groups


def training_batches(recipe, inventory, groups, *, start_step=0):
    """Replay a fixed schedule; resuming skips whole steps without changing its seed."""
    schedule = recipe["schedule"]
    _validate_schedule(schedule)
    if type(start_step) is not int or not 0 <= start_step <= schedule["steps"]:
        raise ValueError("resume step must be inside the complete fixed schedule")
    for step, batch in enumerate(scheduled_batches(inventory, **schedule)):
        if step >= start_step:
            yield [groups[index][position] for index, position in batch]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--corpus", required=True, type=Path)
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--teacher-reads", type=Path)
    prepare.add_argument("--out", required=True, type=Path)
    prepare.add_argument("--steps", required=True, type=int)
    prepare.add_argument("--rows-per-step", required=True, type=int)
    prepare.add_argument("--seed", default=20261004, type=int)
    prepare.add_argument("--distill-weight", default=0.2, type=float)
    prepare.add_argument("--mechanics-only", action="store_true")
    audit = commands.add_parser("audit")
    audit.add_argument("--bundle", required=True, type=Path)
    audit.add_argument("--mechanics-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        cfg = ElectraConfig(**json.loads(args.config.read_bytes()))
        tok = local_tokenizer(cfg, allow_tiny=args.mechanics_only)
        splits = {
            split: [
                Sample.from_json(json.loads(line))
                for line in (args.corpus / f"{split}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            for split in SPLITS
        }
        teachers = json.loads(args.teacher_reads.read_bytes()) if args.teacher_reads else {}
        manifest = prepare_bundle(
            args.out,
            splits,
            tok,
            cfg,
            teachers,
            steps=args.steps,
            rows_per_step=args.rows_per_step,
            seed=args.seed,
            weights=LossWeights(
                gold_nll_with_teacher=True, distill=args.distill_weight, pointer_aux=0
            ),
            allow_tiny=args.mechanics_only,
        )
    else:
        manifest, _, _, _, _ = audit_bundle(args.bundle, allow_tiny=args.mechanics_only)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "optimizer_steps_executed": 0,
                "promotable": False,
                "paid_execution_started": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
