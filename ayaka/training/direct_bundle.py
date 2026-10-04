"""Immutable direct-student preparation and audit, entirely CPU/offline by default.

No optimizer, model weights or paid execution is available through this CLI.
Prepared tokens are regenerated during audit rather than trusted from disk.
"""

from __future__ import annotations

import argparse
import json
import math
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
from .direct_corpus_plan import (
    MARKER,
    plan_sha256,
    preparation_binding,
    prepared_groups_sha256,
    recipe_plan,
    split_summary,
    validate_contract,
    validate_runtime_replay,
    whole_epochs,
)
from .direct_distillation import prepare_direct_distillation
from .direct_holdout import (
    DEVELOPMENT_SPLITS,
    holdout_destination,
    make_commitment,
    validate_commitment,
    write_holdout,
)
from .direct_preflight import inspect_direct_model
from .native_metadata import inspect_metadata, verify_metadata
from .optimization import OptimizationConfig
from .prepare_v2 import audit_splits, canonical, sha256
from .swift_direct import encode_direct_sample, input_serving_recipe, normalize_input_encoding
from .tokenizer_identity import backend_fingerprints, scoped_tokenizer_preparation
from .workload import describe_rows, finite_workload, scheduled_batches

VERSION = "ayaka-direct-bundle-6"
STATUS = "cpu_prepared_no_model_or_optimizer_execution"
FILES = {f"{split}.jsonl" for split in DEVELOPMENT_SPLITS} | {
    "test_commitment.json",
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


def local_tokenizer(cfg, *, allow_tiny=False, native_path=None, expected_metadata=None):
    _model_policy(cfg, allow_tiny=allow_tiny)
    metadata, root = inspect_metadata(cfg.backbone, cfg.backbone_revision, path=native_path)
    if expected_metadata is not None and metadata != expected_metadata:
        raise ValueError("native configuration/tokenizer assets changed before tokenizer loading")
    if root is None:
        return ToyTokenizer()
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    verify_metadata(metadata, cfg.backbone, cfg.backbone_revision, path=root)
    return HFTokenizer(hf, cfg.backbone)


def bound_native_root(cfg, recipe, *, native_path=None):
    """Validate metadata before native config/tokenizer loaders can select assets."""
    if recipe.get("version") != VERSION or "native_metadata" not in recipe:
        raise ValueError("v6 direct bundle requires bound native metadata; prepare a new bundle")
    return verify_metadata(
        recipe["native_metadata"], cfg.backbone, cfg.backbone_revision, path=native_path
    )


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
    if not isinstance(tok, HFTokenizer) or tok.name != cfg.backbone:
        raise ValueError("tokenizer must match the declared model")
    backend = getattr(tok.hf, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("exact fast-tokenizer serialization is required")
    return fingerprint(
        {
            "backend_sha256": backend_fingerprints(tok)["raw_sha256"],
            "chat_template": tok.hf.chat_template,
            "special_tokens": tok.hf.special_tokens_map,
            "decision_chat": tok.decision_chat,
            "bos": tok.bos_id,
            "pad": tok.pad_id,
            "model_revision": cfg.backbone_revision,
        }
    )


def _gold_verifier(splits, natural_registry, *, allow_tiny):
    natural = any(
        s.metadata.get("data_kind") == "natural" for samples in splits.values() for s in samples
    )
    if natural:
        from ..data.direct_natural import NaturalGoldRegistry

        natural_registry = natural_registry or NaturalGoldRegistry()
        if not isinstance(natural_registry, NaturalGoldRegistry):
            raise ValueError("natural gold requires the pinned raw-source registry")
        if not natural_registry.local_files_verified and not allow_tiny:
            raise ValueError("injected natural raw fixtures are restricted to tiny CPU mechanics")
        natural_registry.verify_files()
    elif natural_registry is not None:
        raise ValueError("unused natural registry does not belong to an authored-only bundle")

    def verify(sample, question):
        return (
            natural_registry(sample, question)
            if sample.metadata.get("data_kind") == "natural"
            else verify_authored_gold(sample, question)
        )

    verify.verify_files = natural_registry.verify_files if natural else lambda: None
    verify.natural_registry = natural_registry

    return verify, natural_registry.binding if natural else None


@scoped_tokenizer_preparation
def _context_audit(splits, tok, cfg, architecture, verify_gold, input_encoding=None):
    contexts = {}
    verify_gold.verify_files()
    vocab_size = architecture["native_output_shape"][0]
    for split, samples in splits.items():
        rendered_rows = []
        for sample in samples:
            source = sample.metadata.get("source_example_id")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("explicit source IDs are required for every reserved split")
            limit = (
                cfg.max_seq_len if split == "train" else max(cfg.max_seq_len, cfg.serve_max_seq_len)
            )
            encoded = encode_direct_sample(
                sample, tok, cfg, input_encoding=input_encoding, context_limit=limit
            )
            for q, item in zip(sample.questions, encoded, strict=True):
                gold = verify_gold(sample, q)
                if gold != {c.id: q.target_distribution.get(c.id, 0) for c in q.candidates}:
                    raise ValueError(
                        "independently verified reserved gold disagrees with stored targets"
                    )
                rendered = item.enc.rendered
                tokens = item.enc.prefix_ids + rendered.suffix_ids
                if len(tokens) > limit:
                    raise ValueError(
                        "full direct evaluation/training input overflows; refuse truncation"
                    )
                if rendered.label_ids is None:
                    raise ValueError("this native LM training arm requires labelled candidate sets")
                if any(
                    type(t) is not int or not 0 <= t < vocab_size
                    for t in tokens + rendered.label_ids
                ):
                    raise ValueError(
                        "actual input or label tokens exceed the native output vocabulary"
                    )
                rendered_rows.append(
                    {
                        "id": source + "/" + q.id,
                        "tokens": len(tokens),
                        "input_sha256": fingerprint(tokens),
                    }
                )
        contexts[split] = {
            "max_tokens": max(r["tokens"] for r in rendered_rows),
            "rendered_rows_sha256": fingerprint(rendered_rows),
            "questions": len(rendered_rows),
        }
    verify_gold.verify_files()
    return contexts


@scoped_tokenizer_preparation
def _prepare(
    splits,
    tok,
    cfg,
    teacher_reads,
    weights,
    schedule,
    architecture,
    *,
    natural_registry=None,
    allow_tiny=False,
    holdout_commitment,
    input_encoding=None,
    corpus_plan=None,
):
    # Verify development gold and opaque test commitments. Original holdout
    # gold/context was checked during preparation; the trainer cannot open it.
    if (
        type(weights.base_replay) not in (int, float)
        or not math.isfinite(weights.base_replay)
        or weights.base_replay < 0
    ):
        raise ValueError("frozen-base replay weight must be finite and nonnegative")
    if weights.base_replay and not any(
        s.metadata.get("data_kind") == "natural" for s in splits["train"]
    ):
        raise ValueError("frozen-base replay requires natural train rehearsal")
    audit_splits(splits, development_only=True)
    validate_commitment(holdout_commitment, splits)
    verify_gold, gold_sources = _gold_verifier(
        splits, natural_registry, allow_tiny=allow_tiny and cfg.backbone == "tiny"
    )
    contexts = _context_audit(splits, tok, cfg, architecture, verify_gold, input_encoding)
    contexts["test"] = holdout_commitment["context_audit"]
    items, report = prepare_direct_distillation(
        splits,
        tok,
        cfg,
        teacher_reads,
        verify_gold,
        weights=weights,
        development_only=True,
        input_encoding=input_encoding,
    )
    report["split_counts"]["test"] = holdout_commitment["counts"]
    report["split_sha256"]["test"] = holdout_commitment["split_sha256"]
    inventory, groups, offset = [], [], 0
    for sample in splits["train"]:
        group = items[offset : offset + len(sample.questions)]
        inventory.append(describe_rows(sample, group))
        groups.append(group)
        offset += len(group)
    report.update(
        verifier=VERIFIER_VERSION,
        gold_sources=gold_sources,
        context_audit=contexts,
        workload=finite_workload(inventory, **schedule),
    )
    contract = validate_contract(
        corpus_plan, splits, holdout_commitment, inventory, groups, schedule
    )
    if contract is not None:
        report["corpus_contract"] = contract
    verify_gold.verify_files()
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
    optimizations=None,
    natural_registry=None,
    holdout_out=None,
    input_encoding=None,
    native_path=None,
    corpus_plan=None,
    preparation_guard=None,
):
    """Validate all data/tokens/settings before creating a fresh output directory."""
    root = Path(out)
    if root.exists():
        raise ValueError("direct bundle output must be a new directory")
    holdout_root = holdout_destination(root, holdout_out)
    policy = _model_policy(cfg, allow_tiny=allow_tiny)
    if type(allow_tiny) is not bool:
        raise ValueError("mechanics-only mode must be boolean")
    weights = weights or LossWeights(gold_nll_with_teacher=True, distill=0.2, pointer_aux=0)
    if (
        type(weights.base_replay) not in (int, float)
        or not math.isfinite(weights.base_replay)
        or weights.base_replay < 0
    ):
        raise ValueError("frozen-base replay weight must be finite and nonnegative")
    if weights.base_replay and not any(
        s.metadata.get("data_kind") == "natural" for s in splits["train"]
    ):
        raise ValueError("frozen-base replay requires natural train rehearsal")
    if weights.pointer_aux != 0:
        raise ValueError("native direct-student arm does not train a pointer auxiliary loss")
    if type(seed) is not int or seed < 0:
        raise ValueError("schedule seed must be a nonnegative integer")
    schedule = {"steps": steps, "rows_per_step": rows_per_step, "seed": seed}
    _validate_schedule(schedule)
    tokenizer_sha256 = _tokenizer_identity(tok, cfg)
    input_encoding = normalize_input_encoding(input_encoding)
    input_recipe = input_serving_recipe(tok, input_encoding)
    optimizations = optimizations or OptimizationConfig()
    metadata, native_root = inspect_metadata(cfg.backbone, cfg.backbone_revision, path=native_path)
    architecture = inspect_direct_model(
        cfg,
        official_weight_elements=policy.get("backbone_weight_elements"),
        optimizations=optimizations,
        native_path=native_root,
        expected_config={
            "kind": "config_file",
            "sha256": metadata["files"]["config.json"]["sha256"],
        }
        if metadata is not None
        else None,
    )
    counts = audit_splits(splits)
    if corpus_plan is None and any(
        MARKER in sample.metadata for samples in splits.values() for sample in samples
    ):
        raise ValueError("partial corpus plan markers require a complete planned contract")
    verify_test_gold, _ = _gold_verifier(
        splits, natural_registry, allow_tiny=allow_tiny and cfg.backbone == "tiny"
    )
    natural_registry = verify_test_gold.natural_registry
    preparation_binding(
        corpus_plan,
        cfg,
        tokenizer_sha256,
        metadata,
        input_encoding,
        natural_registry.binding if natural_registry is not None else None,
    )
    test_context = _context_audit(
        {"test": splits["test"]}, tok, cfg, architecture, verify_test_gold, input_encoding
    )["test"]
    commitment, holdout_raw = make_commitment(
        splits["test"],
        context=test_context,
        counts=counts["test"],
        corpus_contract={
            "plan_sha256": plan_sha256(corpus_plan),
            "summary": split_summary(splits["test"], plan_sha256(corpus_plan)),
        }
        if corpus_plan is not None
        else None,
    )
    development = {split: splits[split] for split in DEVELOPMENT_SPLITS}
    development_registry = (
        natural_registry
        if any(
            s.metadata.get("data_kind") == "natural" for rows in development.values() for s in rows
        )
        else None
    )
    items, report, _, _ = _prepare(
        development,
        tok,
        cfg,
        teacher_reads,
        weights,
        schedule,
        architecture,
        natural_registry=development_registry,
        allow_tiny=allow_tiny,
        holdout_commitment=commitment,
        input_encoding=input_encoding,
        corpus_plan=corpus_plan,
    )
    recipe = {
        "version": VERSION,
        "model": asdict(cfg),
        "loss_weights": asdict(weights),
        "schedule": schedule,
        "tokenizer_sha256": tokenizer_sha256,
        "input_encoding": input_encoding,
        "input_recipe": input_recipe,
        "allow_tiny": allow_tiny,
        "verifier": VERIFIER_VERSION,
        "gold_sources": report["gold_sources"],
        "model_policy": policy,
        "native_architecture": architecture,
        "native_metadata": metadata,
        "optimizations": asdict(optimizations),
        "inference_mode": "off",
        "reasoning_training_tokens": 0,
        "initialization": "fresh_lora_from_pinned_native_base",
    }
    if corpus_plan is not None:
        recipe.update(
            corpus_plan=corpus_plan,
            corpus_plan_sha256=plan_sha256(corpus_plan),
            corpus_contract=report["corpus_contract"],
        )
    payloads = {
        f"{split}.jsonl": b"\n".join(canonical(s.to_json()) for s in splits[split]) + b"\n"
        for split in DEVELOPMENT_SPLITS
    }
    payloads.update(
        {
            "teacher_reads.json": canonical(teacher_reads) + b"\n",
            "recipe.json": canonical(recipe) + b"\n",
            "preparation.json": canonical(report) + b"\n",
            "train_items.jsonl": _item_bytes(items),
            "test_commitment.json": canonical(commitment) + b"\n",
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
        "test_storage": "separate_holdout_directory_with_opaque_commitments",
        "data_scope": "authored mechanics plus pinned human rehearsal when supplied; not JevBench quality evidence",
        "pending": [
            "independent review",
            "actual pinned native weights",
            "GPU parity and throughput",
            "whole-workload credit admission",
            "trained calibration/dev/frozen independent test",
        ],
    }
    verify_metadata(metadata, cfg.backbone, cfg.backbone_revision, path=native_root)
    verify_test_gold.verify_files()
    if preparation_guard is not None:
        preparation_guard()
    root.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        (root / name).write_bytes(raw)
    manifest_raw = canonical(manifest) + b"\n"
    (root / "manifest.json").write_bytes(manifest_raw)
    write_holdout(holdout_root, holdout_raw, commitment, sha256(manifest_raw))
    return manifest


def audit_bundle(
    path,
    tok=None,
    *,
    allow_tiny=False,
    natural_registry=None,
    expected_manifest_sha256=None,
    native_path=None,
):
    """Recompute corpus gold, original tokens, teacher filtering and the entire schedule."""
    root = Path(path)
    manifest_raw = (root / "manifest.json").read_bytes()
    if expected_manifest_sha256 is not None and (
        not isinstance(expected_manifest_sha256, str)
        or len(expected_manifest_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_manifest_sha256)
        or sha256(manifest_raw) != expected_manifest_sha256
    ):
        raise ValueError("bundle manifest differs from the externally pinned preparation digest")
    manifest = json.loads(manifest_raw)
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
    if (root / "test.jsonl").exists():
        raise ValueError("development bundle must not contain original test inputs")
    for name, digest in manifest["files"].items():
        if sha256((root / name).read_bytes()) != digest:
            raise ValueError(f"direct bundle checksum mismatch: {name}")
    if manifest.get("source_sha256") != _source_hashes():
        raise ValueError("training source snapshot changed; rebuild the CPU bundle")
    recipe = json.loads((root / "recipe.json").read_bytes())
    corpus_plan = recipe_plan(recipe)
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
    native_root = bound_native_root(cfg, recipe, native_path=native_path)
    architecture = inspect_direct_model(
        cfg,
        official_weight_elements=recipe["model_policy"].get("backbone_weight_elements"),
        optimizations=OptimizationConfig(**recipe["optimizations"]),
        native_path=native_root,
        expected_config=recipe["native_architecture"]["native_config"],
    )
    if architecture != recipe.get("native_architecture"):
        raise ValueError("actual native architecture or LoRA placements changed")
    tok = (
        tok
        if tok is not None
        else local_tokenizer(
            cfg,
            allow_tiny=allow_tiny,
            native_path=native_root,
            expected_metadata=recipe["native_metadata"],
        )
    )
    if _tokenizer_identity(tok, cfg) != recipe.get("tokenizer_sha256"):
        raise ValueError("tokenizer or chat template changed")
    if "input_encoding" not in recipe or "input_recipe" not in recipe:
        raise ValueError("v6 direct bundle requires an explicit input encoder and serving recipe")
    input_encoding = normalize_input_encoding(recipe["input_encoding"])
    if (
        input_encoding != recipe["input_encoding"]
        or input_serving_recipe(tok, input_encoding) != recipe["input_recipe"]
    ):
        raise ValueError("direct input encoder or actual serving recipe changed")
    weights = LossWeights(**recipe["loss_weights"])
    if weights.pointer_aux != 0:
        raise ValueError("native direct-student arm does not train a pointer auxiliary loss")
    splits = {
        split: [
            Sample.from_json(json.loads(line))
            for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for split in DEVELOPMENT_SPLITS
    }
    teachers = json.loads((root / "teacher_reads.json").read_bytes())
    verify_gold, _ = _gold_verifier(
        splits, natural_registry, allow_tiny=allow_tiny and cfg.backbone == "tiny"
    )
    natural_registry = verify_gold.natural_registry
    items, report, inventory, groups = _prepare(
        splits,
        tok,
        cfg,
        teachers,
        weights,
        recipe["schedule"],
        architecture,
        natural_registry=natural_registry,
        allow_tiny=allow_tiny,
        holdout_commitment=json.loads((root / "test_commitment.json").read_bytes()),
        input_encoding=input_encoding,
        corpus_plan=corpus_plan,
    )
    if report["gold_sources"] != recipe.get("gold_sources"):
        raise ValueError("raw human source file binding changed; prepare a new bundle")
    if corpus_plan is not None and report["corpus_contract"] != recipe["corpus_contract"]:
        raise ValueError("recipe corpus contract differs from regenerated preparation")
    if (
        canonical(report) + b"\n" != (root / "preparation.json").read_bytes()
        or _item_bytes(items) != (root / "train_items.jsonl").read_bytes()
    ):
        raise ValueError("regenerated direct items or workload disagree with saved preparation")
    verify_metadata(
        recipe["native_metadata"], cfg.backbone, cfg.backbone_revision, path=native_root
    )
    verify_gold.verify_files()
    return manifest, recipe, items, inventory, groups


def training_batches(recipe, inventory, groups, *, start_step=0):
    """Replay a fixed schedule; resuming skips whole steps without changing its seed."""
    schedule = recipe["schedule"]
    _validate_schedule(schedule)
    plan = recipe_plan(recipe)
    if plan is not None:
        contract = recipe["corpus_contract"]
        if (
            fingerprint(inventory) != contract["inventory_sha256"]
            or whole_epochs(plan, inventory, schedule) != contract["whole_epochs"]
            or len(groups) != len(inventory)
            or any(
                len(group) != len(row["rows"]) for group, row in zip(groups, inventory, strict=True)
            )
            or prepared_groups_sha256(groups) != contract["prepared_groups_sha256"]
        ):
            raise ValueError("actual corpus training groups differ from the frozen complete epochs")
        validate_runtime_replay(recipe, inventory, groups)
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
    prepare.add_argument(
        "--holdout-out", type=Path, help="separate storage; defaults to sibling OUT-holdout"
    )
    prepare.add_argument("--steps", required=True, type=int)
    prepare.add_argument("--rows-per-step", required=True, type=int)
    prepare.add_argument("--seed", default=20261004, type=int)
    prepare.add_argument("--distill-weight", default=0.2, type=float)
    prepare.add_argument("--base-replay-weight", default=0, type=float)
    prepare.add_argument("--mechanics-only", action="store_true")
    prepare.add_argument(
        "--native-path", type=Path, help="complete offline native model/tokenizer directory"
    )
    prepare.add_argument(
        "--input-encoder", choices=("ayaka_segmented", "swift_canonical"), default="ayaka_segmented"
    )
    prepare.add_argument("--prompt-variant", choices=("min", "cygnet", "rules", "labeled"))
    prepare.add_argument("--state-format", choices=("pretty", "compact"))
    prepare.add_argument(
        "--attention", choices=("native", "sdpa", "flash_attention_2"), default="native"
    )
    prepare.add_argument("--liger", action="store_true")
    audit = commands.add_parser("audit")
    audit.add_argument("--bundle", required=True, type=Path)
    audit.add_argument("--mechanics-only", action="store_true")
    audit.add_argument("--expected-manifest-sha256")
    audit.add_argument("--native-path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        encoding = {"encoder": args.input_encoder}
        for field in ("prompt_variant", "state_format"):
            if getattr(args, field) is not None:
                encoding[field] = getattr(args, field)
        encoding = normalize_input_encoding(encoding)
        cfg = ElectraConfig(**json.loads(args.config.read_bytes()))
        tok = local_tokenizer(cfg, allow_tiny=args.mechanics_only, native_path=args.native_path)
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
                gold_nll_with_teacher=True,
                distill=args.distill_weight,
                pointer_aux=0,
                base_replay=args.base_replay_weight,
            ),
            allow_tiny=args.mechanics_only,
            optimizations=OptimizationConfig(attention=args.attention, liger=args.liger),
            holdout_out=args.holdout_out,
            input_encoding=encoding,
            native_path=args.native_path,
        )
    else:
        manifest, _, _, _, _ = audit_bundle(
            args.bundle,
            allow_tiny=args.mechanics_only,
            expected_manifest_sha256=args.expected_manifest_sha256,
            native_path=args.native_path,
        )
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
