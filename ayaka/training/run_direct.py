"""Offline native direct-student profile/train/evaluate/export pipeline.

Audit is CPU-only. Profile/train require --execute or an explicitly random tiny
CPU mechanics run. This command never allocates a cloud instance or authorizes
paid execution. All outputs remain experimental and non-promotable.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ..checkpoint import apply_lora, load_checkpoint, save_checkpoint
from ..config import ElectraConfig
from ..data.reasoning_v2 import SPLITS
from ..data.schema import Sample
from ..eval.read_artifact import fingerprint
from ..eval.v2 import summarize, typed_row
from ..losses import LossWeights
from ..model.electra import ElectraDecisionModel
from ..primitives import QuestionSpec
from .batching import _noul_canonical, sample_to_items
from .calibrate import apply_temperatures, fit_temperatures
from .direct_budget import STAGES, admit_workflow
from .direct_budget import VERSION as BUDGET_VERSION
from .direct_bundle import audit_bundle, local_tokenizer, training_batches
from .direct_state import (
    load_training_state,
    save_training_state,
    train_fixed_schedule,
    training_binding,
    validate_training_config,
)
from .frozen_replay import attach_base_replay
from .native_snapshot import verify_snapshot
from .optimization import OptimizationConfig, optimize_and_verify
from .prepare_v2 import canonical
from .throughput import completion_plan, profile_overheads, profile_production
from .trainer import TrainConfig, Trainer
from .workload import stress_indices


def training_config(recipe, values=None):
    values = dict(values or {})
    if "loss_weights" in values:
        values["loss_weights"] = LossWeights(**values["loss_weights"])
    defaults = {
        "steps": recipe["schedule"]["steps"],
        "questions_per_step": recipe["schedule"]["rows_per_step"],
        "seed": recipe["schedule"]["seed"],
        "loss_weights": LossWeights(**recipe["loss_weights"]),
        "reasoning_ce_weight": 0,
        "proposal_ce_weight": 0,
    }
    cfg = TrainConfig(**{**defaults, **values})
    validate_training_config(recipe, cfg)
    return cfg


def mechanics_budget():
    return {
        "version": BUDGET_VERSION,
        "hourly_usd": 0,
        "prepaid_usd": 0,
        "other_reserved_usd": 0,
        "billing_quantum_seconds": 1,
        "safety_factor": 1.25,
        "stages": {
            stage: {"seconds": 0, "basis": "random CPU mechanics; no paid allocation"}
            for stage in STAGES
        },
    }


def read_splits(root):
    return {
        split: [
            Sample.from_json(json.loads(line))
            for line in (Path(root) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for split in SPLITS
    }


def evaluate_direct(trainer, samples, split):
    """Serial, isolated original-input reads. No gradients or trace generation."""
    rows = []
    for sample in samples:
        if sample.metadata["split"] != split:
            raise ValueError("evaluation samples must retain the reserved split identity")
        for original in sample.questions:
            q = _noul_canonical(original)
            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            start = time.perf_counter()
            metadata = {
                key: sample.metadata[key]
                for key in ("source_example_id", "task_family", "language")
                if key in sample.metadata
            }
            item = sample_to_items(
                Sample(sample.state, [q], metadata), trainer.tok, trainer.model.cfg
            )[0]
            probs = trainer.predict([item])[0]
            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            latency = time.perf_counter() - start
            spec = QuestionSpec(
                q.type,
                q.instruction,
                [c.description for c in q.candidates],
                [c.ordinal for c in q.candidates] if q.type == "score" else None,
            )
            rows.append(
                {
                    **typed_row(spec, probs, item.target),
                    "id": sample.metadata["source_example_id"] + "/" + q.id,
                    "cluster_id": sample.metadata["source_lineage"],
                    "split": split,
                    "type": q.type,
                    "family": sample.metadata.get("task_family", "unknown"),
                    "language": sample.metadata["language"],
                    "tier": sample.metadata.get("tier", "standard"),
                    "candidate_ids": [c.id for c in q.candidates],
                    "probs": probs,
                    "target": item.target,
                    "tokens": item.length,
                    "latency_s": latency,
                    "reasoning_tokens": 0,
                    "route": "direct",
                }
            )
    return {
        "rows": rows,
        "summary": summarize(rows),
        "latency_scope": "serial HF one-question reads including tokenization; CUDA synchronized",
    }


def _write(root, name, value):
    with (root / name).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def run_pipeline(
    bundle,
    out,
    *,
    action="profile",
    device="cpu",
    execute=False,
    mechanics_only=False,
    budget=None,
    training=None,
    snapshot_record=None,
    snapshot_path=None,
    checkpoint_every=100,
    resume=None,
    paid_elapsed_seconds=None,
    expected_bundle_sha256=None,
    saved_base_reads=None,
):
    started = time.monotonic()
    if action not in {"audit", "profile", "train"}:
        raise ValueError("unknown direct pipeline action")
    manifest, recipe, _, inventory, groups = audit_bundle(
        bundle, allow_tiny=mechanics_only, expected_manifest_sha256=expected_bundle_sha256
    )
    if action == "audit":
        return {
            "status": "audited_cpu_only",
            "bundle_sha256": fingerprint(manifest),
            "optimizer_steps": 0,
        }
    cfg = ElectraConfig(**recipe["model"])
    tcfg = training_config(recipe, training)
    dev = torch.device(device)
    if mechanics_only:
        if cfg.backbone != "tiny" or dev.type != "cpu":
            raise ValueError("mechanics execution requires the random tiny model on CPU")
        budget = budget or mechanics_budget()
        paid_elapsed_seconds = paid_elapsed_seconds or 0
    elif not execute or dev.type != "cuda" or cfg.backbone == "tiny":
        raise ValueError("native production profile/train requires explicit --execute and CUDA")
    elif paid_elapsed_seconds is None:
        raise ValueError("include all already billed setup/download/teacher time explicitly")
    elif expected_bundle_sha256 is None:
        raise ValueError("native execution requires an externally pinned bundle manifest digest")
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint interval must be positive")
    if action == "profile" and resume is not None:
        raise ValueError("zero-update profiling cannot resume optimizer state")
    admitted = admit_workflow(
        budget, elapsed_seconds=paid_elapsed_seconds + time.monotonic() - started
    )
    if not mechanics_only and (
        budget["hourly_usd"] <= 0
        or budget["prepaid_usd"] <= 0
        or any(budget["stages"][stage]["seconds"] <= 0 for stage in ("teardown", "recovery"))
    ):
        raise ValueError(
            "native execution requires explicit positive credit/rate and cleanup reserves"
        )
    if not admitted["fits"]:
        raise ValueError(
            "whole setup/train/test/download/teardown workflow exceeds prepaid credit before weight loading"
        )
    root = Path(out)
    if root.exists():
        raise ValueError("direct execution output must be a fresh directory")
    tok = local_tokenizer(cfg, allow_tiny=mechanics_only)
    splits = read_splits(bundle)
    if mechanics_only:
        native_sha, native_root = fingerprint({"tiny_initial_seed": 0, "dtype": "float32"}), None
    else:
        if not isinstance(snapshot_record, dict) or (
            snapshot_record.get("repo"),
            snapshot_record.get("revision"),
        ) != (cfg.backbone, cfg.backbone_revision):
            raise ValueError("native snapshot must match the bundle's immutable model")
        native_root = verify_snapshot(snapshot_record, path=snapshot_path)
        native_sha = snapshot_record["snapshot_sha256"]
    binding = training_binding(manifest, recipe, tcfg, native_weights_sha256=native_sha)
    binding["external_bundle_manifest_sha256"] = expected_bundle_sha256
    torch.manual_seed(tcfg.seed)
    model = ElectraDecisionModel.from_config(
        cfg,
        dtype=torch.float32 if mechanics_only else torch.bfloat16,
        device=dev,
        backbone_path=str(native_root) if native_root else None,
        local_files_only=True,
        strict_loading=True,
    )
    model.backbone.requires_grad_(False)
    apply_lora(model)
    trainer = Trainer(model, tok, tcfg, dev)
    base_reads = None
    if tcfg.loss_weights.base_replay:
        if resume is not None and saved_base_reads is None:
            raise ValueError("optimizer resume requires the original frozen-base reads")
        base_reads = attach_base_replay(
            trainer, splits["train"], groups, native_sha, saved=saved_base_reads
        )
    elif saved_base_reads is not None:
        raise ValueError("saved frozen-base reads require a positive replay coefficient")
    binding["frozen_base_reads_sha256"] = fingerprint(base_reads)
    probes = [item for index in stress_indices(inventory) for item in groups[index]]
    application = optimize_and_verify(
        trainer, probes, OptimizationConfig(**recipe["optimizations"])
    )
    root.mkdir(parents=True, exist_ok=False)
    _write(root, "binding.json", binding)
    _write(root, "initial_admission.json", admitted)
    _write(root, "kernel_parity.json", application.report)
    _write(root, "training_config.json", asdict(tcfg))
    if base_reads is not None:
        _write(root, "frozen_base_reads.json", base_reads)
    by_id = {
        sample.metadata["source_example_id"]: group
        for sample, group in zip(splits["train"], groups, strict=True)
    }
    profile = profile_production(
        trainer,
        itertools.cycle(training_batches(recipe, inventory, groups)),
        splits["train"],
        inventory,
        lambda sample: by_id[sample.metadata["source_example_id"]],
        steps=tcfg.steps,
        seed=tcfg.seed,
    )
    overheads = profile_overheads(trainer, root / "io_profile")
    # This train-only component includes all steps and checkpoint IO. The outer
    # workflow margin is applied once by admit_workflow below, not twice.
    completion = completion_plan(
        profile,
        tcfg.steps,
        admitted["forecast_total_seconds"],
        overheads=overheads,
        checkpoint_every=checkpoint_every,
        safety_factor=1,
    )
    measured = admit_workflow(
        budget,
        measured_training_seconds=completion["estimated_remaining_seconds"],
        elapsed_seconds=paid_elapsed_seconds + time.monotonic() - started,
        completed=(
            "environment_setup",
            "weight_download",
            "teacher_collection",
            "model_load",
            "preflight",
        ),
    )
    _write(root, "throughput.json", profile)
    _write(root, "overheads.json", overheads)
    _write(root, "completion_plan.json", {"training": completion, "workflow": measured})
    if action == "profile":
        application.rollback()
        return {
            "status": "profiled_zero_updates",
            "optimizer_steps": 0,
            "admission": measured,
            "disk_fits": completion["disk_fits"],
            "promotable": False,
        }
    if not measured["fits"] or not completion["disk_fits"]:
        application.rollback()
        raise ValueError("measured complete workflow does not fit; no optimizer updates executed")
    if resume is not None:
        load_training_state(trainer, resume, binding)

    def save_progress(step, record):
        if step == 1 or step % checkpoint_every == 0 or step == tcfg.steps:
            save_training_state(trainer, root / f"state-{step:08d}", binding)

    history = train_fixed_schedule(trainer, recipe, inventory, groups, on_step=save_progress)
    _write(root, "history.json", history)
    application.rollback()  # calibration/export use portable native inference kernels
    raw_calibration = evaluate_direct(trainer, splits["calibration"], "calibration")
    rows = raw_calibration["rows"]
    temperatures = fit_temperatures(
        [[math.log(max(p, 1e-12)) for p in row["probs"]] for row in rows],
        [row["target"] for row in rows],
        [row["type"] for row in rows],
        [row["tokens"] for row in rows],
        cfg.long_prompt_tokens,
    )
    apply_temperatures(model, temperatures)
    _write(root, "calibration_raw.json", raw_calibration)
    _write(
        root,
        "calibration.json",
        {"temperatures": temperatures, "split": "calibration", "binding": binding},
    )
    for split in ("dev", "test"):
        _write(root, f"{split}.json", evaluate_direct(trainer, splits[split], split))
    # No checkpoint/temperature/effort choice consults test. This diagnostic test
    # is not a release gate or an official sealed-inclusive benchmark result.
    probe_probs = trainer.predict(probes)
    meta = {
        "binding": binding,
        "optimizer_steps": trainer.step_i,
        "complete_schedule": trainer.step_i == tcfg.steps,
        "inference_mode": "off",
        "promotable": False,
        "mechanics_only": mechanics_only,
        "calibration_split": "calibration",
        "test_used_for_selection": False,
    }
    save_checkpoint(model, str(root / "checkpoint"), meta)
    trainer, model = None, None
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    reloaded = load_checkpoint(
        str(root / "checkpoint"),
        device=dev,
        dtype=torch.float32 if mechanics_only else torch.bfloat16,
        merge=False,
        backbone_path=str(native_root) if native_root else None,
        local_files_only=True,
        strict_loading=True,
    )
    verifier = Trainer(reloaded, tok, tcfg, dev)
    for expected, actual in zip(probe_probs, verifier.predict(probes), strict=True):
        torch.testing.assert_close(
            torch.tensor(actual), torch.tensor(expected), atol=1e-5, rtol=1e-4
        )
    _write(
        root,
        "export_verified.json",
        {
            **meta,
            "reload_probability_parity": True,
            "elapsed_pipeline_seconds": time.monotonic() - started,
            "remaining_external_stages": ["artifact_download", "teardown"],
            "official_composite": None,
        },
    )
    return {"status": "complete_experimental_direct_run", **meta, "reload_probability_parity": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "profile", "train"))
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mechanics-only", action="store_true")
    parser.add_argument("--budget", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--snapshot-record", type=Path)
    parser.add_argument("--snapshot-path", type=Path)
    parser.add_argument("--expected-bundle-sha256")
    parser.add_argument(
        "--base-reads", type=Path, help="original frozen native reads, required for replay resume"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", default=100, type=int)
    parser.add_argument(
        "--paid-elapsed-seconds",
        type=float,
        help="all already billed time including pod setup, downloads and teacher collection",
    )
    args = parser.parse_args(argv)
    if args.action != "audit" and args.out is None:
        parser.error("profile/train require --out")

    def read(path):
        return json.loads(path.read_bytes()) if path else None

    result = run_pipeline(
        args.bundle,
        args.out,
        action=args.action,
        device=args.device,
        execute=args.execute,
        mechanics_only=args.mechanics_only,
        budget=read(args.budget),
        training=read(args.training_config),
        snapshot_record=read(args.snapshot_record),
        snapshot_path=args.snapshot_path,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        paid_elapsed_seconds=args.paid_elapsed_seconds,
        expected_bundle_sha256=args.expected_bundle_sha256,
        saved_base_reads=read(args.base_reads),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
