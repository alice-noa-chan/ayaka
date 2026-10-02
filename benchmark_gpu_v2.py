"""Matched native GPU backward comparison; never updates pretrained parameters."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from ayaka.config import ElectraConfig
from ayaka.training.prepare_v2 import canonical, prepared_items, sha256, validate_bundle
from ayaka.training.run_v2 import (
    fresh_image_backend,
    job_deadline,
    source_matches,
    trainable_digest,
)
from ayaka.training.throughput import profile_backward, profile_overheads
from ayaka.training.trainer import TrainConfig, Trainer
from ayaka.training.workload import finite_workload, profile_schedule

INITIAL_POLICY = {"micro_tokens": 4096, "micro_ckpt_tokens": 8192, "checkpoint_threshold": 1024}


def policy(trainer):
    return {
        "micro_tokens": trainer.micro_tokens,
        "micro_ckpt_tokens": trainer.micro_ckpt_tokens,
        "checkpoint_threshold": trainer.ckpt_threshold,
    }


def prepare_batch(samples, row_indices, prepare):
    prepared, rows = {}, []
    for index, position in row_indices:
        if index not in prepared:
            prepared[index] = prepare(samples[index])
        rows.append(prepared[index][position])
    return rows


def matched_profile(trainer, samples, inventory, prepare, out, *, steps=1200, seed=0, weights=None):
    """Cover a fixed full-plan subset, cold images, warmed kernels and lazy moments."""
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    selection = profile_schedule(
        inventory, steps, trainer.cfg.questions_per_step, seed, weights, evenly_spaced=8
    )
    workload = finite_workload(inventory, steps, trainer.cfg.questions_per_step, seed, weights)
    before = trainable_digest(trainer.model)
    result = {
        "status": "running",
        "initial_policy": policy(trainer),
        "initial_trainable_sha256": before,
        "workload": workload,
        "selected_steps": [index for index, _ in selection],
        "regular_steps": sorted({round(i * (steps - 1) / 7) for i in range(8)}),
        "selection": "8 evenly spaced full-plan steps plus six actual inventory cost maxima",
        "batches": [],
        "optimizer_steps": 0,
    }

    def save():
        temporary = root / "progress.tmp"
        temporary.write_bytes(canonical(result) + b"\n")
        temporary.replace(root / "comparison.json")

    save()
    for attempt in range(2):
        stable_policy = policy(trainer)
        result["batches"] = []
        result["policy_attempt"] = attempt
        for number, (step, indices) in enumerate(selection):
            if trainer.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(trainer.device)

            def stream(indices=indices):
                for _ in range(3):
                    yield prepare_batch(samples, indices, prepare)

            start = time.monotonic()
            measured = profile_backward(
                trainer,
                stream(),
                warmup=1,
                repeats=2,
                cold_images=True,
                reserve_optimizer_state=True,
            )
            memory = (
                {
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(trainer.device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(trainer.device),
                }
                if trainer.device.type == "cuda"
                else {}
            )
            result["batches"].append(
                {
                    "step_index": step,
                    "policy": policy(trainer),
                    "wall_seconds": time.monotonic() - start,
                    **memory,
                    **measured,
                }
            )
            result["effective_policy"] = policy(trainer)
            save()
            print(
                f"[compare] {number + 1}/{len(selection)} step={step} "
                f"median={measured['median_seconds']:.3f}s policy={policy(trainer)}",
                flush=True,
            )
        if policy(trainer) == stable_policy:
            break
        if attempt == 1:
            raise ValueError("GPU memory policy did not stabilize; comparison incomplete")
    if trainable_digest(trainer.model) != before or trainer.step_i or trainer.opt.state:
        raise ValueError("comparison unexpectedly updated live model or optimizer")
    result.update(
        status="complete",
        weights_unchanged=True,
        median_seconds=statistics.median(row["median_seconds"] for row in result["batches"]),
        representative_mean_seconds=statistics.mean(
            row["median_seconds"]
            for row in result["batches"]
            if row["step_index"] in result["regular_steps"]
        ),
        max_seconds=max(row["max_seconds"] for row in result["batches"]),
        policy_changed=policy(trainer) != result["initial_policy"],
        peak_allocated_bytes=max(
            (row.get("peak_allocated_bytes", 0) for row in result["batches"]), default=0
        ),
        peak_reserved_bytes=max(
            (row.get("peak_reserved_bytes", 0) for row in result["batches"]), default=0
        ),
        memory_scope="PyTorch allocator peaks including lazy AdamW moment reservation; not total driver-process memory",
        timing_scope="cold CPU input preparation/vision and production backward after per-batch kernel warmup; no optimizer update",
    )
    save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--deadline", type=float, default=970)
    args = parser.parse_args()
    manifest, splits = validate_bundle(args.bundle)
    if not source_matches(manifest) or not torch.cuda.is_available():
        raise ValueError("comparison requires matching immutable package sources and CUDA")
    if not 0 < args.deadline <= 970:
        raise ValueError("comparison deadline must be positive and no more than 970 seconds")
    recipe = json.loads((Path(args.bundle) / "training_config.json").read_text())
    audit = json.loads((Path(args.bundle) / "model_preflight.json").read_text())
    with job_deadline(args.deadline):
        start = time.monotonic()
        torch.manual_seed(recipe["training"]["seed"])
        torch.set_num_threads(4)
        config = {**recipe["model"], "lora_targets": tuple(recipe["model"]["lora_targets"])}
        cfg = ElectraConfig(**config)
        model, tok, backend = fresh_image_backend(cfg, "cuda", offline=True)
        trainer = Trainer(
            model, tok, TrainConfig(steps=1, **recipe["training"]), "cuda", image_backend=backend
        )
        trainer.micro_tokens = INITIAL_POLICY["micro_tokens"]
        trainer.micro_ckpt_tokens = INITIAL_POLICY["micro_ckpt_tokens"]
        trainer.ckpt_threshold = INITIAL_POLICY["checkpoint_threshold"]
        trainer.model.train()
        result = matched_profile(
            trainer,
            splits["train"],
            audit["train_inventory"],
            lambda sample: prepared_items(sample, tok, cfg, backend),
            args.out,
            seed=trainer.cfg.seed,
            weights=recipe["language_sampling"],
        )
        result["status"] = "backward_complete_overheads_pending"
        (Path(args.out) / "comparison.json").write_bytes(canonical(result) + b"\n")
        overheads = profile_overheads(trainer, Path(args.out) / "io_probe")
        gpu = torch.cuda.get_device_properties(0)
        result.update(
            status="complete",
            gpu=gpu.name,
            gpu_memory_bytes=gpu.total_memory,
            compute_capability=list(torch.cuda.get_device_capability(0)),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            cpu_threads=torch.get_num_threads(),
            training_mode=trainer.model.training,
            backbone_training_mode=trainer.model.backbone.training,
            bundle_manifest_sha256=sha256((Path(args.bundle) / "manifest.json").read_bytes()),
            benchmark_sha256=sha256(Path(__file__).read_bytes()),
            elapsed_seconds=time.monotonic() - start,
            overheads=overheads,
        )
        (Path(args.out) / "comparison.json").write_bytes(canonical(result) + b"\n")
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "gpu",
                        "median_seconds",
                        "max_seconds",
                        "peak_allocated_bytes",
                        "elapsed_seconds",
                        "optimizer_steps",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
