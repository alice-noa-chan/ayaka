"""Training entry point: bundle checks by default; execution requires explicit budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from ..checkpoint import apply_lora, save_checkpoint
from ..config import ElectraConfig
from ..model.electra import ElectraDecisionModel
from ..multimodal import ImageBackend
from ..tokenization import HFTokenizer
from .prepare_v2 import canonical, prepared_items, sha256, validate_bundle
from .trainer import TrainConfig, Trainer


def source_matches(manifest):
    root = Path(__file__).resolve().parents[1]
    current = {
        str(p.relative_to(root)).replace("\\", "/"): sha256(p.read_bytes())
        for p in root.rglob("*.py")
    }
    return bool(manifest.get("source_sha256")) and manifest["source_sha256"] == current


@contextmanager
def job_deadline(seconds):
    """Hard process cap includes loading, backward preflight, training and saving."""
    import os
    from threading import Timer

    def expired():
        print("[v2] hard job deadline reached; stopping before further GPU spend", flush=True)
        os._exit(124)

    timer = Timer(seconds, expired)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def trainable_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def backward_preflight(trainer, items):
    """Run real losses/backward without clipping, optimizer steps or parameter changes."""
    before, old_mode = trainable_digest(trainer.model), trainer.model.training
    trainer.model.train()
    trainer.opt.zero_grad(set_to_none=True)
    losses = []
    try:
        components = trainer.backward_step(items)
        if not components:
            raise ValueError("preflight produced missing losses")
        if any(not torch.isfinite(loss) for loss in components.values()):
            raise ValueError("preflight produced nonfinite loss")
        losses.append(float(components["total"]))
        grads = [
            p.grad for p in trainer.model.parameters() if p.requires_grad and p.grad is not None
        ]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise ValueError("preflight produced missing or nonfinite gradients")
        if not any(g.abs().sum() > 0 for g in grads):
            raise ValueError("preflight gradients are all zero")
        if trainable_digest(trainer.model) != before or trainer.step_i != 0 or trainer.opt.state:
            raise ValueError("preflight unexpectedly updated weights or optimizer state")
        return {
            "losses": losses,
            "finite_gradients": True,
            "weights_unchanged": True,
            "optimizer_steps": 0,
            "loss_components": {key: float(value) for key, value in components.items()},
        }
    finally:
        trainer.opt.zero_grad(set_to_none=True)
        trainer.model.train(old_mode)


def fresh_image_backend(cfg, device, *, offline=True):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    lm = AutoModelForImageTextToText.from_pretrained(
        cfg.backbone,
        revision=cfg.backbone_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": str(device)},
        local_files_only=offline,
    )
    native, text = lm.model, lm.model.language_model
    if lm.get_output_embeddings().weight is not text.get_input_embeddings().weight:
        text.add_module("_ayaka_lm_head", lm.get_output_embeddings())
    model = ElectraDecisionModel(cfg, text, lm.config.text_config).to(device)
    model.backbone.requires_grad_(False)
    apply_lora(model)
    processor = AutoProcessor.from_pretrained(
        cfg.backbone, revision=cfg.backbone_revision, local_files_only=offline
    )
    tok = HFTokenizer(processor.tokenizer, cfg.backbone)
    backend = ImageBackend(native, processor, model, tok, processing_device="cpu")
    native.language_model = model.backbone
    total = sum(
        p.numel() for p in {id(p): p for p in [*native.parameters(), *model.parameters()]}.values()
    )
    if total > 14_000_000_000:
        raise ValueError("training model exceeds 14B parameters")
    return model, tok, backend


def sample_stream(
    samples,
    tok,
    cfg,
    backend,
    questions_per_step,
    seed,
    language_weights=None,
    prepared_cache_bytes=256 * 1024 * 1024,
):
    """Process media lazily; never retain an entire image corpus on GPU or host."""
    if not samples or type(questions_per_step) is not int or questions_per_step < 1:
        raise ValueError("stream requires samples and a positive question count")
    pending = []
    from .prepared_cache import PreparedSampleCache
    from .workload import sample_indices

    prepared = PreparedSampleCache(
        lambda sample: prepared_items(sample, tok, cfg, backend), prepared_cache_bytes
    )
    for index in sample_indices(
        [sample.metadata.get("language", "unknown") for sample in samples], seed, language_weights
    ):
        pending += prepared.get(samples[index])
        while len(pending) >= questions_per_step:
            yield pending[:questions_per_step]
            pending = pending[questions_per_step:]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--execute",
        action="store_true",
        help="explicitly start GPU training; omitted means audit only",
    )
    execution.add_argument(
        "--backward-only",
        action="store_true",
        help="CUDA loss/gradient preflight only; never call an optimizer step",
    )
    execution.add_argument(
        "--profile-only",
        action="store_true",
        help="time production backward batches without optimizer updates",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--max-train-seconds", type=float)
    parser.add_argument("--out")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-weight-downloads", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args(argv)
    manifest, splits = validate_bundle(args.bundle)
    recipe = json.loads((Path(args.bundle) / "training_config.json").read_text(encoding="utf-8"))
    if (
        recipe["initialization"] != "fresh_lora_from_pinned_base"
        or recipe["vision_policy"] != "frozen"
    ):
        raise ValueError("unsupported initialization or vision checkpoint policy")
    config = dict(recipe["model"])
    config["lora_targets"] = tuple(config["lora_targets"])
    cfg = ElectraConfig(**config)
    if not args.execute and not args.backward_only and not args.profile_only:
        result = {
            "status": "bundle_verified_no_training",
            "source_code_matches": source_matches(manifest),
            "manifest_sha256": sha256((Path(args.bundle) / "manifest.json").read_bytes()),
            "optimizer_steps": 0,
            "execution_requires": recipe["execution_requires"],
            "counts": manifest["counts"],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    if args.backward_only or args.profile_only:
        if args.steps is not None:
            raise ValueError("zero-step preflight/profile does not accept optimizer steps")
        args.steps = 1  # scheduler construction only; the optimizer is never executed
    if (
        args.steps is None
        or args.steps < 1
        or args.max_train_seconds is None
        or not math.isfinite(args.max_train_seconds)
        or not 0 < args.max_train_seconds <= 28800
    ):
        raise ValueError("execution requires positive steps and an explicit 0–28800 second budget")
    if not args.out or Path(args.out).exists():
        raise ValueError("execution needs a new output directory")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint interval must be positive")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("pretrained execution requires an available CUDA device")
    if not source_matches(manifest):
        raise ValueError("source files changed after bundle preparation; prepare a new bundle")
    with job_deadline(args.max_train_seconds):
        return execute_training(args, cfg, recipe, splits)


def execute_training(args, cfg, recipe, splits):
    start = time.monotonic()
    torch.manual_seed(recipe["training"]["seed"])
    model, tok, backend = fresh_image_backend(
        cfg, args.device, offline=not args.allow_weight_downloads
    )
    trainer = Trainer(
        model,
        tok,
        TrainConfig(
            steps=args.steps, max_train_seconds=args.max_train_seconds, **recipe["training"]
        ),
        args.device,
        image_backend=backend,
    )
    # Only train data is touched for gradients. Dev/calibration/test never enter the stream.
    representative, seen = [], set()
    for sample in splits["train"]:
        signature = (
            sample.metadata.get("modality", "text"),
            sample.metadata["task_family"],
            sample.metadata["language"],
            tuple(question.type for question in sample.questions),
        )
        if signature not in seen:
            representative += prepared_items(sample, tok, cfg, backend)
            seen.add(signature)
    smoke = backward_preflight(trainer, representative)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    (root / "preflight.json").write_bytes(canonical(smoke) + b"\n")
    if args.backward_only:
        return {"status": "backward_preflight_only", **smoke}

    from .throughput import completion_plan, profile_backward

    def stream():
        return sample_stream(
            splits["train"],
            tok,
            cfg,
            backend,
            trainer.cfg.questions_per_step,
            trainer.cfg.seed,
            recipe["language_sampling"],
            recipe.get("prepared_cache_bytes", 256 * 1024 * 1024),
        )

    profile = profile_backward(trainer, stream())
    (root / "throughput.json").write_bytes(canonical(profile) + b"\n")
    if getattr(args, "profile_only", False):
        return {"status": "throughput_profile_only", **profile}
    target_seconds = min(
        args.max_train_seconds, recipe.get("completion_target_seconds", args.max_train_seconds)
    )
    plan = completion_plan(profile, args.steps, max(0, target_seconds - (time.monotonic() - start)))
    (root / "completion_plan.json").write_bytes(canonical(plan) + b"\n")
    if not plan["fits"]:
        raise ValueError(
            "complete planned schedule does not fit the measured budget; no optimizer steps executed"
        )
    # Normal completion follows the fixed step/LR schedule. The outer deadline
    # remains an emergency process cap, never the intended training endpoint.
    trainer.cfg.max_train_seconds = 0

    def save_progress(step, record):
        if step != 1 and step % args.checkpoint_every:
            return
        checkpoint = root / f"step-{step:08d}"
        save_checkpoint(
            model,
            str(checkpoint),
            {
                "steps": step,
                "bundle_manifest_sha256": sha256(
                    (Path(args.bundle) / "manifest.json").read_bytes()
                ),
                "record": record,
            },
        )
        torch.save(
            {
                "optimizer": trainer.opt.state_dict(),
                "schedule": trainer.sched.state_dict(),
                "step": step,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all(),
            },
            checkpoint / "training_state.pt",
        )
        (checkpoint / "complete.json").write_bytes(
            canonical({"steps": step, "complete": True}) + b"\n"
        )

    history = trainer.train(
        stream(),
        on_step=save_progress,
    )
    meta = {
        "bundle_manifest_sha256": sha256((Path(args.bundle) / "manifest.json").read_bytes()),
        "preflight": smoke,
        "steps": trainer.step_i,
        "stopped_early": trainer.stopped_early,
        "calibration": "not_fitted",
        "router": "not_promoted",
        "test": "not_used",
        "initialization": recipe["initialization"],
        "completion_plan": plan,
        "complete": trainer.step_i == args.steps and not trainer.stopped_early,
    }
    save_checkpoint(model, str(root / "checkpoint"), meta)
    (root / "checkpoint" / "complete.json").write_bytes(
        canonical({"steps": trainer.step_i, "complete": meta["complete"]}) + b"\n"
    )
    (root / "history.json").write_bytes(canonical(history) + b"\n")
    return meta


if __name__ == "__main__":
    main()
