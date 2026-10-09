"""Complete direct schedules and atomic, source-bound optimizer continuation.

These functions never load pretrained weights or start a GPU job. The caller
must audit the bundle, verify loaded native weights/kernels and admit the full
paid workload before invoking its production training loop.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..eval.read_artifact import fingerprint
from .direct_bundle import training_batches
from .prepare_v2 import canonical

VERSION = "ayaka-direct-optimizer-state-1"
FILES = {"trainable.safetensors", "training.pt"}


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_config(recipe, cfg):
    schedule = recipe["schedule"]
    if (
        any(
            type(value) is not int or value < 1
            for value in (
                cfg.steps,
                cfg.questions_per_step,
                cfg.micro_batch_tokens,
                cfg.min_micro_batch_tokens,
            )
        )
        or type(cfg.seed) is not int
    ):
        raise ValueError(
            "direct training requires positive integer batch/step counts and an integer seed"
        )
    if (cfg.steps, cfg.questions_per_step, cfg.seed) != (
        schedule["steps"],
        schedule["rows_per_step"],
        schedule["seed"],
    ):
        raise ValueError("training schedule or seed differs from the prepared complete workload")
    if asdict(cfg.loss_weights) != recipe["loss_weights"]:
        raise ValueError("training loss weights differ from the gold-anchored bundle")
    if cfg.max_train_seconds != 0 or cfg.reasoning_ce_weight != 0 or cfg.proposal_ce_weight != 0:
        raise ValueError(
            "direct runs require complete schedules without time truncation or auxiliary CE"
        )
    if cfg.loss_weights.pointer_aux != 0 or cfg.loss_weights.gold_nll_with_teacher is not True:
        raise ValueError("native direct runs require gold anchoring and no pointer auxiliary loss")
    values = asdict(cfg.loss_weights)
    if (
        any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for key, value in values.items()
            if key != "gold_nll_with_teacher"
        )
        or cfg.loss_weights.nll <= 0
    ):
        raise ValueError(
            "direct loss weights must be finite and nonnegative with positive gold NLL"
        )
    if any(not math.isfinite(x) or x <= 0 for x in (cfg.lr, cfg.head_lr, cfg.grad_clip)):
        raise ValueError("direct learning rates and gradient clipping must be positive and finite")


def training_binding(manifest, recipe, cfg, *, native_weights_sha256):
    validate_training_config(recipe, cfg)
    if (
        not isinstance(native_weights_sha256, str)
        or len(native_weights_sha256) != 64
        or any(c not in "0123456789abcdef" for c in native_weights_sha256)
    ):
        raise ValueError("actual native weights require an exact SHA256 before binding the run")
    return {
        "version": VERSION,
        "bundle_sha256": fingerprint(manifest),
        "recipe_sha256": fingerprint(recipe),
        "training_sha256": fingerprint(asdict(cfg)),
        "native_weights_sha256": native_weights_sha256,
        "source_sha256": manifest["source_sha256"],
        "model": recipe["model"],
        "optimizations": recipe["optimizations"],
        "scope": "declared run binding; loaded-weight and CUDA execution proof supplied by runner",
    }


def _trainable(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def _optimizer_names(trainer):
    names = {id(p): name for name, p in trainer.model.named_parameters()}
    return [[names[id(p)] for p in group["params"]] for group in trainer.opt.param_groups]


def _validate_adam_state(trainer, state, step):
    optimizer = state["optimizer"]
    current = trainer.opt.state_dict()

    def fixed_scheduler(values):
        return {
            key: value
            for key, value in values.items()
            if key not in {"last_epoch", "_step_count", "_last_lr"}
        }

    if (
        fixed_scheduler(state["scheduler"]) != fixed_scheduler(trainer.sched.state_dict())
        or state["scheduler"].get("_step_count") != step + 1
    ):
        raise ValueError("scheduler definition or step counter differs from the bound run")
    if len(optimizer["param_groups"]) != len(current["param_groups"]):
        raise ValueError("optimizer group inventory differs from checkpoint")
    parameters = {}
    last_lr = state["scheduler"]["_last_lr"]
    if len(last_lr) != len(current["param_groups"]):
        raise ValueError("scheduler LR inventory differs from optimizer")
    for index, (saved, expected, live) in enumerate(
        zip(
            optimizer["param_groups"],
            current["param_groups"],
            trainer.opt.param_groups,
            strict=True,
        )
    ):
        if saved["params"] != expected["params"] or {
            key: value for key, value in saved.items() if key not in {"params", "lr"}
        } != {key: value for key, value in expected.items() if key not in {"params", "lr"}}:
            raise ValueError("optimizer parameter IDs or hyperparameters differ from checkpoint")
        expected_lr = trainer.sched.base_lrs[index] * trainer.sched.lr_lambdas[index](step)
        if (
            not math.isclose(saved["lr"], expected_lr, rel_tol=1e-12, abs_tol=1e-15)
            or saved["lr"] != last_lr[index]
        ):
            raise ValueError("optimizer LR disagrees with the bound schedule")
        parameters.update(zip(saved["params"], live["params"], strict=True))
    if set(optimizer["state"]) - set(parameters):
        raise ValueError("optimizer moments refer to unknown parameters")
    for index, moments in optimizer["state"].items():
        if set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("optimizer must contain the exact AdamW state")
        parameter = parameters[index]
        counter = moments["step"]
        if (
            not isinstance(counter, torch.Tensor)
            or counter.numel() != 1
            or counter.ndim != 0
            or counter.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(counter).all()
            or not 0 < counter.item() <= step
            or counter.item() != int(counter.item())
        ):
            raise ValueError("invalid optimizer step counter")
        for key in ("exp_avg", "exp_avg_sq"):
            value = moments[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError("invalid optimizer moment tensor")
        if (moments["exp_avg_sq"] < 0).any():
            raise ValueError("optimizer squared moments must be nonnegative")


def save_training_state(trainer, destination, binding):
    """Publish a fresh whole-step directory only after every payload is durable."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError("optimizer checkpoint destination must be new")
    if type(trainer.step_i) is not int or not 0 < trainer.step_i <= trainer.cfg.steps:
        raise ValueError("save only completed optimizer-step boundaries")
    if any(p.grad is not None for p in trainer.model.parameters()):
        raise ValueError("optimizer checkpoint requires cleared step-boundary gradients")
    if fingerprint(asdict(trainer.cfg)) != binding["training_sha256"]:
        raise ValueError("trainer settings changed since run binding")
    if fingerprint(asdict(trainer.model.cfg)) != fingerprint(binding["model"]):
        raise ValueError("trainer model differs from the bound native configuration")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    params = _trainable(trainer.model)
    if not params or any(not torch.isfinite(p).all() for p in params.values()):
        raise ValueError("cannot checkpoint missing or nonfinite trainable weights")
    save_file(
        {name: p.detach().cpu().contiguous() for name, p in params.items()},
        str(staging / "trainable.safetensors"),
    )
    state = {
        "step": trainer.step_i,
        "optimizer": trainer.opt.state_dict(),
        "scheduler": trainer.sched.state_dict(),
        "optimizer_names": _optimizer_names(trainer),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if trainer.device.type == "cuda" else [],
        "device_type": trainer.device.type,
        "micro_tokens": trainer.micro_tokens,
        "micro_ckpt_tokens": trainer.micro_ckpt_tokens,
        "ckpt_threshold": trainer.ckpt_threshold,
        "ckpt_active": trainer._ckpt_active,
    }
    _validate_adam_state(trainer, state, trainer.step_i)
    torch.save(state, staging / "training.pt")
    for name in FILES:
        with (staging / name).open("r+b") as stream:
            os.fsync(stream.fileno())
    header = {
        "version": VERSION,
        "binding": binding,
        "step": trainer.step_i,
        "files": {name: file_digest(staging / name) for name in sorted(FILES)},
        "parameter_schema": {
            name: {"shape": list(p.shape), "dtype": str(p.dtype)} for name, p in params.items()
        },
        "complete_schedule": trainer.step_i == trainer.cfg.steps,
        "promotable": False,
    }
    with (staging / "manifest.json").open("wb") as stream:
        stream.write(canonical(header) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Same-parent rename publishes the complete directory atomically. A failed
    # save leaves an unpublished staging directory for diagnosis, never a valid
    # checkpoint; no recursive cleanup or existing output is performed here.
    if destination.exists():
        raise ValueError("optimizer checkpoint destination appeared during save")
    staging.rename(destination)
    return header


def load_training_state(trainer, source, binding):
    """Validate payload, model schema and state before restoring exact continuation."""
    root = Path(source)
    if trainer.step_i != 0 or trainer.opt.state:
        raise ValueError("optimizer continuation requires a fresh zero-step trainer")
    header = json.loads((root / "manifest.json").read_bytes())
    if header.get("version") != VERSION or header.get("binding") != binding:
        raise ValueError("optimizer checkpoint belongs to another data/model/source/training run")
    if fingerprint(asdict(trainer.cfg)) != binding["training_sha256"]:
        raise ValueError("current trainer settings differ from the bound optimizer run")
    if fingerprint(asdict(trainer.model.cfg)) != fingerprint(binding["model"]):
        raise ValueError("current model differs from the bound native configuration")
    if set(header.get("files", {})) != FILES:
        raise ValueError("optimizer manifest must cover both exact payloads")
    if any(file_digest(root / name) != digest for name, digest in header["files"].items()):
        raise ValueError("optimizer checkpoint checksum mismatch")
    params = _trainable(trainer.model)
    expected = {name: {"shape": list(p.shape), "dtype": str(p.dtype)} for name, p in params.items()}
    if header.get("parameter_schema") != expected:
        raise ValueError("current trainable parameter schema differs from checkpoint")
    tensors = load_file(str(root / "trainable.safetensors"), device="cpu")
    if set(tensors) != set(params) or any(
        tensors[name].shape != p.shape
        or tensors[name].dtype != p.dtype
        or not torch.isfinite(tensors[name]).all()
        for name, p in params.items()
    ):
        raise ValueError("checkpoint trainable tensors are incomplete or invalid")
    state = torch.load(root / "training.pt", map_location="cpu", weights_only=True)
    step = state.get("step")
    if type(step) is not int or not 0 < step <= trainer.cfg.steps or header["step"] != step:
        raise ValueError("checkpoint step lies outside the complete schedule")
    if (
        header.get("complete_schedule") is not (step == trainer.cfg.steps)
        or header.get("promotable") is not False
    ):
        raise ValueError("optimizer checkpoint falsely declares completion or promotion")
    if state.get("optimizer_names") != _optimizer_names(trainer):
        raise ValueError("optimizer parameter ordering differs from checkpoint")
    if state["scheduler"].get("last_epoch") != step or state["device_type"] != trainer.device.type:
        raise ValueError("scheduler step or device type differs from the continuation contract")
    _validate_adam_state(trainer, state, step)
    if any(
        type(state[k]) is not int or state[k] < 1 for k in ("micro_tokens", "micro_ckpt_tokens")
    ):
        raise ValueError("invalid adaptive microbatch budgets")
    threshold = state["ckpt_threshold"]
    if threshold is not None and (type(threshold) is not int or threshold < 0):
        raise ValueError("invalid activation checkpoint threshold")
    if type(state["ckpt_active"]) is not bool:
        raise ValueError("invalid activation checkpointing state")
    torch.Generator().set_state(state["torch_rng"])
    random.Random().setstate(state["python_rng"])
    if trainer.device.type == "cuda" and len(state["cuda_rng"]) != torch.cuda.device_count():
        raise ValueError("CUDA RNG device inventory differs from checkpoint")
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(tensors[name].to(p.device))
    trainer.opt.load_state_dict(state["optimizer"])
    trainer.sched.load_state_dict(state["scheduler"])
    trainer.step_i = step
    trainer.micro_tokens, trainer.micro_ckpt_tokens = (
        state["micro_tokens"],
        state["micro_ckpt_tokens"],
    )
    trainer.ckpt_threshold = threshold
    trainer._set_checkpointing(state["ckpt_active"])
    trainer.stopped_early = False
    trainer.opt.zero_grad(set_to_none=True)
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    if trainer.device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return header


def train_fixed_schedule(trainer, recipe, inventory, groups, *, on_step=None):
    """Consume every prepared optimizer step; exhaustion or route leakage is an error.

    ``on_step`` may return True to stop early (validation-based checkpoint
    selection); any other return value continues the schedule.
    """
    validate_training_config(recipe, trainer.cfg)
    if any(item.direct_distillation is not True for group in groups for item in group):
        raise ValueError("every prepared row in a direct run must be marked direct-distillation")
    history = []
    stopped = False
    for batch in training_batches(recipe, inventory, groups, start_step=trainer.step_i):
        if len(batch) != trainer.cfg.questions_per_step:
            raise ValueError("prepared batch differs from the fixed row count")
        record = trainer.train_step(batch)
        history.append(record)
        if on_step is not None and on_step(trainer.step_i, record) is True:
            stopped = True
            break
    if not stopped and trainer.step_i != trainer.cfg.steps:
        raise ValueError("direct run did not complete its entire prepared schedule")
    return history
