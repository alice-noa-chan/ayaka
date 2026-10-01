"""Zero-update production backward timing and complete-schedule budget forecasts."""

import math
import statistics
import time

import torch


def profile_backward(trainer, stream, *, warmup=1, repeats=3, cold_images=False):
    from .run_v2 import trainable_digest

    if type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("profile counts must be nonnegative warmup and positive repeats")
    if trainer.step_i or trainer.opt.state:
        raise ValueError("completion profiling must precede every optimizer update")
    before, mode = trainable_digest(trainer.model), trainer.model.training
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if trainer.device.type == "cuda" else None
    timings, batches = [], []

    def synchronize():
        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)

    try:
        for iteration in range(warmup + repeats):
            trainer.opt.zero_grad(set_to_none=True)
            if cold_images and trainer.image_features is not None:
                trainer.image_features.entries.clear()
                trainer.image_features.bytes = 0
            synchronize()
            start = time.perf_counter()
            items = next(stream)  # includes tokenization/CPU media preparation and transfers
            losses = trainer.backward_step(items)
            synchronize()
            seconds = time.perf_counter() - start
            if not losses or any(not torch.isfinite(value).all() for value in losses.values()):
                raise ValueError("throughput profile produced nonfinite losses")
            if iteration >= warmup:
                timings.append(seconds)
                batches.append(
                    {
                        "rows": len(items),
                        "text_tokens": sum(
                            len(it.enc.prefix_ids) + len(it.enc.rendered.suffix_ids) for it in items
                        ),
                        "trace_tokens": sum(len(it.reasoning_labels or []) for it in items),
                        "proposal_tokens": sum(len(it.proposal_labels or []) for it in items),
                        "image_rows": sum(it.native_inputs is not None for it in items),
                        "forward_chunks": len(trainer._plan(items)),
                    }
                )
        if trainable_digest(trainer.model) != before or trainer.step_i or trainer.opt.state:
            raise ValueError("throughput profile unexpectedly updated weights or optimizer state")
        return {
            "device": str(trainer.device),
            "warmup_batches": warmup,
            "measured_batches": repeats,
            "seconds": timings,
            "median_seconds": statistics.median(timings),
            "max_seconds": max(timings),
            "batches": batches,
            "last_losses": {key: float(value) for key, value in losses.items()},
            "optimizer_steps": 0,
            "weights_unchanged": True,
            "scope": "production forward/backward and input preparation; optimizer/save not timed",
            "cold_image_features": cold_images,
        }
    finally:
        trainer.opt.zero_grad(set_to_none=True)
        trainer.model.train(mode)
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)


def completion_plan(
    profile,
    steps,
    remaining_seconds,
    *,
    save_seconds=120,
    optimizer_seconds=1,
    safety_factor=1.25,
    overheads=None,
    checkpoint_every=100,
):
    if type(steps) is not int or steps < 1:
        raise ValueError("completion plan requires a positive fixed optimizer step count")
    numbers = [
        profile["max_seconds"],
        remaining_seconds,
        save_seconds,
        optimizer_seconds,
        safety_factor,
    ]
    if any(not math.isfinite(n) or n < 0 for n in numbers) or safety_factor < 1:
        raise ValueError("completion plan needs finite positive measurements and margins")
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint interval must be positive")
    saves = 2 + steps // checkpoint_every - int(checkpoint_every == 1)
    if overheads is not None:
        optimizer_seconds = overheads["max_optimizer_seconds"]
        save_seconds = overheads["checkpoint_seconds"] * saves
        if any(not math.isfinite(n) or n < 0 for n in (optimizer_seconds, save_seconds)):
            raise ValueError("measured overheads must be finite and nonnegative")
    estimated = (
        profile["max_seconds"] + optimizer_seconds
    ) * steps * safety_factor + save_seconds * (safety_factor if overheads else 1)
    disk_needed = (overheads["checkpoint_bytes"] * saves * 1.1) if overheads else None
    disk_fits = disk_needed <= overheads["disk_free_bytes"] if overheads else True
    return {
        "planned_steps": steps,
        "observed_max_backward_seconds": profile["max_seconds"],
        "optimizer_seconds_allowance_per_step": optimizer_seconds,
        "safety_factor": safety_factor,
        "save_seconds_reserved": save_seconds,
        "estimated_remaining_seconds": estimated,
        "available_remaining_seconds": remaining_seconds,
        "checkpoint_writes": saves,
        "estimated_checkpoint_bytes": disk_needed,
        "disk_fits": disk_fits,
        "overheads_measured": overheads is not None,
        "fits": estimated <= remaining_seconds and disk_fits,
        "semantics": "estimate for completing every scheduled step, not a time-based partial curriculum",
        "limitations": (
            "measured cold/warm stress and isolated optimizer/save; future hardware contention can differ"
            if overheads
            else "short train-only timing sample; optimizer/save allowances are assumptions; no wall-clock guarantee"
        ),
    }


def profile_overheads(trainer, out, *, repeats=3):
    """Time clipping/AdamW on disposable tensors and actual checkpoint disk writes."""
    import inspect
    import os
    import shutil
    from pathlib import Path

    from ..checkpoint import save_checkpoint
    from .run_v2 import trainable_digest

    if type(repeats) is not int or repeats < 1 or trainer.opt.state or trainer.step_i:
        raise ValueError("overhead profile must precede optimizer training")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    before, rng = trainable_digest(trainer.model), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if trainer.device.type == "cuda" else None
    times = []
    shadow_groups = []
    for group in trainer.opt.param_groups:
        clones = [torch.nn.Parameter(torch.zeros_like(parameter)) for parameter in group["params"]]
        shadow_groups.append(
            {**{k: v for k, v in group.items() if k != "params"}, "params": clones}
        )
    accepted = inspect.signature(torch.optim.AdamW).parameters
    optimizer = torch.optim.AdamW(
        shadow_groups,
        **{key: value for key, value in trainer.opt.defaults.items() if key in accepted},
    )
    parameters = [parameter for group in shadow_groups for parameter in group["params"]]

    def synchronize():
        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)

    try:
        for _ in range(repeats + 1):
            for parameter in parameters:
                parameter.grad = torch.full_like(parameter, 1e-4)
            synchronize()
            start = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(parameters, trainer.cfg.grad_clip)
            optimizer.step()  # independent disposable tensors, never live model parameters
            synchronize()
            times.append(time.perf_counter() - start)
        synchronize()
        start = time.perf_counter()
        save_checkpoint(
            trainer.model,
            str(root / "untrained_io_probe"),
            {
                "purpose": "untrained checkpoint IO probe; not a trained checkpoint",
                "optimizer_steps": 0,
            },
        )
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "schedule": trainer.sched.state_dict(),
                "step": 0,
                "rng_cpu": rng,
                "rng_cuda": cuda_rng,
            },
            root / "untrained_io_probe" / "training_state.pt",
        )
        for path in (root / "untrained_io_probe").rglob("*"):
            if path.is_file():
                with path.open("r+b") as handle:
                    os.fsync(handle.fileno())
        seconds = time.perf_counter() - start
        size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        if trainable_digest(trainer.model) != before or trainer.opt.state or trainer.step_i:
            raise ValueError("overhead profiling unexpectedly changed live training state")
        return {
            "optimizer_seconds": times,
            "max_optimizer_seconds": max(times),
            "checkpoint_seconds": seconds,
            "checkpoint_bytes": size,
            "disk_free_bytes": shutil.disk_usage(root).free,
            "model_optimizer_steps": 0,
            "disposable_optimizer_steps": repeats + 1,
            "weights_unchanged": True,
            "scope": "same shapes/dtypes/AdamW/clipping on disposable tensors; full adapter/head and optimizer-state IO with fsync",
            "probe_is_trained_checkpoint": False,
        }
    finally:
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def profile_production(trainer, stream, samples, inventory, prepare):
    """Measure random production batches plus cold largest-row strata, train only."""
    import itertools
    from collections import defaultdict

    def policy():
        return {
            "micro_tokens": trainer.micro_tokens,
            "micro_ckpt_tokens": trainer.micro_ckpt_tokens,
            "checkpoint_threshold": trainer.ckpt_threshold,
        }

    initial = policy()
    ordinary = profile_backward(trainer, stream, repeats=6)
    groups = defaultdict(list)
    for index, sample in enumerate(inventory):
        for position, row in enumerate(sample["rows"]):
            key = (
                sample["language"],
                row["type"],
                row["image"],
                bool(row["trace_tokens"]),
                bool(row["proposal_tokens"]),
                row["flagged"],
            )
            groups[key].append(
                (row["length"] + row["trace_tokens"] + row["proposal_tokens"], index, position)
            )
    stress = []
    for key, candidates in sorted(groups.items()):

        def batch(candidates=candidates):
            rows, prepared = [], {}
            for _, index, position in sorted(candidates, reverse=True)[
                : trainer.cfg.questions_per_step
            ]:
                if index not in prepared:
                    prepared[index] = prepare(samples[index])
                rows.append(prepared[index][position])
            rows = (rows * math.ceil(trainer.cfg.questions_per_step / len(rows)))[
                : trainer.cfg.questions_per_step
            ]
            yield rows

        # Every measured stress batch starts with an empty feature cache; even
        # cached corpora cannot mask cold vision cost or largest-row OOM fallback.
        measured = profile_backward(trainer, batch(), warmup=0, repeats=1, cold_images=True)
        stress.append({"stratum": list(key), **measured})
    if policy() != initial:
        ordinary = profile_backward(trainer, stream, repeats=6)
    all_seconds = list(itertools.chain(ordinary["seconds"], *(row["seconds"] for row in stress)))
    return {
        **ordinary,
        "stress": stress,
        "max_seconds": max(all_seconds),
        "effective_policy": policy(),
        "oom_policy_changed": policy() != initial,
        "scope": "production sample and cold largest-row language/type/route/flag strata including CPU preparation",
    }
