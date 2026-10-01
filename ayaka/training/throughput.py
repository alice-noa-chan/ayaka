"""Zero-update production backward timing and complete-schedule budget forecasts."""

import math
import statistics
import time

import torch


def profile_backward(trainer, stream, *, warmup=1, repeats=3):
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
        }
    finally:
        trainer.opt.zero_grad(set_to_none=True)
        trainer.model.train(mode)
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)


def completion_plan(
    profile, steps, remaining_seconds, *, save_seconds=120, optimizer_seconds=1, safety_factor=1.25
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
    estimated = (profile["max_seconds"] + optimizer_seconds) * steps * safety_factor + save_seconds
    return {
        "planned_steps": steps,
        "observed_max_backward_seconds": profile["max_seconds"],
        "optimizer_seconds_allowance_per_step": optimizer_seconds,
        "safety_factor": safety_factor,
        "save_seconds_reserved": save_seconds,
        "estimated_remaining_seconds": estimated,
        "available_remaining_seconds": remaining_seconds,
        "fits": estimated <= remaining_seconds,
        "semantics": "estimate for completing every scheduled step, not a time-based partial curriculum",
        "limitations": "short train-only timing sample; optimizer/save allowances are assumptions; no wall-clock guarantee",
    }
