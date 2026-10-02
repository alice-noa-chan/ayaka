"""Reuse a bound zero-update profile after a fresh, identical-batch speed check."""

import copy
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from .prepare_v2 import canonical
from .throughput import profile_backward
from .workload import finite_workload


def profile_binding(cfg, recipe, inventory, steps):
    if not inventory or any(not row.get("content_sha256") for row in inventory):
        raise ValueError(
            "reference reuse requires content-bound rows; regenerate legacy preparation"
        )
    package = Path(__file__).resolve().parents[1]
    # These entry points change admission/preparation, not numerical training.
    excluded = {
        "training/run_v2.py",
        "training/prepare_recovery.py",
        "training/reference_profile.py",
    }
    engine = {
        str(p.relative_to(package)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in package.rglob("*.py")
        if str(p.relative_to(package)).replace("\\", "/") not in excluded
    }
    workload = finite_workload(
        inventory,
        steps,
        recipe["training"]["questions_per_step"],
        recipe["training"]["seed"],
        recipe["language_sampling"],
    )
    return {
        "parent": recipe.get("initial_checkpoint_sha256"),
        "model": hashlib.sha256(canonical(asdict(cfg))).hexdigest(),
        "training": hashlib.sha256(
            canonical({"training": recipe["training"], "languages": recipe["language_sampling"]})
        ).hexdigest(),
        "schedule": workload["schedule_sha256"],
        "inventory": workload["inventory_sha256"],
        "engine": engine,
    }


def runtime_signature(device):
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "cpu_threads": torch.get_num_threads(),
    }


def validate_reference(root, binding, runtime):
    root = Path(root)
    certificate = json.loads((root / "binding.json").read_text())
    if certificate["binding"] != binding or certificate["runtime"] != runtime:
        raise ValueError("reference profile model/workload/engine/runtime mismatch")
    path = root / "throughput.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != certificate["throughput_sha256"]:
        raise ValueError("reference throughput checksum mismatch")
    profile = json.loads(path.read_text())
    if not profile.get("weights_unchanged") or profile.get("optimizer_steps") != 0:
        raise ValueError("reference profile must have no parameter updates")
    if profile["schedule"]["schedule_sha256"] != binding["schedule"]:
        raise ValueError("reference schedule mismatch")
    return profile


def reuse_profile(root, trainer, stream, cfg, recipe, inventory, steps):
    if trainer.device.type != "cuda":
        raise ValueError("reference timing reuse requires the same CUDA device class")
    reference = validate_reference(
        root, profile_binding(cfg, recipe, inventory, steps), runtime_signature(trainer.device)
    )
    policy = {
        "micro_tokens": trainer.micro_tokens,
        "micro_ckpt_tokens": trainer.micro_ckpt_tokens,
        "checkpoint_threshold": trainer.ckpt_threshold,
    }
    if policy != reference["effective_policy"]:
        raise ValueError("reference memory policy mismatch")
    probe = profile_backward(trainer, stream, repeats=3, reserve_optimizer_state=True)
    if probe["batches"] != reference["batches"][:3]:
        raise ValueError("fresh reference probe must repeat identical production batches")
    current_policy = {
        "micro_tokens": trainer.micro_tokens,
        "micro_ckpt_tokens": trainer.micro_ckpt_tokens,
        "checkpoint_threshold": trainer.ckpt_threshold,
    }
    if current_policy != policy:
        raise ValueError("fresh memory fallback invalidates reference profile")
    scale = max(1.0, probe["max_seconds"] / max(reference["seconds"][:3]))
    result = copy.deepcopy(reference)
    result["max_seconds"] *= scale
    result["schedule"]["max_seconds"] *= scale
    result["reference_revalidation"] = {
        "probe": probe,
        "scale": scale,
        "scope": "same bound engine/model/schedule/runtime, identical fresh batches; estimate, not a wall-clock guarantee",
    }
    return result
