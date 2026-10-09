"""Validation-based checkpoint selection with early stopping for direct training.

A fixed schedule can overfit a large corpus late in training. The selector
reads a reserved validation split of the bundle every ``every`` optimizer
steps (and at the last step), keeps the trainable weights with the lowest
validation loss, and asks the schedule to stop once ``patience`` reads in a
row fail to improve by more than ``min_delta``.

The loss is the equal-type mean of raw (untempered) NLL, so later temperature
fitting cannot hide an overconfident checkpoint. The validation split must be
a bundle split that calibration, export and the held-out comparison do not
use for any fit; ``router_train`` of a direct bundle is such a split.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

VERSION = "ayaka-direct-checkpoint-selection-1"
PROGRESS = "checkpoint_selection.json"
BEST = "best-trainable-{step:08d}.safetensors"


def selection_loss(summary):
    """Equal-type mean raw NLL of a ``summarize`` result."""
    by_type = summary["by_type"]
    if not by_type:
        raise ValueError("checkpoint selection needs at least one validation question")
    values = [by_type[kind]["nll"] for kind in sorted(by_type)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("checkpoint selection loss must be finite")
    return sum(values) / len(values)


def _trainable(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


class CheckpointSelector:
    """Track the best validation read; ``__call__`` returns True to stop early."""

    def __init__(self, evaluate, *, split, every, patience, min_delta=0.0, root=None):
        if type(every) is not int or every < 1:
            raise ValueError("checkpoint selection interval must be a positive integer")
        if type(patience) is not int or patience < 1:
            raise ValueError("checkpoint selection patience must be a positive integer")
        if type(min_delta) not in (int, float) or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("checkpoint selection min_delta must be finite and nonnegative")
        self.evaluate = evaluate
        self.split = split
        self.every, self.patience, self.min_delta = every, patience, float(min_delta)
        self.root = Path(root) if root is not None else None
        self.history = []
        self.best = None
        self.stale = 0
        self._best_params = None

    def due(self, step, final):
        return final or step % self.every == 0

    def __call__(self, trainer, step, *, final):
        if not self.due(step, final):
            return False
        result = self.evaluate(trainer)
        loss = selection_loss(result["summary"])
        improved = self.best is None or loss < self.best["loss"] - self.min_delta
        if improved:
            self.best = {"step": step, "loss": loss}
            self.stale = 0
            self._best_params = {
                name: p.detach().to("cpu", copy=True).contiguous()
                for name, p in _trainable(trainer.model).items()
            }
        else:
            self.stale += 1
        by_type = result["summary"]["by_type"]
        self.history.append(
            {
                "step": step,
                "loss": loss,
                "improved": improved,
                "nll_by_type": {kind: by_type[kind]["nll"] for kind in sorted(by_type)},
                "cc_equal_types": result["summary"]["cc_equal_types"],
            }
        )
        stop = self.stale >= self.patience
        if self.root is not None:
            self.save(self.root)
        return stop and not final

    def restore(self, trainer):
        """Load the best weights into the trainer's model (no optimizer change)."""
        if self._best_params is None:
            raise ValueError("no validation read was made; nothing to restore")
        params = _trainable(trainer.model)
        if set(params) != set(self._best_params):
            raise ValueError("trainable parameters changed since the best read")
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(self._best_params[name].to(device=p.device, dtype=p.dtype))
        return self.best

    def report(self, final_step):
        if self.best is None:
            raise ValueError("no validation read was made")
        return {
            "version": VERSION,
            "split": self.split,
            "loss": "equal-type mean raw NLL",
            "every": self.every,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": dict(self.best),
            "last_step": final_step,
            "stopped_early": self.stale >= self.patience,
            "history": list(self.history),
        }

    def save(self, root):
        """Persist progress and the best weights so a resumed run keeps its selection."""
        root = Path(root)
        if self._best_params is not None:
            # One file per improved read: a resumed run may need an earlier best.
            name = BEST.format(step=self.best["step"])
            if not (root / name).exists():
                staging = root / f".{name}.tmp"
                save_file(self._best_params, str(staging))
                staging.replace(root / name)
        progress = {
            "version": VERSION,
            "split": self.split,
            "every": self.every,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self.best,
            "stale": self.stale,
            "history": self.history,
        }
        staging = root / f".{PROGRESS}.tmp"
        staging.write_text(json.dumps(progress, sort_keys=True), encoding="utf-8")
        staging.replace(root / PROGRESS)

    def load(self, root, *, resumed_step):
        """Restore progress saved by an interrupted run that is resumed at ``resumed_step``."""
        root = Path(root)
        progress = json.loads((root / PROGRESS).read_text(encoding="utf-8"))
        settings = (progress.get("version"), progress.get("split"), progress.get("every"))
        expected = (VERSION, self.split, self.every)
        if settings != expected or progress.get("patience") != self.patience:
            raise ValueError("saved checkpoint selection used different settings")
        if progress.get("min_delta") != self.min_delta:
            raise ValueError("saved checkpoint selection used a different min_delta")
        # Reads after the resumed optimizer state are replayed by the resumed run.
        self.history = [h for h in progress["history"] if h["step"] <= resumed_step]
        best = [h for h in self.history if h["improved"]]
        if best:
            self.best = {"step": best[-1]["step"], "loss": best[-1]["loss"]}
            self._best_params = load_file(
                str(root / BEST.format(step=self.best["step"])), device="cpu"
            )
        last_best = self.best["step"] if self.best else None
        self.stale = sum(1 for h in self.history if last_best is None or h["step"] > last_best)
        return self
