"""Electra decision models on Gemma 4 backbones.

from ayaka import Decision, QuestionSpec, load_checkpoint
model = load_checkpoint("artifacts/<run>/checkpoint", device="cuda")
decision = Decision(model, HFTokenizer.from_pretrained(model.cfg.backbone))
decision.choice(state, "What does the user want?", ["refund", "track order"])
"""

from importlib import import_module
from typing import Any

__all__ = ["Decision", "DecisionResult", "HFTokenizer", "QuestionSpec", "load_checkpoint"]


def __getattr__(name: str) -> Any:
    """Load model dependencies only when the model API is used."""
    modules = {
        "Decision": ".primitives",
        "DecisionResult": ".primitives",
        "QuestionSpec": ".primitives",
        "HFTokenizer": ".tokenization",
        "load_checkpoint": ".checkpoint",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
