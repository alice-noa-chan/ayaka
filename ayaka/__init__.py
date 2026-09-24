"""Electra decision models on Gemma 4 backbones.

from ayaka import Decision, QuestionSpec, load_checkpoint
model = load_checkpoint("artifacts/<run>/checkpoint", device="cuda")
decision = Decision(model, HFTokenizer.from_pretrained(model.cfg.backbone))
decision.choice(state, "What does the user want?", ["refund", "track order"])
"""

from .checkpoint import load_checkpoint
from .primitives import Decision, DecisionResult, QuestionSpec
from .tokenization import HFTokenizer

__all__ = ["Decision", "DecisionResult", "HFTokenizer", "QuestionSpec", "load_checkpoint"]
