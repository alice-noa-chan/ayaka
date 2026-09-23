"""Primitive semantics API (docs.md section 15).

One internal kernel — Decision(state, question, candidates) ->
probability[K] — with three interpretations at the API layer:

- Choice: candidates -> softmax distribution
- Noul:   proposition -> two implicit candidates -> P(true)
- Score:  ordered semantic levels -> distribution + expected value
          (ordinal meaning lives in candidate metadata, addendum A3)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .collate import build_decision_inputs
from .model.model import ElectraDecisionModel
from .tokenizer import Tokenizer


@dataclass
class QuestionSpec:
    type: str  # "choice" | "noul" | "score"
    instruction: str
    candidates: list[str]
    ordinals: list[int] | None = None  # score only (addendum A3)

    def __post_init__(self):
        if self.type == "score" and self.ordinals is None:
            self.ordinals = list(range(len(self.candidates)))


@dataclass
class DecisionResult:
    type: str
    distribution: dict[str, float]
    expected: float | None = None
    extras: dict = field(default_factory=dict)


class Decision:
    """Batched decision facade over ElectraDecisionModel."""

    def __init__(self, model: ElectraDecisionModel, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def decide(self, state, questions: list[QuestionSpec], device=None) -> list[DecisionResult]:
        """Evaluate many isolated questions against one shared state —
        identical results to running each question alone (sec 48.5)."""
        self.model.eval()
        dev = device or next(self.model.parameters()).device
        inp = build_decision_inputs(
            [
                {
                    "state": state,
                    "questions": [
                        {
                            "type": q.type,
                            "instruction": q.instruction,
                            "candidates": q.candidates,
                        }
                        for q in questions
                    ],
                }
            ],
            self.tokenizer,
            device=dev,
        )
        out = self.model(
            state_ids=inp.state_ids,
            state_cu=inp.state_cu,
            question_ids=inp.question_ids,
            question_cu=inp.question_cu,
            question_state_index=inp.question_state_index,
            candidate_ids=inp.candidate_ids,
            candidate_cu=inp.candidate_cu,
            candidate_question_index=inp.candidate_question_index,
            primitive_index=inp.primitive_index,
            apply_temperature=True,
        )
        probs = out.probs()
        results = []
        for qi, q in enumerate(questions):
            s, e = int(out.cand_cu[qi]), int(out.cand_cu[qi + 1])
            p = probs[s:e].tolist()
            dist = dict(zip(q.candidates, p, strict=True))
            res = DecisionResult(type=q.type, distribution=dist)
            if q.type == "score":
                ordinals = q.ordinals or list(range(len(p)))
                res.expected = sum(o * pi for o, pi in zip(ordinals, p, strict=True))
            if q.type == "noul":
                # true candidate is the last of the implicit pair
                res.extras["p_true"] = p[-1]
            results.append(res)
        return results

    def choice(
        self, state, instruction: str, candidates: list[str], device=None
    ) -> dict[str, float]:
        return self.decide(state, [QuestionSpec("choice", instruction, candidates)], device)[
            0
        ].distribution

    def noul(self, state, proposition: str, device=None) -> float:
        res = self.decide(
            state,
            [QuestionSpec("noul", proposition, ["false", "true"])],
            device,
        )[0]
        return res.extras["p_true"]

    def score(
        self,
        state,
        instruction: str,
        levels: list[str],
        ordinals: list[int] | None = None,
        device=None,
    ) -> tuple[float, dict[str, float]]:
        res = self.decide(state, [QuestionSpec("score", instruction, levels, ordinals)], device)[0]
        return res.expected, res.distribution
