"""Canonical Ayaka training schema (docs.md section 31 + A3/A5).

Every source dataset is converted into this schema before the model
ever sees it — raw dataset formats are never exposed to the model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

PRIMITIVES = ("noul", "choice", "score")

EVIDENCE_STATES = ("intact", "deleted", "partial", "contradictory", "irrelevant_pad")


@dataclass
class Candidate:
    id: str
    description: str
    ordinal: int | float | Decimal | None = None  # score levels only (A3)
    is_nota: bool = False  # explicit none-of-the-above marker (A5)


@dataclass
class Question:
    id: str
    type: str  # noul | choice | score
    instruction: str
    candidates: list[Candidate]
    target_distribution: dict[str, float]  # candidate id -> prob

    def __post_init__(self):
        if self.type not in PRIMITIVES:
            raise ValueError(f"unknown primitive type: {self.type}")
        ids = {c.id for c in self.candidates}
        if len(ids) != len(self.candidates):
            raise ValueError("candidate ids must be unique")
        unknown = set(self.target_distribution) - ids
        if unknown:
            raise ValueError(f"target references missing candidates: {unknown}")
        if any(
            not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 or p > 1
            for p in self.target_distribution.values()
        ):
            raise ValueError("target probabilities must be finite values in [0, 1]")
        total = sum(self.target_distribution.values())
        if self.target_distribution and abs(total - 1.0) > 1e-3:
            raise ValueError(f"target distribution sums to {total}, not 1")
        if self.type == "score":
            ords = [c.ordinal for c in self.candidates]
            if any(o is None for o in ords):
                raise ValueError("score candidates require ordinal")
            if any(
                isinstance(o, bool)
                or not isinstance(o, (int, float, Decimal))
                or (isinstance(o, float) and not math.isfinite(o))
                or (isinstance(o, Decimal) and not o.is_finite())
                for o in ords
            ):
                raise ValueError("score ordinal must be a finite numeric value")
            if len(set(ords)) != len(ords):
                raise ValueError("score ordinals must be unique")

    @classmethod
    def noul(
        cls,
        id: str,
        proposition: str,
        p_true: float,
        false_desc: str = "false",
        true_desc: str = "true",
    ) -> Question:
        """Noul as an implicit [false, true] pair (A3)."""
        return cls(
            id=id,
            type="noul",
            instruction=proposition,
            candidates=[Candidate("false", false_desc), Candidate("true", true_desc)],
            target_distribution={"false": 1.0 - p_true, "true": p_true},
        )


@dataclass
class Sample:
    """One state + isolated questions — the shared-state unit."""

    state: Any  # JsonValue or plain text
    questions: list[Question]
    metadata: dict = field(default_factory=dict)
    # metadata keys used downstream: language, source, task_family,
    # source_family, source_example_id, evidence_state,
    # translation_of, derived_from, generator_template_id,
    # evidence_blocks (router supervision)

    @property
    def evidence_state(self) -> str:
        return self.metadata.get("evidence_state", "intact")

    def to_json(self) -> dict:
        result = asdict(self)
        for question in result["questions"]:
            for candidate in question["candidates"]:
                if isinstance(candidate["ordinal"], Decimal):
                    candidate["ordinal"] = str(candidate["ordinal"])
        return result

    @classmethod
    def from_json(cls, d: dict) -> Sample:
        qs = []
        for q in d["questions"]:
            cands = []
            for c in q["candidates"]:
                candidate = Candidate(**c)
                if q["type"] == "score" and isinstance(candidate.ordinal, str):
                    try:
                        ordinal = Decimal(candidate.ordinal)
                    except InvalidOperation as exc:
                        raise ValueError("score ordinal must be a finite numeric value") from exc
                    if not ordinal.is_finite():
                        raise ValueError("score ordinal must be a finite numeric value")
                    candidate.ordinal = ordinal
                cands.append(candidate)
            qs.append(
                Question(
                    id=q["id"],
                    type=q["type"],
                    instruction=q["instruction"],
                    candidates=cands,
                    target_distribution=q["target_distribution"],
                )
            )
        return cls(state=d["state"], questions=qs, metadata=d.get("metadata", {}))


def one_hot(candidates: list[Candidate], correct_id: str) -> dict[str, float]:
    return {c.id: 1.0 if c.id == correct_id else 0.0 for c in candidates}


def uniform(candidates: list[Candidate]) -> dict[str, float]:
    n = len(candidates)
    return {c.id: 1.0 / n for c in candidates}
