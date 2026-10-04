"""Per-primitive temperature scaling and optional Noul commitment."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .prompt import validate_prompt_variant

COMMIT_LO = 0.199
COMMIT_HI = 0.801
FALSE_LABELS = {"false", "no", "0"}
TRUE_LABELS = {"true", "yes", "1"}


def normalize(probs: dict[str, float]) -> dict[str, float]:
    if not probs or any(not math.isfinite(p) or p < 0 for p in probs.values()):
        raise ValueError("probabilities must be finite, nonnegative and nonempty")
    total = math.fsum(probs.values())
    if total <= 0:
        raise ValueError("probabilities need positive mass")
    return {label: p / total for label, p in probs.items()}


def temperature_scale(probs: dict[str, float], temperature: float) -> dict[str, float]:
    """Apply p**(1/T), stably, preserving zero mass."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    probs = normalize(probs)
    if temperature == 1:
        return probs
    logs = {label: math.log(p) / temperature for label, p in probs.items() if p > 0}
    peak = max(logs.values())
    return normalize(
        {label: math.exp(logs[label] - peak) if p > 0 else 0.0 for label, p in probs.items()}
    )


def noul_labels(probs: dict[str, float]) -> tuple[str, str]:
    if len(probs) != 2:
        raise ValueError("noul needs false and true labels")
    false = next((label for label in probs if label.lower() in FALSE_LABELS), None)
    true = next((label for label in probs if label.lower() in TRUE_LABELS), None)
    if false is None or true is None:
        raise ValueError("noul labels must be false/true or no/yes")
    return false, true


@dataclass
class Policy:
    """A None margin disables commit; legacy noul_commit=False also disables it."""

    t_choice: float = 1.0
    t_noul: float = 1.0
    t_score: float = 1.0
    noul_commit: bool = True
    commit_lo: float = COMMIT_LO
    commit_hi: float = COMMIT_HI
    fitted_on: str = "unfitted"
    commit_margin: float | None = None
    search: dict | None = None
    prompt_variant: str = "min"
    promotable: bool = True

    def __post_init__(self) -> None:
        validate_prompt_variant(self.prompt_variant)
        if any(not math.isfinite(t) or t <= 0 for t in (self.t_choice, self.t_noul, self.t_score)):
            raise ValueError("temperatures must be finite and positive")
        if not 0 <= self.commit_lo <= 0.2 or not 0.8 <= self.commit_hi <= 1:
            raise ValueError("commit boundaries must be in [0,.2] and [.8,1]")
        if not isinstance(self.noul_commit, bool) or not isinstance(self.fitted_on, str):
            raise ValueError("noul_commit must be boolean and fitted_on must be text")
        if self.commit_margin is not None and (
            not math.isfinite(self.commit_margin) or self.commit_margin < 0
        ):
            raise ValueError("commit_margin must be None or finite and nonnegative")
        # Legacy constructors/files still disable commit via noul_commit=False.
        if not self.noul_commit:
            self.commit_margin = None
        self.noul_commit = self.commit_margin is not None

    def apply(
        self,
        question_type: str,
        label_probs: dict[str, float],
        *,
        candidate_log_masses: dict[str, float] | None = None,
    ) -> dict[str, float]:
        temperatures = {"choice": self.t_choice, "noul": self.t_noul, "score": self.t_score}
        if question_type not in temperatures:
            raise ValueError(f"unknown question type: {question_type!r}")
        if candidate_log_masses is None:
            probs = temperature_scale(label_probs, temperatures[question_type])
        else:
            if set(candidate_log_masses) != set(label_probs) or any(
                not math.isfinite(value) for value in candidate_log_masses.values()
            ):
                raise ValueError("candidate log masses must align and be finite")
            peak = max(candidate_log_masses.values())
            shifted = {
                label: (mass - peak) / temperatures[question_type]
                for label, mass in candidate_log_masses.items()
            }
            if any(not math.isfinite(value) for value in shifted.values()):
                raise ValueError("temperature-scaled logit span exceeds finite arithmetic")
            probs = normalize({label: math.exp(shifted[label]) for label in label_probs})
        if question_type == "noul":
            false, true = noul_labels(probs)
            p = probs[true]
            distance = abs(p - 0.5)
            if (
                self.commit_margin is not None
                and 0.2 < p < 0.8
                # Preserve inclusive equality through temperature roundoff.
                and (
                    distance >= self.commit_margin
                    or math.isclose(distance, self.commit_margin, rel_tol=0, abs_tol=math.ulp(p))
                )
            ):
                # Exactly 0.5 commits to Yes; threshold endpoints already commit.
                p = self.commit_hi if p >= 0.5 else self.commit_lo
                probs = {label: p if label == true else 1 - p for label in probs}
        return probs

    def decide(
        self,
        question_type: str,
        label_probs: dict[str, float],
        *,
        candidate_log_masses: dict[str, float] | None = None,
    ) -> dict:
        probs = self.apply(question_type, label_probs, candidate_log_masses=candidate_log_masses)
        if question_type == "noul":
            return {"type": "noul", "noul": probs[noul_labels(probs)[1]]}
        if question_type == "choice":
            return {
                "type": "choice",
                "choice": max(probs, key=probs.__getitem__),
                "probabilities": probs,
            }
        expected = sum(int(label) * p for label, p in probs.items())
        return {"type": "score", "score": expected, "probabilities": probs}

    def save(self, path: str | Path = "policy.json") -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path = "policy.json") -> Policy:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if "commit_margin" not in values and values.get("noul_commit") is True:
            values["commit_margin"] = 0.0
        return cls(**values)
