"""Validated targets and cross entropy without invented probability floors."""

from __future__ import annotations

import math

from ayaka.eval.read_artifact import resolve_target

from .policy import normalize


def target_distribution(row: dict) -> dict[str, float]:
    labels = row.get("labels", list(row["raw_probs"]))
    if len(labels) < 2 or len(set(labels)) != len(labels) or set(labels) != set(row["raw_probs"]):
        raise ValueError("target labels and raw probabilities must align with unique options")
    gold = row["gold"]
    distribution = row.get("gold_distribution")
    if len(labels) <= 26:
        return resolve_target(labels, gold if isinstance(gold, dict) else str(gold), distribution)
    # The native helper deliberately supports only single-pass (2..26) reads.
    target = distribution if distribution is not None else gold
    if not isinstance(target, dict):
        target = {str(target): 1.0}
    if (
        set(target) - set(labels)
        or any(
            type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in target.values()
        )
        or not math.isclose(math.fsum(target.values()), 1.0, rel_tol=0, abs_tol=1e-9)
    ):
        raise ValueError("target distribution must align and be finite, nonnegative and normalized")
    if isinstance(gold, dict) and distribution is not None and gold != distribution:
        raise ValueError("two explicit target distributions disagree")
    return {label: target.get(label, 0.0) for label in labels}


def row_nll(row: dict, temperature: float = 1.0) -> float:
    """Same logspace CE semantics as read_artifact.logspace_nll, for Swift rows."""
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    target = target_distribution(row)
    masses = row.get("candidate_log_masses")
    if masses is None:
        probabilities = normalize(row["raw_probs"])
        masses = {label: math.log(p) if p > 0 else -math.inf for label, p in probabilities.items()}
    elif set(masses) != set(target) or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in masses.values()
    ):
        raise ValueError("candidate log masses must align and be finite")
    peak = max(masses.values())
    shifted = {label: (value - peak) / temperature for label, value in masses.items()}
    if row.get("candidate_log_masses") is not None and any(
        not math.isfinite(value) for value in shifted.values()
    ):
        raise ValueError("temperature-scaled logit span exceeds finite arithmetic")
    partition = math.log(math.fsum(math.exp(value) for value in shifted.values()))
    return math.fsum(
        mass * (partition - shifted[label]) for label, mass in target.items() if mass > 0
    )
