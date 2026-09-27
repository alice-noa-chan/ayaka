"""Fixed fusion policy; no task families, reference answers or gold inputs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvidencePolicy:
    readout: str = "executed"
    weight: float = 0.5
    boolean_confidence: float | None = None
    baseline_cutoff: float = 0.9
    recover_ids: bool = False
    gate: str = "broad"  # broad: needs_evidence; calculation: needs_calculation

    def __post_init__(self):
        if self.readout not in ("quotes", "executed", "native", "reasoned", "baseline"):
            raise ValueError("unknown evidence readout")
        if self.gate not in ("broad", "calculation"):
            raise ValueError("unknown evidence gate")
        if not 0 <= self.weight <= 1 or not 0 <= self.baseline_cutoff <= 1:
            raise ValueError("invalid probability weight/cutoff")
        if self.boolean_confidence is not None and not 0.5 <= self.boolean_confidence < 1:
            raise ValueError("invalid Boolean confidence")

    def to_dict(self):
        return asdict(self)


def fuse_probabilities(baseline, auxiliary, policy, *, computed_label=None, usable=False):
    """Return a calibrated-shaped distribution; do not assert calibration.

    A computed predicate is still a model-selected expression. Its validity is
    a mechanical gate, not proof of semantic correctness or probability quality.
    """
    if not baseline or any(not math.isfinite(p) or p < 0 for p in baseline.values()):
        raise ValueError("invalid baseline probabilities")
    if abs(sum(baseline.values()) - 1) > 1e-4:
        raise ValueError("baseline distribution must sum to one")
    if (
        not usable
        or policy.readout == "baseline"
        or max(baseline.values()) > policy.baseline_cutoff
    ):
        return dict(baseline), "baseline"
    if set(auxiliary) != set(baseline):
        raise ValueError("candidate distribution mismatch")
    if (
        any(not math.isfinite(p) or p < 0 for p in auxiliary.values())
        or abs(sum(auxiliary.values()) - 1) > 1e-4
    ):
        raise ValueError("invalid auxiliary probabilities")
    routed = dict(auxiliary)
    reason = policy.readout
    if policy.boolean_confidence is not None and computed_label is not None:
        if len(baseline) != 2 or computed_label not in baseline:
            raise ValueError("computed predicate requires aligned binary candidates")
        routed = {
            lb: policy.boolean_confidence if lb == computed_label else 1 - policy.boolean_confidence
            for lb in baseline
        }
        reason = "computed_predicate"
    out = {lb: (1 - policy.weight) * baseline[lb] + policy.weight * routed[lb] for lb in baseline}
    return out, reason
