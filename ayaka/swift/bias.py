"""Calibration-only convex NLL fitting of canonical letter position offsets.

The objective is mean pre-commit NLL + lambda/2 * sum(b**2). Temperatures
stay frozen. Full position vectors have zero mean at the optimum; Noul uses
one true-logit intercept with the false logit fixed. No tensors or inference.
"""

from __future__ import annotations

import math

from .losses import target_distribution
from .policy import Policy, noul_labels
from .provenance import group_reads

# Predeclared, independent of calibration/dev performance.
BIAS_L2 = 0.001
BIAS_MIN_QUESTIONS = 30
BIAS_MAX_ITERATIONS = 2000
BIAS_GRADIENT_TOLERANCE = 1e-10


def fit_letter_bias(rows: list[dict], policy: Policy) -> dict:
    """Return fitted parameters/diagnostics; fitting alone never adopts a bias."""
    groups = group_reads(rows, "calibration", require_variants=False)
    if set(groups) != {policy.prompt_variant} or policy.letter_bias:
        raise ValueError("fit bias for the accepted variant with no existing bias")
    buckets = {}
    for row in groups[policy.prompt_variant]:
        buckets.setdefault((row["type"], len(row["labels"])), []).append(row)
    fitted, reports = {}, {}
    for (kind, count), bucket in sorted(buckets.items()):
        name = f"{kind}/{count}"
        if len(bucket) < BIAS_MIN_QUESTIONS:
            reports[name] = {"n": len(bucket), "fitted": False, "reason": "too_few_questions"}
            continue
        temperature = getattr(policy, f"t_{kind}")
        samples = []
        for row in bucket:
            target = target_distribution(row)
            labels = row["labels"]
            if kind == "noul":
                false, true = noul_labels(row["raw_probs"])
                labels = [false, true]
            masses = row["candidate_log_masses"]
            peak = max(masses.values())
            samples.append(
                ([masses[label] - peak for label in labels], [target[label] for label in labels])
            )
        dimension = 1 if kind == "noul" else count
        bias = [0.0] * dimension

        def objective_gradient(
            values, *, dimension=dimension, samples=samples, kind=kind, temperature=temperature
        ):
            loss = 0.0
            gradient = [0.0] * dimension
            for masses, target in samples:
                offsets = [0.0, values[0]] if kind == "noul" else values
                logits = [
                    (mass + offset) / temperature
                    for mass, offset in zip(masses, offsets, strict=True)
                ]
                peak = max(logits)
                shifted = [value - peak for value in logits]
                if any(not math.isfinite(value) for value in shifted):
                    raise ValueError("bias-scaled logit span exceeds finite arithmetic")
                partition = math.log(math.fsum(math.exp(value) for value in shifted))
                loss += math.fsum(
                    t * (partition - z) for t, z in zip(target, shifted, strict=True) if t > 0
                )
                residual = [
                    (math.exp(z - partition) - t) / temperature
                    for z, t in zip(shifted, target, strict=True)
                ]
                if kind == "noul":
                    gradient[0] += residual[1]
                else:
                    gradient = [g + r for g, r in zip(gradient, residual, strict=True)]
            loss = loss / len(samples) + BIAS_L2 / 2 * math.fsum(v * v for v in values)
            gradient = [
                g / len(samples) + BIAS_L2 * v for g, v in zip(gradient, values, strict=True)
            ]
            return loss, gradient

        initial, _ = objective_gradient(bias)
        # Global softmax Hessian bound; deterministic convergent gradient descent.
        step = 1 / ((0.25 if kind == "noul" else 0.5) / temperature**2 + BIAS_L2)
        for _iteration in range(BIAS_MAX_ITERATIONS):
            loss, gradient = objective_gradient(bias)
            if max(abs(value) for value in gradient) <= BIAS_GRADIENT_TOLERANCE:
                break
            bias = [v - step * g for v, g in zip(bias, gradient, strict=True)]
        loss, gradient = objective_gradient(bias)
        if loss > initial + 1e-12:
            raise ValueError("bias optimizer failed to improve its convex objective")
        fitted.setdefault(kind, {})[str(count)] = bias
        reports[name] = {
            "n": len(bucket),
            "fitted": True,
            "initial_objective": initial,
            "objective": loss,
            "iterations": _iteration + 1,
            "gradient_max_abs": max(abs(value) for value in gradient),
        }
    return {
        "letter_bias": fitted,
        "buckets": reports,
        "l2": BIAS_L2,
        "minimum_questions": BIAS_MIN_QUESTIONS,
        "objective": "mean_NLL_plus_L2",
        "temperature_frozen": True,
        "commit_in_NLL": False,
    }
