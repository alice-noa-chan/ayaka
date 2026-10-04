"""Local JevBench v1.5 axes for a single supplied split."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from statistics import mean

from .losses import row_nll, target_distribution
from .policy import Policy, noul_labels

TIER_WEIGHTS = {"easy": 0.10, "standard": 0.20, "judge": 0.30, "hard": 0.40}
TYPES = ("choice", "noul", "score")


@dataclass(frozen=True)
class ScoredItem:
    type: str
    tier: str
    labels: list[str]
    probs: dict[str, float]
    gold: str
    soft_target: dict[str, float] | None

    @property
    def prediction(self) -> str:
        return max(self.labels, key=self.probs.__getitem__)


def prepare_rows(rows: list[dict], policy: Policy | None = None) -> list[ScoredItem]:
    policy = policy or Policy(noul_commit=False)
    items = []
    for row in rows:
        kind = row["type"]
        if kind not in TYPES:
            raise ValueError(f"unknown type: {kind}")
        tier = row.get("tier", "standard")
        tier = "standard" if tier == "original" else tier
        if tier not in TIER_WEIGHTS:
            raise ValueError(f"unknown tier: {tier}")
        labels = row["labels"]
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("labels need at least two unique options")
        if set(labels) != set(row["raw_probs"]):
            raise ValueError("raw_probs and labels must align")
        probs = policy.apply(
            kind,
            {label: row["raw_probs"][label] for label in labels},
            candidate_log_masses=row.get("candidate_log_masses"),
        )
        gold = row["gold"]
        target = target_distribution(row)
        soft = (
            target if isinstance(gold, dict) or row.get("gold_distribution") is not None else None
        )
        if isinstance(gold, dict):
            gold = max(labels, key=lambda label: gold.get(label, 0.0))
        gold = str(gold)
        if gold not in labels:
            raise ValueError("gold label absent from labels")
        items.append(ScoredItem(kind, tier, labels, probs, gold, soft))
    return items


def choice_cc(items: list[ScoredItem]) -> float:
    accuracy = mean(item.prediction == item.gold for item in items)
    chance = mean(1 / len(item.labels) for item in items)
    return 100 * (accuracy - chance) / (1 - chance)


def noul_cc(items: list[ScoredItem]) -> float:
    correct = []
    for item in items:
        false, true = noul_labels(item.probs)
        p = item.probs[true]
        prediction = false if p <= 0.2 else true if p >= 0.8 else None
        correct.append(prediction == item.gold)
    return 100 * (mean(correct) - 0.5) / 0.5


def score_cc(items: list[ScoredItem]) -> float:
    errors, chances = [], []
    for item in items:
        k = len(item.labels)
        gold = item.labels.index(item.gold)
        prediction = sum(i * item.probs[label] for i, label in enumerate(item.labels))
        errors.append(abs(prediction - gold) / (k - 1))
        chances.append(mean(abs(level - gold) / (k - 1) for level in range(k)))
    return 100 * (1 - mean(errors) / mean(chances))


def intelligence(items: list[ScoredItem]) -> dict:
    """Renormalize tier weights within each type; require all three for I."""
    per_type, per_tier = {}, {}
    functions = {"choice": choice_cc, "noul": noul_cc, "score": score_cc}
    for kind in TYPES:
        tiers = {}
        for tier in TIER_WEIGHTS:
            subset = [item for item in items if item.type == kind and item.tier == tier]
            if subset:
                tiers[tier] = {"n": len(subset), "CC": functions[kind](subset)}
        if tiers:
            weight = sum(TIER_WEIGHTS[tier] for tier in tiers)
            per_type[kind] = (
                sum(TIER_WEIGHTS[tier] * report["CC"] for tier, report in tiers.items()) / weight
            )
            per_tier[kind] = tiers
    missing = [kind for kind in TYPES if kind not in per_type]
    return {
        "per_type_tier": per_tier,
        "per_type_CC": per_type,
        "I": mean(per_type.values()) if not missing else None,
        "missing_types": missing,
    }


def ece(confidences: list[float], outcomes: list[float]) -> float:
    """Ten equal-width bins, including p=1 in bin 9."""
    if len(confidences) != len(outcomes) or not confidences:
        raise ValueError("ECE needs nonempty aligned confidences and outcomes")
    counts = [0] * 10
    conf_sums = [0.0] * 10
    outcome_sums = [0.0] * 10
    for p, outcome in zip(confidences, outcomes, strict=True):
        if not 0 <= p <= 1 or not 0 <= outcome <= 1:
            raise ValueError("ECE inputs must be in [0,1]")
        index = min(int(p * 10), 9)
        counts[index] += 1
        conf_sums[index] += p
        outcome_sums[index] += outcome
    return sum(abs(conf_sums[i] - outcome_sums[i]) for i in range(10)) / len(confidences)


def ece_score(value: float) -> float:
    return 100 * (1 - value / 0.5)


def top_ece(items: list[ScoredItem]) -> float:
    return ece(
        [item.probs[item.prediction] for item in items],
        [float(item.prediction == item.gold) for item in items],
    )


def normalized_rps(item: ScoredItem) -> float:
    """Sum squared CDF errors through K-2, divided by K-1."""
    cdf = 0.0
    total = 0.0
    gold_index = item.labels.index(item.gold)
    for index, label in enumerate(item.labels[:-1]):
        cdf += item.probs[label]
        total += (cdf - float(index >= gold_index)) ** 2
    return total / (len(item.labels) - 1)


def calibration(items: list[ScoredItem]) -> dict:
    """Choice hard-tier top-label ECE plus all available Choice gold TVD.

    If the supplied split has no hard Choice items, use all its Choice items
    for ECE and mark that fallback. Noul and Score always use all their items.
    """
    parts, details = {}, {}
    choice_ece_scope = None
    for kind in TYPES:
        subset = [item for item in items if item.type == kind]
        if not subset:
            continue
        if kind == "noul":
            value = ece(
                [item.probs[noul_labels(item.probs)[1]] for item in subset],
                [float(item.gold == noul_labels(item.probs)[1]) for item in subset],
            )
            parts[kind] = ece_score(value)
            details[kind] = {"ECE": value}
        elif kind == "choice":
            hard = [item for item in subset if item.tier == "hard"]
            ece_items = hard or subset
            choice_ece_scope = "hard" if hard else "all_choice_fallback"
            value = top_ece(ece_items)
            tvds = [
                0.5
                * sum(
                    abs(item.probs[label] - item.soft_target.get(label, 0.0))
                    for label in item.labels
                )
                for item in subset
                if item.soft_target is not None
            ]
            tvd = mean(tvds) if tvds else None
            parts[kind] = (
                (ece_score(value) + 100 * (1 - tvd)) / 2 if tvd is not None else ece_score(value)
            )
            details[kind] = {
                "ECE": value,
                "ece_score": ece_score(value),
                "mean_TVD": tvd,
                "soft_target_n": len(tvds),
                "n": len(subset),
                "n_ece": len(ece_items),
                "n_ece_hard": len(hard),
            }
        else:
            value = top_ece(subset)
            nrps = mean(normalized_rps(item) for item in subset)
            parts[kind] = (100 * (1 - nrps) + ece_score(value)) / 2
            details[kind] = {"top_ECE": value, "nRPS": nrps}
    return {
        "calibration_parts": parts,
        "calibration_details": details,
        "choice_ece_scope": choice_ece_scope,
        "C": mean(parts.values()) if len(parts) == 3 else None,
    }


def speed_score(seconds: float) -> float:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("latency must be finite and positive")
    return 100 - 20 * math.log10(seconds / 0.1)


def speed_axis(p50_raw: float, p95_raw: float) -> float:
    if not 0 <= p50_raw <= p95_raw or not math.isfinite(p95_raw):
        raise ValueError("raw latencies need 0 <= p50 <= p95 < infinity")
    return mean([speed_score(2 * p50_raw + 0.15), speed_score(2 * p95_raw + 0.15)])


def cost_score(usd_per_1000_decisions: float) -> float:
    if not math.isfinite(usd_per_1000_decisions) or usd_per_1000_decisions <= 0:
        raise ValueError("cost must be finite and positive")
    return 100 - 30 * math.log10(usd_per_1000_decisions / 0.001)


def decision_cost(rows: list[dict], usd_in_per_m: float, usd_out_per_m: float) -> float:
    if not rows or any(not math.isfinite(p) or p < 0 for p in (usd_in_per_m, usd_out_per_m)):
        raise ValueError("cost needs reads and finite nonnegative prices")
    return (
        mean(row["input_tokens"] for row in rows) * usd_in_per_m
        + mean(row["output_tokens"] for row in rows) * usd_out_per_m
    ) / 1000


def composite(
    intelligence: float, calibration: float, speed: float, cost: float, *, view: str = "A"
) -> float:
    """Weighted harmonic mean with low I, S, Cost penalties."""
    if view not in ("A", "B"):
        raise ValueError("view must be A or B")
    axes = [intelligence, calibration, speed, cost]
    if any(not math.isfinite(axis) for axis in axes):
        raise ValueError("axes must be finite")
    if min(axes) <= 0:
        return 0.0
    weights = [0.25] * 4 if view == "A" else [0.4, 0.2, 0.2, 0.2]
    result = 1 / sum(weight / axis for weight, axis in zip(weights, axes, strict=True))
    for axis in (intelligence, speed, cost):
        if axis < 50:
            result *= (axis / 50) ** 2
    return result


def probability_metrics(rows: list[dict], items: list[ScoredItem], policy: Policy) -> dict:
    """NLL/Brier and all-item ECE of emitted probabilities, per primitive."""
    result = {}
    for kind in TYPES:
        pairs = [(row, item) for row, item in zip(rows, items, strict=True) if item.type == kind]
        if not pairs:
            continue
        losses, briers, confidences, outcomes = [], [], [], []
        for row, item in pairs:
            target = target_distribution(row)
            temperature = getattr(policy, f"t_{kind}")
            ordered_probs = {label: row["raw_probs"][label] for label in item.labels}
            scaled = replace(policy, noul_commit=False, commit_margin=None).apply(
                kind, ordered_probs, candidate_log_masses=row.get("candidate_log_masses")
            )
            if item.probs == scaled:
                masses = policy.biased_log_masses(
                    kind, ordered_probs, row.get("candidate_log_masses")
                )
                losses.append(row_nll({**row, "candidate_log_masses": masses}, temperature))
            else:
                losses.append(
                    row_nll({**row, "raw_probs": item.probs, "candidate_log_masses": None})
                )
            briers.append(
                math.fsum((item.probs[label] - target[label]) ** 2 for label in item.labels)
            )
            label = noul_labels(item.probs)[1] if kind == "noul" else item.prediction
            confidences.append(item.probs[label])
            outcomes.append(target[label])
        infinite = sum(not math.isfinite(value) for value in losses)
        result[kind] = {
            "n": len(pairs),
            "NLL": None if infinite else mean(losses),
            "nll_infinite_n": infinite,
            "Brier": mean(briers),
            "ECE": ece(confidences, outcomes),
            "hard_n": sum(item.soft_target is None for _, item in pairs),
            "soft_n": sum(item.soft_target is not None for _, item in pairs),
        }
    return result


def ordinal_diagnostics(items: list[ScoredItem]) -> dict | None:
    scores = [item for item in items if item.type == "score"]
    if not scores:
        return None
    errors, chances = [], []
    for item in scores:
        values = {label: int(label) for label in item.labels}
        span = max(values.values()) - min(values.values())
        gold = (
            sum(values[label] * item.soft_target.get(label, 0) for label in item.labels)
            if item.soft_target is not None
            else values[item.gold]
        )
        predicted = sum(values[label] * item.probs[label] for label in item.labels)
        errors.append(abs(predicted - gold) / span)
        chances.append(mean(abs(value - gold) / span for value in values.values()))
    return {
        "n": len(scores),
        "nMAE": mean(errors),
        "chance_nMAE": mean(chances),
        "CC": 100 * (1 - mean(errors) / mean(chances)),
        "metric": "ordinal_value",
    }


def score_reads(
    rows: list[dict], policy: Policy | None = None, *, include_grouped: bool = False
) -> dict:
    policy = policy or Policy()
    grouped = [row for row in rows if row.get("readout") == "grouped_approx"]
    if not include_grouped:
        rows = [row for row in rows if row.get("readout") != "grouped_approx"]
    items = prepare_rows(rows, policy)
    if not items and not grouped:
        raise ValueError("cannot evaluate empty reads")
    result = {
        "n": len(items),
        "split": "supplied",
        **intelligence(items),
        **calibration(items),
        "per_type_probability": probability_metrics(rows, items, policy),
        "ordinal_value_diagnostics": ordinal_diagnostics(items),
        "score_competence_metric": "jevbench_ordinal_position",
        "readout_scope": "including_grouped" if include_grouped else "single_pass",
        "grouped_excluded_n": 0 if include_grouped else len(grouped),
        "total_passes": sum(row.get("passes", 1) for row in rows),
    }
    if grouped and not include_grouped:
        result["grouped_approx"] = score_reads(grouped, policy, include_grouped=True)
    return result
