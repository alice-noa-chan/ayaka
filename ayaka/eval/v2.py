"""Versioned local typed diagnostics; never a sealed-inclusive JevBench score."""

import hashlib
import json
import math
import random
from collections import defaultdict

TIERS = {"easy": 0.1, "standard": 0.2, "judge": 0.3, "hard": 0.4}
METHOD = "jevbench-v1.5-headline-A-equal-types-local"


def nll(probs, target):
    return -sum(y * math.log(max(p, 1e-12)) for p, y in zip(probs, target, strict=True))


def typed_row(spec, probs, target):
    if (
        len(probs) != len(target)
        or any(not math.isfinite(p) or p < 0 for p in probs)
        or abs(sum(probs) - 1) > 1e-4
    ):
        raise ValueError(
            "probabilities must be a finite normalized distribution aligned with targets"
        )
    if (
        len(probs) < 2
        or any(y < 0 or not math.isfinite(y) for y in target)
        or abs(sum(target) - 1) > 1e-4
    ):
        raise ValueError("targets must be a finite normalized distribution")
    pred = max(range(len(probs)), key=probs.__getitem__)
    row = {
        "type": spec.type,
        "nll": nll(probs, target),
        "brier": sum((p - y) ** 2 for p, y in zip(probs, target, strict=True)),
        "confidence": probs[pred],
        "correct": target[pred],
        "chance": 1 / len(probs),
    }
    if spec.type == "noul":
        selected = 0 if probs[1] <= 0.2 else (1 if probs[1] >= 0.8 else None)
        row.update(
            correct=0 if selected is None else target[selected],
            abstained=selected is None,
            confidence=probs[1],
            calibration_target=target[1],
            chance=0.5,
            brier=(probs[1] - target[1]) ** 2,
        )
    elif spec.type == "score":
        order = sorted(range(len(probs)), key=lambda i: spec.ordinals[i])
        p, y = [probs[i] for i in order], [target[i] for i in order]
        k = len(p)
        expected = sum(i * v for i, v in enumerate(p))
        # Discrete gold distribution: average error over gold positions, not
        # error to its mean (which would reward an uninformative midpoint).
        error = sum(v * abs(expected - i) for i, v in enumerate(y)) / (k - 1)
        chance = sum(v * sum(abs(j - i) for j in range(k)) / k for i, v in enumerate(y)) / (k - 1)
        cp, cy, rps = 0.0, 0.0, 0.0
        for pi, yi in zip(p[:-1], y[:-1], strict=True):
            cp, cy = cp + pi, cy + yi
            rps += (cp - cy) ** 2 / (k - 1)
        row.update(expected_position=expected, nmae=error, nmae_chance=chance, rps=rps)
    row.setdefault("calibration_target", target[pred])
    return row


def ece(rows, bins=10):
    total = len(rows)
    value = 0.0
    for b in range(bins):
        group = [r for r in rows if min(int(r["confidence"] * bins), bins - 1) == b]
        if group:
            value += abs(sum(r["confidence"] - r["calibration_target"] for r in group)) / total
    return value


def competence(rows):
    if rows[0]["type"] == "score":
        return 100 * (1 - sum(r["nmae"] for r in rows) / sum(r["nmae_chance"] for r in rows))
    chance = sum(r["chance"] for r in rows) / len(rows)
    accuracy = sum(r["correct"] for r in rows) / len(rows)
    return 100 * (accuracy - chance) / (1 - chance)


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    at = (len(ordered) - 1) * q
    low = int(at)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (at - low)


def summarize(rows):
    by_type = {}
    for kind in ("choice", "noul", "score"):
        subset = [r for r in rows if r["type"] == kind]
        if not subset:
            continue
        tiers = {t: [r for r in subset if r.get("tier", "standard") == t] for t in TIERS}
        weight = sum(TIERS[t] for t, rs in tiers.items() if rs)
        cc = sum(TIERS[t] * competence(rs) for t, rs in tiers.items() if rs) / weight
        by_type[kind] = {
            "n": len(subset),
            "cc": cc,
            "nll": sum(r["nll"] for r in subset) / len(subset),
            "ece": ece(subset),
            "brier": sum(r["brier"] for r in subset) / len(subset),
        }
        if kind == "score":
            by_type[kind]["rps"] = sum(r["rps"] for r in subset) / len(subset)
        if kind == "noul":
            by_type[kind]["abstentions"] = sum(r["abstained"] for r in subset)
    if not by_type:
        raise ValueError("cannot summarize an empty evaluation")
    latencies = [r["latency_s"] for r in rows if "latency_s" in r]
    return {
        "method": METHOD,
        "official_composite": None,
        "sealed": False,
        "n": len(rows),
        "by_type": by_type,
        "cc_equal_types": sum(v["cc"] for v in by_type.values()) / len(by_type),
        "proper_loss": sum(v.get("rps", v["nll"]) for v in by_type.values()) / len(by_type),
        "p50_s": percentile(latencies, 0.5),
        "p95_s": percentile(latencies, 0.95),
        "reasoning_tokens": sum(r.get("reasoning_tokens", 0) for r in rows),
    }


def paired_report(direct, reasoned, replicates=2000):
    if [r["id"] for r in direct] != [r["id"] for r in reasoned]:
        raise ValueError("paired results must have identical item order")
    clusters = defaultdict(list)
    for i, row in enumerate(direct):
        clusters[row.get("cluster_id", row["id"])].append(i)
    strata = defaultdict(list)
    for indices in clusters.values():
        composition = tuple(sorted({direct[i]["type"] for i in indices}))
        tiers = tuple(sorted({direct[i].get("tier", "standard") for i in indices}))
        strata[(composition, tiers)].append(indices)
    rng = random.Random(15)
    diffs = []
    for _ in range(replicates):
        indices = [i for group in strata.values() for _ in group for i in rng.choice(group)]
        diffs.append(
            summarize([reasoned[i] for i in indices])["cc_equal_types"]
            - summarize([direct[i] for i in indices])["cc_equal_types"]
        )
    classification_gain = [
        b["correct"] - a["correct"]
        for a, b in zip(direct, reasoned, strict=True)
        if a["type"] != "score"
    ]
    score_gain = [
        a["nmae"] - b["nmae"] for a, b in zip(direct, reasoned, strict=True) if a["type"] == "score"
    ]
    # Continuous Score error changes are not repaired/broken classification
    # answers. Suppress roundoff in its diagnostic counts; retain every
    # measured difference in aggregate competence, NLL and the bootstrap.
    score_tolerance = 1e-6
    return {
        "fixed": sum(g > 0 for g in classification_gain),
        "broken": sum(g < 0 for g in classification_gain),
        "score_improved": sum(g > score_tolerance for g in score_gain),
        "score_worsened": sum(g < -score_tolerance for g in score_gain),
        "score_count_tolerance": score_tolerance,
        "counts_definition": "classification thresholded correctness; separate Score nMAE changes",
        "mean_nll_gain": sum(a["nll"] - b["nll"] for a, b in zip(direct, reasoned, strict=True))
        / len(direct),
        "cc_delta_95ci": [percentile(diffs, 0.025), percentile(diffs, 0.975)],
        "bootstrap_seed": 15,
        "bootstrap_replicates": replicates,
        "bootstrap_unit": "underlying_case",
        "independent_cases": len(clusters),
    }


def clustered_mean_interval(rows, values, replicates=2000, seed=15):
    groups = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        groups[row.get("cluster_id", row["id"])].append(value)
    units = list(groups.values())
    rng = random.Random(seed)
    boot = []
    for _ in range(replicates):
        selected = [rng.choice(units) for _ in units]
        boot.append(sum(map(sum, selected)) / sum(map(len, selected)))
    return [percentile(boot, 0.025), percentile(boot, 0.975)], len(units)


def select_candidates(reports):
    complete = [
        r
        for r in reports
        if r.get("status") == "complete"
        and r["n"] == 96
        and all(r["by_type"].get(t, {}).get("n") == 32 for t in ("choice", "noul", "score"))
    ]
    if not complete:
        return []
    best = max(r["cc_equal_types"] for r in complete)
    tied = [r for r in complete if best - r["cc_equal_types"] <= 1]
    tied.sort(
        key=lambda r: (
            r["proper_loss"],
            r["p95_s"] if r["p95_s"] is not None else math.inf,
            r.get("usd_per_1000") if r.get("usd_per_1000") is not None else math.inf,
        )
    )
    return tied + sorted([r for r in complete if r not in tied], key=lambda r: -r["cc_equal_types"])


def assert_isolated(splits):
    """Fail closed when templates, combinations, expressions or lineage overlap."""
    seen = {}
    fields = (
        "generator_template_id",
        "rule_combination",
        "document_voice",
        "source_example_id",
        "translation_of",
        "derived_from",
        "case_facts_sha256",
    )
    hashes = {}
    for split, samples in splits.items():
        for sample in samples:
            digest = hashlib.sha256(
                json.dumps(sample.state, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError("state overlap between independent splits")
            hashes[digest] = split
            for field in fields:
                value = sample.metadata.get(field)
                if value is None:
                    continue
                key = field, json.dumps(value, sort_keys=True)
                if key in seen and seen[key] != split:
                    raise ValueError(f"{field} overlap between {seen[key]} and {split}")
                seen[key] = split


def token_cost(
    rows, input_usd_per_million=None, output_usd_per_million=None, *, source=None, measured_on=None
):
    if input_usd_per_million is None or output_usd_per_million is None:
        return {"usd_per_1000": None, "basis": "unknown"}
    if not source or not measured_on or min(input_usd_per_million, output_usd_per_million) < 0:
        raise ValueError("nonnegative reference prices require a source and measurement date")
    total = (
        sum(
            r["input_tokens"] * input_usd_per_million
            + r["reasoning_tokens"] * output_usd_per_million
            for r in rows
        )
        / 1e6
    )
    return {
        "usd_per_1000": total * 1000 / len(rows),
        "basis": "reference-token-estimate",
        "source": source,
        "measured_on": measured_on,
    }
