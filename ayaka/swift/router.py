"""Pure-Python calibration-only logistic routing and resource projections."""

from __future__ import annotations

import json
import math
import re

from ayaka.eval.read_artifact import fingerprint

from .binding import SwiftReadIndex, validate_bound_reads
from .losses import row_nll
from .provenance import group_reads
from .reasoning import TRACE_INSTRUCTION, TRACE_MAX_TOKENS, reasoning_recipe
from .score import composite, cost_score, decision_cost, score_reads, speed_axis

ROUTE_RATE_CAP = 0.10
ROUTER_L2 = 0.01
ROUTER_ITERATIONS = 400
FEATURES = [
    "choice",
    "noul",
    "score",
    "K",
    "max_prob",
    "margin",
    "entropy",
    "digits",
    "dates",
    "currency_percent",
    "comparisons",
    "log_state_length",
]
DATE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")
COMPARISON = re.compile(
    r"\b(?:more|less|greater|fewer|higher|lower|before|after|earlier|later|least|most|equal|than|minimum|maximum)\b",
    re.I,
)


def request_text(state, instruction):
    state = (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
    )
    return state, state + "\n" + instruction


def features(kind, probs, state, instruction):
    state, text = request_text(state, instruction)
    values = sorted(probs.values(), reverse=True)
    return [
        *[float(kind == k) for k in ("choice", "noul", "score")],
        len(values),
        values[0],
        values[0] - values[1],
        -math.fsum(p * math.log(p) for p in values if p > 0),
        len(re.findall(r"\d", text)),
        len(DATE.findall(text)),
        len(re.findall(r"[$€£¥₩%]|\b(?:USD|EUR|GBP|KRW|percent)\b", text, re.I)),
        len(COMPARISON.findall(text)),
        math.log1p(len(state)),
    ]


def candidate(probs, state, instruction):
    """Predeclared before generation; no tier, target or trace features."""
    return len(probs) <= 26 and (
        max(probs.values()) <= 0.95 or bool(re.search(r"\d", request_text(state, instruction)[1]))
    )


def row_features(row):
    binding = row["binding"]
    return features(
        row["type"], row["raw_probs"], binding["state"], binding["question"]["instruction"]
    )


def eligible(row):
    binding = row["binding"]
    return candidate(row["raw_probs"], binding["state"], binding["question"]["instruction"])


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-700, min(700, value))))


def router_score(router, kind, probs, state, instruction):
    x = features(kind, probs, state, instruction)
    z = [
        (v - mean) / scale
        for v, mean, scale in zip(x, router["means"], router["scales"], strict=True)
    ]
    return sigmoid(
        router["intercept"] + math.fsum(w * v for w, v in zip(router["weights"], z, strict=True))
    )


def should_route(router, kind, probs, state, instruction):
    return (
        candidate(probs, state, instruction)
        and router_score(router, kind, probs, state, instruction) >= router["threshold"]
    )


def validate_router(router):
    if not isinstance(router, dict) or router.get("features") != FEATURES:
        raise ValueError("invalid reasoning router feature contract")
    for key in ("means", "scales", "weights"):
        values = router.get(key)
        if (
            not isinstance(values, list)
            or len(values) != len(FEATURES)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
        ):
            raise ValueError("reasoning router requires finite feature vectors")
    if any(v <= 0 for v in router["scales"]) or any(
        type(router.get(k)) not in (int, float) or not math.isfinite(router[k])
        for k in ("threshold", "intercept", "rate_cap")
    ):
        raise ValueError("invalid reasoning router parameters")
    if not 0 <= router["threshold"] <= 2 or not 0 <= router["rate_cap"] <= ROUTE_RATE_CAP:
        raise ValueError("invalid reasoning route cap or threshold")
    reasoning_recipe(router.get("max_tokens"))


# Refits on another platform differ at ~1e-16 (float summation order); a saved
# artifact is accepted when an independent refit agrees within this tolerance.
ROUTER_REFIT_TOLERANCE = 1e-9


def router_refit_difference(saved, refit):
    """Max absolute difference between two routers; discrete fields must match exactly."""
    validate_router(saved)
    validate_router(refit)
    for key in ("features", "threshold", "rate_cap", "max_tokens"):
        if saved.get(key) != refit.get(key):
            raise ValueError(f"router refit disagrees on {key}")
    pairs = [(saved["intercept"], refit["intercept"])]
    for key in ("means", "scales", "weights"):
        pairs += list(zip(saved[key], refit[key], strict=True))
    return max(abs(a - b) for a, b in pairs)


def validate_pairs(direct, paired, *, require_candidates=True):
    """Sparse reasoned artifacts retain the exact direct record and recipe."""
    validate_bound_reads(paired)
    reference = {row["id"]: row for row in direct}
    found, budgets = {}, set()
    for row in paired:
        original = reference.get(row["id"])
        nested = row.get("reasoned_read", {})
        plain = {k: v for k, v in row.items() if k not in ("record_sha256", "reasoned_read")}
        if (
            original is None
            or plain != {k: v for k, v in original.items() if k != "record_sha256"}
            or nested.get("direct_record_sha256") != original["record_sha256"]
            or nested.get("direct_binding_sha256") != original["binding_sha256"]
        ):
            raise ValueError("reasoned read differs from its exact direct read binding")
        recipe = nested.get("recipe", {})
        if recipe != reasoning_recipe(recipe.get("max_tokens")) or nested.get(
            "recipe_sha256"
        ) != fingerprint(recipe):
            raise ValueError("invalid bound reasoning recipe")
        budgets.add(recipe["max_tokens"])
        messages = [
            {
                "role": "user",
                "content": original["binding"]["messages"][-1]["content"]
                + "\n\n"
                + TRACE_INSTRUCTION,
            }
        ]
        inputs = nested.get("pass_inputs", [])
        generation = nested.get("generation_input", {})
        if nested.get("generation_messages") != messages or len(inputs) != 1:
            raise ValueError("reasoned read message binding mismatch")
        final = inputs[0]["messages"]
        if (
            len(final) != 3
            or final[0] != messages[0]
            or final[1].get("role") != "assistant"
            or not isinstance(final[1].get("content"), str)
            or final[2] != {"role": "user", "content": recipe["final_instruction"]}
        ):
            raise ValueError("reasoned read requires bound three-turn messages")
        # Reuse the canonical raw gather validator with the actual final prefix.
        token_input = {k: inputs[0][k] for k in ("input_token_ids", "canonical_token_ids")}
        token_input.update({k + "_sha256": fingerprint(v) for k, v in list(token_input.items())})
        binding = {
            **original["binding"],
            "messages": final,
            "rendered_input_sha256": fingerprint(final),
            "token_inputs": [token_input],
            "canonical_token_ids_sha256": fingerprint([token_input["canonical_token_ids"]]),
        }
        binding.pop("binding_sha256")
        binding["binding_sha256"] = fingerprint(binding)
        synthetic = {
            **original,
            **{
                k: nested[k]
                for k in (
                    "raw_probs",
                    "candidate_log_masses",
                    "input_tokens",
                    "output_tokens",
                    "latency_s",
                )
            },
            "binding": binding,
            "binding_sha256": binding["binding_sha256"],
            "pass_bindings": inputs,
        }
        synthetic.pop("record_sha256")
        synthetic["record_sha256"] = fingerprint(synthetic)
        SwiftReadIndex([synthetic])
        if (
            generation.get("input_token_ids_sha256")
            != fingerprint(generation.get("input_token_ids"))
            or not generation.get("input_token_ids")
            or nested.get("finish_reason") not in ("eos", "length")
            or nested.get("length_capped") != (nested["finish_reason"] == "length")
            or type(nested.get("trace_tokens")) is not int
            or not 0 <= nested["trace_tokens"] <= recipe["max_tokens"]
            or nested["input_tokens"] != nested["trace_input_tokens"] + nested["read_input_tokens"]
            or nested["output_tokens"] != nested["trace_tokens"] + nested["read_output_tokens"]
            or nested["read_output_tokens"] != 1
            or nested.get("readout") != "canonical_letter_raw"
            or nested.get("passes") != 2
            or any(
                type(nested.get(k)) is not int or nested[k] < 1
                for k in ("trace_input_tokens", "read_input_tokens")
            )
            or any(
                type(nested.get(k)) not in (int, float)
                or not math.isfinite(nested[k])
                or nested[k] <= 0
                for k in ("trace_latency_s", "read_latency_s", "latency_s")
            )
            or not math.isclose(
                nested["latency_s"], nested["trace_latency_s"] + nested["read_latency_s"]
            )
        ):
            raise ValueError("reasoned read usage/finish/latency binding mismatch")
        if not eligible(original):
            raise ValueError("reasoned row is outside the predeclared candidate subset")
        found[row["id"]] = nested
    if len(budgets) > 1:
        raise ValueError("reasoned reads must use one predeclared trace budget")
    if require_candidates and set(found) != {r["id"] for r in direct if eligible(r)}:
        raise ValueError("reasoning requires all predeclared candidates with both reads")
    return found


def routed_rows(rows, router, paired):
    system, flags = [], []
    for row in rows:
        b = row["binding"]
        route = should_route(
            router, row["type"], row["raw_probs"], b["state"], b["question"]["instruction"]
        )
        flags.append(route)
        if route:
            reasoned = paired[row["id"]]
            system.append(
                {
                    **row,
                    "raw_probs": reasoned["raw_probs"],
                    "candidate_log_masses": reasoned["candidate_log_masses"],
                    **{
                        k: row[k] + reasoned[k]
                        for k in ("input_tokens", "output_tokens", "latency_s")
                    },
                }
            )
        else:
            system.append(row)
    return system, flags


def projected_latency(rows, flags, paired):
    """Mix empirical distributions at the overall share, including direct overhead."""
    overall_share = sum(flags) / len(flags)
    proxy = [
        (r, flag)
        for r, flag in zip(rows, flags, strict=True)
        if r.get("tier") in ("standard", "judge")
    ]
    scope = "non_public_standard_judge" if proxy else "all_items_fallback_assumed"
    if proxy:
        rows, flags = map(list, zip(*proxy, strict=True))
    share = sum(flags) / len(flags)
    # Routing can correlate with slow direct reads. Reusing those reads in
    # both mixture arms would double-count the slow tail (e.g. 4% -> 7.84%).
    direct = [r["latency_s"] for r, flag in zip(rows, flags, strict=True) if not flag]
    routed = [
        r["latency_s"] + paired[r["id"]]["latency_s"]
        for r, flag in zip(rows, flags, strict=True)
        if flag
    ]
    if any(not math.isfinite(v) or v <= 0 for v in direct + routed):
        raise ValueError("projection requires positive measured latencies")
    mixture = [(v, (1 - share) / len(direct)) for v in direct] if direct else []
    if routed:
        mixture += [(v, share / len(routed)) for v in routed]
    mixture.sort()

    def quantile(q):
        cumulative = 0.0
        for v, weight in mixture:
            cumulative += weight
            if cumulative > q + 1e-12:
                return v
        return mixture[-1][0]

    p50, p95 = quantile(0.5), quantile(0.95)
    return {
        "p50_s": p50,
        "p95_s": p95,
        "adjusted_p95_s": 2 * p95 + 0.15,
        "S": speed_axis(p50, p95),
        "route_rate": share,
        "overall_route_rate": overall_share,
        "tier_scope": scope,
        "source": "projected",
        "assumed": True,
        "distribution_source": "collected_backend_latencies",
    }


def fit_router(
    rows,
    paired,
    policy,
    *,
    rate_cap=ROUTE_RATE_CAP,
    usd_in_per_m=0.0403,
    usd_out_per_m=0.0403,
    assumed_cost=56.4,
):
    if any(r.get("split") != "calibration" or r.get("public") is not False for r in rows):
        raise ValueError("router fitting requires non-public calibration only")
    if not 0 <= rate_cap <= ROUTE_RATE_CAP:
        raise ValueError("router requires a valid rate cap")
    groups = group_reads(rows, "calibration", require_variants=False)
    if set(groups) != {policy.prompt_variant}:
        raise ValueError("router fitting requires the accepted prompt variant only")
    rows = groups[policy.prompt_variant]
    enriched = []
    for row in rows:
        if row["id"] in paired:
            value = {**row, "reasoned_read": paired[row["id"]]}
            value["record_sha256"] = fingerprint(
                {k: v for k, v in value.items() if k != "record_sha256"}
            )
            enriched.append(value)
    paired = validate_pairs(rows, enriched)
    training = [r for r in rows if r["id"] in paired]
    if not training:
        router = {
            "features": FEATURES.copy(),
            "means": [0.0] * len(FEATURES),
            "scales": [1.0] * len(FEATURES),
            "weights": [0.0] * len(FEATURES),
            "intercept": 0.0,
            "threshold": 2.0,
            "rate_cap": rate_cap,
            "max_tokens": TRACE_MAX_TOKENS,
        }
        latency = projected_latency(rows, [False] * len(rows), paired)
        score = score_reads(rows, policy)
        cost = (
            cost_score(decision_cost(rows, usd_in_per_m, usd_out_per_m))
            if usd_in_per_m is not None
            else assumed_cost
        )
        off = {
            "threshold": 2.0,
            "A": composite(score["I"], score["C"], latency["S"], cost),
            "route_rate": 0.0,
            "latency": latency,
            "Cost": cost,
        }
        return {
            "router": router,
            "calibration": off,
            "threshold_trials": [off],
            "fit_n": 0,
            "positive_n": 0,
            "l2": ROUTER_L2,
            "iterations": ROUTER_ITERATIONS,
            "fit_role": "non_public_calibration",
            "speed_source": "projected_assumed",
            "reason": "no_candidates",
        }
    x = [row_features(r) for r in training]
    n = len(x)
    means = [math.fsum(v[j] for v in x) / n for j in range(len(FEATURES))]
    scales = [
        max(1e-8, math.sqrt(math.fsum((v[j] - means[j]) ** 2 for v in x) / n))
        for j in range(len(FEATURES))
    ]
    x = [[(v - m) / s for v, m, s in zip(values, means, scales, strict=True)] for values in x]

    def loss(row):
        # Lower proper loss under the accepted readout policy, before commit.
        probs = {label: row["raw_probs"][label] for label in row["labels"]}
        masses = policy.biased_log_masses(row["type"], probs, row["candidate_log_masses"])
        return row_nll({**row, "candidate_log_masses": masses}, getattr(policy, "t_" + row["type"]))

    y = [float(loss({**r, **paired[r["id"]]}) < loss(r)) for r in training]
    weights, intercept = [0.0] * len(FEATURES), 0.0
    for _ in range(ROUTER_ITERATIONS):
        errors = [
            sigmoid(intercept + math.fsum(w * v for w, v in zip(weights, values, strict=True)))
            - target
            for values, target in zip(x, y, strict=True)
        ]
        intercept -= 0.1 * math.fsum(errors) / n
        weights = [
            w
            - 0.1
            * (
                math.fsum(e * values[j] for e, values in zip(errors, x, strict=True)) / n
                + ROUTER_L2 * w
            )
            for j, w in enumerate(weights)
        ]
    router = {
        "features": FEATURES.copy(),
        "means": means,
        "scales": scales,
        "weights": weights,
        "intercept": intercept,
        "threshold": 2.0,
        "rate_cap": rate_cap,
        "max_tokens": next(iter(paired.values()))["recipe"]["max_tokens"],
    }
    scores = [
        router_score(
            router,
            r["type"],
            r["raw_probs"],
            r["binding"]["state"],
            r["binding"]["question"]["instruction"],
        )
        for r in training
    ]
    trials = []
    for threshold in [2.0, *sorted(set(scores), reverse=True)]:
        proposal = {**router, "threshold": threshold}
        system, flags = routed_rows(rows, proposal, paired)
        if sum(flags) / len(rows) > rate_cap + 1e-12:
            continue
        latency = projected_latency(rows, flags, paired)
        score = score_reads(system, policy)
        cost = (
            cost_score(decision_cost(system, usd_in_per_m, usd_out_per_m))
            if usd_in_per_m is not None
            else assumed_cost
        )
        trials.append(
            {
                "threshold": threshold,
                "A": composite(score["I"], score["C"], latency["S"], cost),
                "route_rate": sum(flags) / len(rows),
                "latency": latency,
                "Cost": cost,
            }
        )
    best = max(trials, key=lambda t: t["A"])
    router["threshold"] = best["threshold"]
    validate_router(router)
    return {
        "router": router,
        "calibration": best,
        "threshold_trials": trials,
        "fit_n": n,
        "positive_n": int(sum(y)),
        "l2": ROUTER_L2,
        "iterations": ROUTER_ITERATIONS,
        "fit_role": "non_public_calibration",
        "speed_source": "projected_assumed",
    }


def public_route_diagnostic(rows, policy):
    items = [
        r
        for r in rows
        if r.get("public") is True
        and r.get("tier") in ("standard", "judge")
        and r.get("readout") != "grouped_approx"
        and r.get("prompt_variant") == policy.prompt_variant
    ]
    flags = (
        [
            should_route(
                policy.reasoning_route,
                r["type"],
                r["raw_probs"],
                r["binding"]["state"],
                r["binding"]["question"]["instruction"],
            )
            for r in items
        ]
        if policy.reasoning_route
        else [False] * len(items)
    )
    return {
        "role": "DIAGNOSTIC_ONLY",
        "used_for_fit_or_gates": False,
        "public": True,
        "tiers": ["standard", "judge"],
        "n": len(items),
        "routed_n": sum(flags),
        "route_rate": sum(flags) / len(flags) if flags else None,
    }
