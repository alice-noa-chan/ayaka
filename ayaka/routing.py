"""Paired NLL/token regression used exclusively by auto requests."""

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass

import torch

from .eval.v2 import nll, percentile
from .prompt import render_state

LAMBDAS = (0.0, 0.0005, 0.001, 0.002)


def routing_features(state, spec, baseline, tok, budget=384):
    p = sorted(baseline.probs, reverse=True)
    return [
        max(p),
        p[0] - p[1],
        -sum(v * math.log(max(v, 1e-12)) for v in p),
        math.log1p(len(tok.encode(render_state(state)))),
        math.log1p(len(tok.encode(spec.instruction))),
        math.log1p(len(spec.candidates)),
        float(spec.type == "noul"),
        float(spec.type == "score"),
        budget / 1024,
    ]


def paired_training_rows(direct, reasoned, split):
    if [r["id"] for r in direct] != [r["id"] for r in reasoned]:
        raise ValueError("router pairs must be aligned")
    if split not in ("router_train", "dev") or any(r["split"] != split for r in direct + reasoned):
        raise ValueError("router fitting uses router_train and dev only")
    return [
        {
            "id": a["id"],
            "split": split,
            "features": a["routing_features"][:-1] + [b["budget"] / 1024],
            "gain": nll(a["probs"], a["target"]) - nll(b["probs"], b["target"]),
            "tokens": b["reasoning_tokens"],
        }
        for a, b in zip(direct, reasoned, strict=True)
    ]


@dataclass
class BenefitRouter:
    mean: list[float]
    scale: list[float]
    gain_weights: list[float]
    token_weights: list[float]
    penalty: float
    promoted: bool
    validation: dict

    def predict(self, features):
        x = [1.0] + [(v - m) / s for v, m, s in zip(features, self.mean, self.scale, strict=True)]
        gain = sum(a * b for a, b in zip(x, self.gain_weights, strict=True))
        tokens = min(
            max(0, features[-1] * 1024),
            max(0, sum(a * b for a, b in zip(x, self.token_weights, strict=True)) * 1024),
        )
        return gain, tokens

    def should_reason(self, state, spec, baseline, tok, budget=384):
        # Old single-budget artifacts retain only that measured budget. A
        # constant budget feature cannot establish benefit at other efforts.
        supported = self.validation.get("validated_budgets")
        if supported is None:
            supported = [round(self.mean[-1] * 1024)] if self.scale[-1] <= 1e-6 else []
        if budget not in supported:
            return False
        gain, tokens = self.predict(routing_features(state, spec, baseline, tok, budget))
        return self.promoted and gain - self.penalty * tokens > 0

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            router = cls(**json.load(f))
        if not router.promoted or router.penalty not in LAMBDAS:
            raise ValueError("router must have a validated promotion and an allowed lambda")
        arrays = [router.mean, router.scale, router.gain_weights, router.token_weights]
        if (
            [len(x) for x in arrays] != [9, 9, 10, 10]
            or any(not math.isfinite(v) for x in arrays for v in x)
            or min(router.scale) <= 0
        ):
            raise ValueError("invalid router regression coefficients")
        return router


def fit_router(train, dev):
    if (
        len({r["id"] for r in train}) < 20
        or len({r["id"] for r in dev}) < 20
        or any(r["split"] != "router_train" for r in train)
        or any(r["split"] != "dev" for r in dev)
    ):
        raise ValueError("router needs at least 20 independent router_train and dev rows")
    if {r["id"] for r in train} & {r["id"] for r in dev}:
        raise ValueError("router training/dev item overlap")
    budgets = sorted({round(r["features"][-1] * 1024) for r in train})
    if budgets != sorted({round(r["features"][-1] * 1024) for r in dev}):
        raise ValueError("router training/dev budgets must match")
    for collection in (train, dev):
        keys = [(r["id"], round(r["features"][-1] * 1024)) for r in collection]
        if len(set(keys)) != len(keys) or len(keys) != len({r["id"] for r in collection}) * len(
            budgets
        ):
            raise ValueError("router requires one paired measurement per question and budget")
    raw = torch.tensor([r["features"] for r in train], dtype=torch.float64)
    mean, scale = raw.mean(0), raw.std(0).clamp(min=1e-6)
    x = torch.cat([torch.ones(len(train), 1, dtype=raw.dtype), (raw - mean) / scale], dim=1)
    regularizer = torch.eye(x.shape[1], dtype=x.dtype)
    regularizer[0, 0] = 0
    y = torch.tensor([[r["gain"], r["tokens"] / 1024] for r in train], dtype=x.dtype)
    weights = torch.linalg.solve(x.T @ x + regularizer, x.T @ y)
    router = BenefitRouter(
        mean.tolist(),
        scale.tolist(),
        weights[:, 0].tolist(),
        weights[:, 1].tolist(),
        0.0,
        False,
        {},
    )
    candidates = []
    for penalty in LAMBDAS:
        selected = [
            router.predict(r["features"])[0] - penalty * router.predict(r["features"])[1] > 0
            for r in dev
        ]
        gain = sum(r["gain"] for r, use in zip(dev, selected, strict=True) if use) / len(dev)
        tokens = sum(r["tokens"] for r, use in zip(dev, selected, strict=True) if use) / len(dev)
        candidates.append({"lambda": penalty, "gain": gain, "tokens": tokens, "selected": selected})
    # Compare dev NLL first. Within 0.01 nats of the best gain, minimize cost.
    # This avoids comparing objectives that use different lambda units.
    best_gain = max(c["gain"] for c in candidates)
    chosen = min(
        [c for c in candidates if best_gain - c["gain"] <= 0.01],
        key=lambda c: (c["tokens"], -c["gain"], c["lambda"]),
    )
    realized = [r["gain"] if use else 0 for r, use in zip(dev, chosen["selected"], strict=True)]
    # Repeated effort measurements of one question are one independent unit.
    groups = {}
    for row, gain in zip(dev, realized, strict=True):
        groups.setdefault(row["id"], []).append(gain)
    question_gains = [sum(values) / len(values) for values in groups.values()]
    rng = random.Random(15)
    boot = [
        sum(rng.choice(question_gains) for _ in question_gains) / len(question_gains)
        for _ in range(2000)
    ]
    interval = [percentile(boot, 0.025), percentile(boot, 0.975)]
    router.penalty = chosen["lambda"]
    router.promoted = interval[0] > 0
    validated_budgets = []
    budget_intervals = {}
    for budget in budgets:
        gains = [
            gain
            for row, gain in zip(dev, realized, strict=True)
            if round(row["features"][-1] * 1024) == budget
        ]
        boot = [sum(rng.choice(gains) for _ in gains) / len(gains) for _ in range(2000)]
        bound = [percentile(boot, 0.025), percentile(boot, 0.975)]
        budget_intervals[str(budget)] = bound
        if bound[0] > 0:
            validated_budgets.append(budget)
    router.promoted = router.promoted and bool(validated_budgets)
    router.validation = {
        "mean_nll_gain": chosen["gain"],
        "mean_tokens": chosen["tokens"],
        "nll_gain_95ci": interval,
        "dev_n": len(dev),
        "dev_independent_questions": len(groups),
        "validated_budgets": validated_budgets,
        "budget_nll_gain_95ci": budget_intervals,
        "dev_ids_sha256": hashlib.sha256(
            json.dumps(sorted(r["id"] for r in dev)).encode()
        ).hexdigest(),
        "lambda_grid": [{k: v for k, v in c.items() if k != "selected"} for c in candidates],
    }
    return router
