"""Validation-gate metrics (docs.md section 48).

Metrics take per-question probability vectors and target vectors
(plain lists), so results from many batches aggregate trivially.
Targets may be soft (teacher/human distributions): "correct" means
argmax p == argmax y, and distribution metrics (NLL, KL, JSD, Brier,
RPS) use the full target.
"""

from __future__ import annotations

import math


def _argmax(v: list[float]) -> int:
    return max(range(len(v)), key=v.__getitem__)


def accuracy(p: list[list[float]], y: list[list[float]]) -> float:
    return sum(_argmax(a) == _argmax(b) for a, b in zip(p, y, strict=True)) / max(len(p), 1)


def nll(p: list[list[float]], y: list[list[float]]) -> float:
    tot = 0.0
    for a, b in zip(p, y, strict=True):
        tot -= sum(bi * math.log(max(ai, 1e-12)) for ai, bi in zip(a, b, strict=True))
    return tot / max(len(p), 1)


def kl(y: list[list[float]], p: list[list[float]]) -> float:
    """Mean KL(y || p) — fidelity to a reference (e.g. Jev) distribution."""
    tot = 0.0
    for a, b in zip(p, y, strict=True):
        tot += sum(
            bi * (math.log(max(bi, 1e-12)) - math.log(max(ai, 1e-12)))
            for ai, bi in zip(a, b, strict=True)
            if bi > 0
        )
    return tot / max(len(p), 1)


def brier(p: list[list[float]], y: list[list[float]]) -> float:
    return sum(
        sum((ai - bi) ** 2 for ai, bi in zip(a, b, strict=True)) for a, b in zip(p, y, strict=True)
    ) / max(len(p), 1)


def rps(p: list[list[float]], y: list[list[float]]) -> float:
    """Ranked Probability Score; vectors must already be in ordinal order."""
    tot, n = 0.0, 0
    for a, b in zip(p, y, strict=True):
        k = len(a)
        if k < 2:
            continue
        ca = cb = 0.0
        s = 0.0
        for i in range(k - 1):
            ca += a[i]
            cb += b[i]
            s += (ca - cb) ** 2
        tot += s / (k - 1)
        n += 1
    return tot / max(n, 1)


def jsd(p: list[list[float]], y: list[list[float]]) -> float:
    tot = 0.0
    for a, b in zip(p, y, strict=True):
        m = [(ai + bi) / 2 for ai, bi in zip(a, b, strict=True)]
        for u in (a, b):
            tot += 0.5 * sum(
                ui * (math.log(max(ui, 1e-12)) - math.log(max(mi, 1e-12)))
                for ui, mi in zip(u, m, strict=True)
                if ui > 0
            )
    return tot / max(len(p), 1)


def ece(p: list[list[float]], y: list[list[float]], n_bins: int = 10) -> float:
    conf = [max(a) for a in p]
    corr = [float(_argmax(a) == _argmax(b)) for a, b in zip(p, y, strict=True)]
    out = 0.0
    for k in range(n_bins):
        lo, hi = k / n_bins, (k + 1) / n_bins
        idx = [i for i, c in enumerate(conf) if lo <= c < hi or (k == n_bins - 1 and c == hi)]
        if idx:
            acc = sum(corr[i] for i in idx) / len(idx)
            cf = sum(conf[i] for i in idx) / len(idx)
            out += len(idx) / len(conf) * abs(acc - cf)
    return out


def selective_risk(p: list[list[float]], y: list[list[float]], coverage: float = 0.8) -> float:
    ranked = sorted(zip(p, y, strict=True), key=lambda t: -max(t[0]))
    keep = ranked[: max(1, int(len(ranked) * coverage))]
    return sum(_argmax(a) != _argmax(b) for a, b in keep) / len(keep)


def missing_evidence_pmax(p: list[list[float]], flagged: list[bool]) -> float:
    vals = [max(a) for a, f in zip(p, flagged, strict=True) if f]
    return sum(vals) / len(vals) if vals else float("nan")


def permutation_invariance_error(p1: list[float], p2_aligned: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(p1, p2_aligned, strict=True))


def compute_metrics(
    p: list[list[float]],
    y: list[list[float]],
    types: list[str] | None = None,
    flagged: list[bool] | None = None,
) -> dict[str, float]:
    m = {
        "n": len(p),
        "accuracy": accuracy(p, y),
        "nll": nll(p, y),
        "kl": kl(y, p),
        "brier": brier(p, y),
        "ece": ece(p, y),
        "jsd": jsd(p, y),
        "selective_risk@0.8": selective_risk(p, y) if p else float("nan"),
    }
    if types is not None:
        for t in sorted(set(types)):
            idx = [i for i, tt in enumerate(types) if tt == t]
            m[f"accuracy/{t}"] = accuracy([p[i] for i in idx], [y[i] for i in idx])
            m[f"kl/{t}"] = kl([y[i] for i in idx], [p[i] for i in idx])
        s_idx = [i for i, t in enumerate(types) if t == "score"]
        if s_idx:
            m["rps/score"] = rps([p[i] for i in s_idx], [y[i] for i in s_idx])
    if flagged is not None and any(flagged):
        m["missing_evidence_pmax"] = missing_evidence_pmax(p, flagged)
    return m
