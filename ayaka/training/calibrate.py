"""Per-primitive scalar temperature fitting (sec 40.2 / Stage 6, A8).

One scalar T per primitive is fit on a held-out split by minimizing
NLL of softmax(logits / T). Choice and Score keep scalar temperatures to
preserve the dynamic ontology and permutation equivariance. Canonical binary
Noul can additionally use the guarded correction in ``noul_calibration``.
"""

from __future__ import annotations

import torch

from ..model.electra import LENGTH_BUCKETS, PRIMITIVE_INDEX, ElectraDecisionModel


def fit_temperature(
    logits: list[list[float]], targets: list[list[float]], iters: int = 200
) -> float:
    if not logits:
        return 1.0
    flat = torch.tensor([x for v in logits for x in v], dtype=torch.float64)
    tgt = torch.tensor([x for v in targets for x in v], dtype=torch.float64)
    lens = torch.tensor([len(v) for v in logits])
    seg = torch.repeat_interleave(torch.arange(len(logits)), lens)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters, line_search_fn="strong_wolfe")

    def nll():
        z = flat / log_t.exp()
        m = z.new_full((len(logits),), float("-inf")).scatter_reduce(0, seg, z.detach(), "amax")
        e = (z - m[seg]).exp()
        lse = torch.zeros(len(logits), dtype=z.dtype).index_add(0, seg, e).log() + m
        return -(tgt * (z - lse[seg])).sum() / len(logits)

    def closure():
        opt.zero_grad()
        loss = nll()
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp())
    return min(max(t, 0.05), 20.0)


MIN_QUESTIONS = 20  # fewer -> fall back to the coarser temperature


def fit_temperatures(
    logits: list[list[float]],
    targets: list[list[float]],
    types: list[str],
    lengths: list[int] | None = None,
    long_threshold: int = 1024,
) -> dict[str, float]:
    """Scalar T per primitive ("choice") and, when ``lengths`` are given, per
    primitive and prompt-length bucket ("choice@long"). A bucket with too
    few questions keeps its primitive's temperature."""
    temps: dict[str, float] = {}
    for prim in PRIMITIVE_INDEX:
        idx = [i for i, t in enumerate(types) if t == prim]
        if len(idx) < MIN_QUESTIONS:
            continue
        temps[prim] = fit_temperature([logits[i] for i in idx], [targets[i] for i in idx])
        if lengths is None:
            continue
        for b, name in enumerate(LENGTH_BUCKETS):
            sub = [i for i in idx if (lengths[i] >= long_threshold) == bool(b)]
            if len(sub) >= MIN_QUESTIONS:
                temps[f"{prim}@{name}"] = fit_temperature(
                    [logits[i] for i in sub], [targets[i] for i in sub]
                )
    return temps


def apply_temperatures(model: ElectraDecisionModel, temps: dict[str, float]) -> None:
    with torch.no_grad():
        for prim, p_idx in PRIMITIVE_INDEX.items():
            base = temps.get(prim, 1.0)
            for b, name in enumerate(LENGTH_BUCKETS):
                model.temperature[p_idx, b] = temps.get(f"{prim}@{name}", base)
