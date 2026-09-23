"""Validation-gate metrics (docs.md section 48).

Checkpoint promotion is gated on probability quality, not accuracy:
NLL, Brier, RPS, ECE, JSD to human distributions, selective risk,
OOS detection, permutation-invariance error, missing-evidence
overconfidence, and structural invariant checks.
"""

from __future__ import annotations

import torch

from .model.model import DecisionOutput


def _seg_ids(cu: torch.Tensor) -> torch.Tensor:
    lens = (cu[1:] - cu[:-1]).long()
    return torch.repeat_interleave(torch.arange(lens.numel(), device=cu.device), lens)


def _per_q(x: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(cu.numel() - 1, dtype=x.dtype, device=x.device)
    out.index_add_(0, _seg_ids(cu), x)
    return out


def per_question_argmax(out: DecisionOutput) -> torch.Tensor:
    """Argmax candidate index within each question's set -> [n_q] global idx."""
    cu = out.cand_cu
    winners = torch.empty(cu.numel() - 1, dtype=torch.long, device=out.logits.device)
    for i in range(cu.numel() - 1):
        s, e = int(cu[i]), int(cu[i + 1])
        winners[i] = s + out.logits[s:e].argmax()
    return winners


def accuracy(out: DecisionOutput, targets: torch.Tensor) -> float:
    """Fraction of questions where argmax p == argmax y."""
    cu = out.cand_cu
    correct = 0
    for i in range(cu.numel() - 1):
        s, e = int(cu[i]), int(cu[i + 1])
        if out.logits[s:e].argmax() == targets[s:e].argmax():
            correct += 1
    return correct / max(cu.numel() - 1, 1)


def nll(out: DecisionOutput, targets: torch.Tensor) -> float:
    logp = out.log_probs()
    return float((-_per_q(targets * logp, out.cand_cu)).mean())


def brier(out: DecisionOutput, targets: torch.Tensor) -> float:
    p = out.probs()
    return float(_per_q((p - targets) ** 2, out.cand_cu).mean())


def expected_calibration_error(
    out: DecisionOutput, targets: torch.Tensor, n_bins: int = 10
) -> float:
    """ECE over per-question (confidence, correctness) pairs."""
    cu = out.cand_cu
    p = out.probs()
    confs, corrects = [], []
    for i in range(cu.numel() - 1):
        s, e = int(cu[i]), int(cu[i + 1])
        j = int(out.logits[s:e].argmax())
        confs.append(float(p[s + j]))
        corrects.append(float(targets[s:e].argmax() == j))
    conf = torch.tensor(confs)
    corr = torch.tensor(corrects)
    ece = 0.0
    edges = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.any():
            ece += float(m.float().mean() * (conf[m] - corr[m]).abs().mean())
    return ece


def jsd_to_target(out: DecisionOutput, targets: torch.Tensor) -> float:
    """Mean Jensen-Shannon divergence p || y per question."""
    p = out.probs()
    jsd = torch.zeros(out.cand_cu.numel() - 1, device=p.device)
    cu = out.cand_cu
    for i in range(cu.numel() - 1):
        s, e = int(cu[i]), int(cu[i + 1])
        pi, yi = p[s:e], targets[s:e]
        m = 0.5 * (pi + yi)
        kl_pm = (pi * (pi.clamp(min=1e-9).log() - m.clamp(min=1e-9).log())).sum()
        kl_ym = (yi * (yi.clamp(min=1e-9).log() - m.clamp(min=1e-9).log())).sum()
        jsd[i] = 0.5 * (kl_pm + kl_ym)
    return float(jsd.mean())


def selective_risk(out: DecisionOutput, targets: torch.Tensor, coverage: float = 0.8) -> float:
    """Error rate among the `coverage` most confident questions."""
    cu = out.cand_cu
    p = out.probs()
    conf, err = [], []
    for i in range(cu.numel() - 1):
        s, e = int(cu[i]), int(cu[i + 1])
        j = int(out.logits[s:e].argmax())
        conf.append(float(p[s + j]))
        err.append(float(targets[s:e].argmax() != j))
    conf_t = torch.tensor(conf)
    err_t = torch.tensor(err)
    n_keep = max(1, int(len(conf) * coverage))
    top = conf_t.argsort(descending=True)[:n_keep]
    return float(err_t[top].mean())


def oos_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC for out-of-scope detection (scores=higher means in-scope)."""
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    pos = labels.bool()
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def permutation_invariance_error(p1: torch.Tensor, p2_aligned: torch.Tensor) -> float:
    """max |p - pi(p)| over candidates of two runs (sec 48 invariant 1)."""
    return float((p1 - p2_aligned).abs().max())


def missing_evidence_overconfidence(out: DecisionOutput, flagged_mask: torch.Tensor) -> float:
    """Mean top-probability on insufficient-evidence questions."""
    if not flagged_mask.any():
        return float("nan")
    from .model.pointer import ragged_max

    p_max = ragged_max(out.log_probs(), out.cand_cu).exp()
    return float(p_max[flagged_mask].mean())


def macro_f1(out: DecisionOutput, targets: torch.Tensor) -> float:
    """Macro-F1 over per-question argmax labels (candidate-position labels)."""
    cu = out.cand_cu
    preds, golds = [], []
    for i in range(cu.numel() - 1):
        s, e = int(cu[i]), int(cu[i + 1])
        preds.append(int(out.logits[s:e].argmax()))
        golds.append(int(targets[s:e].argmax()))
    labels = sorted(set(golds) | set(preds))
    f1s = []
    for c in labels:
        tp = sum(1 for p, g in zip(preds, golds, strict=True) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, golds, strict=True) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, golds, strict=True) if p != c and g == c)
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(torch.tensor(f1s).mean()) if f1s else 0.0


def compute_metrics(
    out: DecisionOutput,
    targets: torch.Tensor,
    missing_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    m = {
        "accuracy": accuracy(out, targets),
        "macro_f1": macro_f1(out, targets),
        "nll": nll(out, targets),
        "brier": brier(out, targets),
        "ece": expected_calibration_error(out, targets),
        "jsd": jsd_to_target(out, targets),
        "selective_risk@0.8": selective_risk(out, targets, 0.8),
    }
    if missing_mask is not None:
        m["missing_evidence_pmax"] = missing_evidence_overconfidence(out, missing_mask)
    return m
