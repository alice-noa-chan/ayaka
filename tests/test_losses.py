import math

import pytest
import torch

from ayaka.losses import (
    LossWeights,
    brier_loss,
    decision_loss,
    kl_distill,
    missing_evidence_loss,
    nll_loss,
    rps_loss,
)
from ayaka.model.electra import CHOICE, SCORE, DecisionOutput
from ayaka.model.ragged import ragged_log_softmax, ragged_softmax


def _out(logits, cu, prim, pointer=None):
    logits = torch.tensor(logits, dtype=torch.float32, requires_grad=True)
    cu = torch.tensor(cu)
    seg = torch.repeat_interleave(torch.arange(len(cu) - 1), cu[1:] - cu[:-1])
    ptr = torch.tensor(pointer if pointer is not None else [0.0] * len(logits))
    return DecisionOutput(logits, logits.detach(), ptr, cu, seg, torch.tensor(prim))


def test_ragged_softmax_per_set():
    p = ragged_softmax(torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]), torch.tensor([0, 2, 5]))
    assert p[:2].sum() == pytest.approx(1.0)
    assert p[2:].sum() == pytest.approx(1.0)
    assert p[2] == pytest.approx(1 / 3)


def test_nll_matches_cross_entropy():
    cu = torch.tensor([0, 3])
    logits = torch.tensor([1.0, 2.0, 0.5])
    y = torch.tensor([0.0, 1.0, 0.0])
    got = nll_loss(ragged_log_softmax(logits, cu), y, cu)
    ref = torch.nn.functional.cross_entropy(logits[None], torch.tensor([1]))
    assert float(got) == pytest.approx(float(ref), rel=1e-5)


def test_brier_perfect_is_zero():
    cu = torch.tensor([0, 2])
    assert float(brier_loss(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]), cu)) == 0.0


def test_rps_prefers_adjacent_mistakes():
    cu = torch.tensor([0, 4, 8])
    y = torch.tensor([0, 0, 1.0, 0, 0, 0, 1.0, 0])
    near = torch.tensor([0, 1.0, 0, 0, 0, 1.0, 0, 0])
    far = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 1.0])
    ords = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    mask = torch.tensor([True, True])
    assert float(rps_loss(near, y, cu, ords, mask)) < float(rps_loss(far, y, cu, ords, mask))


def test_rps_respects_ordinals_not_positions():
    cu = torch.tensor([0, 3])
    y = torch.tensor([0.0, 0.0, 1.0])  # gold = ordinal 0 (stored last)
    p = torch.tensor([0.0, 1.0, 0.0])  # predicts ordinal 1
    ords = torch.tensor([2, 1, 0])
    val = float(rps_loss(p, y, cu, ords, torch.tensor([True])))
    assert val == pytest.approx(0.5)  # one CDF step off out of k-1 = 2


def test_missing_evidence_penalizes_only_overconfident_flagged():
    cu = torch.tensor([0, 2, 4])
    logp = ragged_log_softmax(torch.tensor([5.0, 0.0, 5.0, 0.0]), cu)
    flagged = torch.tensor([True, False])
    pen = float(missing_evidence_loss(logp, cu, flagged, tau=0.8))
    p_max = 1 / (1 + math.exp(-5))
    assert pen == pytest.approx((p_max - 0.8) ** 2, rel=1e-4)
    assert float(missing_evidence_loss(logp, cu, torch.tensor([False, False]))) == 0.0


def test_kl_distill_zero_when_equal():
    cu = torch.tensor([0, 3])
    logits = torch.tensor([0.2, 1.0, -0.5])
    lp = ragged_log_softmax(logits, cu)
    assert float(kl_distill(lp.exp(), lp, cu)) == pytest.approx(0.0, abs=1e-6)


def test_decision_loss_composite_and_grad():
    out = _out([1.0, 0.0, 0.0, 2.0, 0.5], [0, 2, 5], [CHOICE, SCORE])
    y = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
    parts = decision_loss(
        out, y, ordinals=torch.tensor([0, 1, 0, 1, 2]), weights=LossWeights(pointer_aux=0.0)
    )
    assert set(parts) >= {"total", "nll", "brier", "rps"}
    parts["total"].backward()
    assert out.logits.grad is not None and out.logits.grad.abs().sum() > 0


def test_teacher_kl_replaces_gold_nll_for_teacher_questions():
    out = _out([0.0, 0.0, 0.0, 0.0], [0, 2, 4], [CHOICE, CHOICE])
    gold = torch.tensor([1.0, 0.0, 1.0, 0.0])
    teacher = torch.tensor([0.5, 0.5, 0.9, 0.1])
    w = LossWeights(brier=0.0, pointer_aux=0.0)
    with_t = decision_loss(
        out, gold, teacher_probs=teacher, teacher_mask=torch.tensor([True, False]), weights=w
    )
    # question 0: KL(0.5/0.5 || uniform) = 0; question 1: gold NLL = log 2
    assert float(with_t["total"].detach()) == pytest.approx(math.log(2) / 2, rel=1e-4)
    assert float(with_t["kl"]) == pytest.approx(0.0, abs=1e-6)
