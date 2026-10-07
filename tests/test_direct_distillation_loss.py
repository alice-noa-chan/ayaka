import math

import pytest
import torch

from ayaka.losses import LossWeights, decision_loss, rps_loss
from ayaka.model.decision import CHOICE, SCORE, DecisionOutput


def output():
    logits = torch.zeros(5, requires_grad=True)
    cu = torch.tensor([0, 2, 5])
    return DecisionOutput(
        logits,
        logits.detach(),
        logits.detach(),
        cu,
        torch.tensor([0, 0, 1, 1, 1]),
        torch.tensor([CHOICE, SCORE]),
    )


def weights(**kwargs):
    return LossWeights(brier=0, pointer_aux=0, gold_nll_with_teacher=True, **kwargs)


def test_teacher_equal_to_student_still_learns_gold_and_freezes_teacher():
    out = output()
    gold = torch.tensor([1.0, 0, 0, 1, 0])
    teacher = torch.tensor([0.5, 0.5, 0, 1, 0.0], requires_grad=True)
    parts = decision_loss(
        out,
        gold,
        teacher_probs=teacher,
        teacher_mask=torch.tensor([True, False]),
        weights=weights(),
    )
    assert parts["total"].item() == pytest.approx((math.log(2) + math.log(3)) / 2)
    assert parts["kl"].item() == pytest.approx(0, abs=1e-7)
    parts["total"].backward()
    assert out.logits.grad[:2].tolist() == pytest.approx([-0.25, 0.25])
    assert teacher.grad is None


def test_auxiliary_kl_is_additive_masked_and_preserves_score_rps():
    out = output()
    gold = torch.tensor([1.0, 0, 0, 1, 0])
    teacher = torch.tensor([0.0, 1, 1, 0, 0])
    ordinals = torch.tensor([0, 1, 2, 0, 1])
    parts = decision_loss(
        out,
        gold,
        ordinals=ordinals,
        teacher_probs=teacher,
        teacher_mask=torch.tensor([True, False]),
        weights=weights(distill=0.2, nll=2),
    )
    rps = rps_loss(out.probs(), gold, out.cand_cu, ordinals, out.primitive == SCORE).item()
    assert parts["total"].item() == pytest.approx(
        math.log(2) + math.log(3) + 0.2 * math.log(2) / 2 + 0.35 * rps
    )
    assert parts["rps"].item() == pytest.approx(rps)
    assert parts["kl"].item() == pytest.approx(math.log(2))
    parts["total"].backward()
    assert torch.isfinite(out.logits.grad).all()


def test_zero_kl_weight_is_gold_objective_and_default_remains_legacy():
    gold = torch.tensor([1.0, 0, 0, 1, 0])
    teacher = torch.tensor([0.5, 0.5, 1, 0, 0])
    mask = torch.tensor([True, False])
    base = decision_loss(output(), gold, weights=weights())["total"].item()
    anchored = decision_loss(
        output(), gold, teacher_probs=teacher, teacher_mask=mask, weights=weights(distill=0)
    )["total"].item()
    legacy = decision_loss(
        output(),
        gold,
        teacher_probs=teacher,
        teacher_mask=mask,
        weights=LossWeights(brier=0, pointer_aux=0),
    )["total"].item()
    assert anchored == pytest.approx(base)
    assert legacy == pytest.approx(math.log(3) / 2)
    assert LossWeights().gold_nll_with_teacher is False


@pytest.mark.parametrize(
    "field,value",
    [("nll", 0), ("nll", -1), ("nll", float("nan")), ("distill", -1), ("distill", float("inf"))],
)
def test_cannot_disable_gold_or_use_invalid_kl_weight(field, value):
    with pytest.raises(ValueError, match="positive gold"):
        decision_loss(output(), torch.tensor([1.0, 0, 0, 1, 0]), weights=weights(**{field: value}))


@pytest.mark.parametrize(
    "teacher", [[0.2, 0.2, 0, 1, 0], [-0.1, 1.1, 0, 1, 0], [float("nan"), 0, 0, 1, 0]]
)
def test_rejects_bad_teacher_distributions(teacher):
    with pytest.raises(ValueError, match="teacher"):
        decision_loss(
            output(),
            torch.tensor([1.0, 0, 0, 1, 0]),
            teacher_probs=torch.tensor(teacher),
            teacher_mask=torch.tensor([True, False]),
            weights=weights(),
        )
