import math
from dataclasses import replace

import pytest
import torch

from ayaka.model.evidence import EvidenceOutput, EvidenceResidualHead
from ayaka.training.evidence_objective import EvidenceLossWeights, evidence_loss


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def output(logits, mask=None, dtype=torch.float64):
    logits = torch.tensor(logits, dtype=dtype, requires_grad=True)
    mask = (
        torch.ones_like(logits, dtype=torch.bool)
        if mask is None
        else torch.tensor(mask, dtype=torch.bool)
    )
    return EvidenceOutput(logits, torch.zeros_like(logits), mask, mask.any(-1))


def ce_only(**kwargs):
    return EvidenceLossWeights(brier=0, rps=0, **kwargs)


def test_soft_gold_ce_and_brier_are_retained():
    out = output([[[2.0, -1.0]]])
    gold = torch.tensor([[[0.7, 0.3]]], dtype=torch.float64)
    result = evidence_loss(out, gold, torch.tensor([[0]]))
    expected_ce = -(gold * out.log_probs()).sum()
    expected_br = ((out.probs() - gold) ** 2).sum()
    assert result["nll"].item() == pytest.approx(expected_ce.item())
    assert result["brier"].item() == pytest.approx(expected_br.item())
    assert result["total"].item() == pytest.approx((expected_ce + 0.5 * expected_br).item())
    assert result["total"].requires_grad
    assert not result["nll"].requires_grad


def test_reference_is_additive_and_cannot_replace_gold():
    out = output([[[1.0, 0.0]]])
    gold = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
    ref = torch.tensor([[[0.0, 3.0]]], dtype=torch.float64, requires_grad=True)
    weight = 0.2
    result = evidence_loss(
        out, gold, torch.tensor([[0]]), reference_logits=ref, weights=ce_only(reference=weight)
    )
    p = ref.softmax(-1).detach()
    expected_kl = (p * (ref.detach().log_softmax(-1) - out.log_probs())).sum()
    assert result["reference_kl"].item() == pytest.approx(expected_kl.item())
    assert result["total"].item() == pytest.approx(
        result["nll"].item() + weight * expected_kl.item()
    )
    result["total"].backward()
    assert ref.grad is None
    assert out.logits.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="gold nll"):
        evidence_loss(
            out,
            gold,
            torch.tensor([[0]]),
            reference_logits=ref,
            weights=EvidenceLossWeights(nll=0, reference=1),
        )


def test_reference_kl_direction_is_reference_to_student():
    out = output([[[0.0, -2.0]]])
    gold = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
    ref = torch.tensor([[[-1.0, 0.0]]], dtype=torch.float64)
    result = evidence_loss(
        out, gold, torch.tensor([[0]]), reference_logits=ref, weights=ce_only(reference=1)
    )
    p, q = ref.softmax(-1), out.probs()
    forward = (p * (p.log() - q.log())).sum().item()
    reverse = (q * (q.log() - p.log())).sum().item()
    assert result["reference_kl"].item() == pytest.approx(forward)
    assert abs(forward - reverse) > 0.01


def test_identical_reference_zero_kl():
    out = output([[[7.0, -1000.0]]])
    gold = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
    result = evidence_loss(
        out, gold, torch.tensor([[0]]), reference_logits=out.logits, weights=ce_only(reference=1)
    )
    assert result["reference_kl"].item() == 0
    assert result["nll"].item() == pytest.approx(1007)
    result["total"].backward()
    assert torch.isfinite(out.logits.grad).all()


def test_reference_student_tiny_tail_is_not_floored():
    out = output([[[0.0, -1000.0]]], dtype=torch.float32)
    gold = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
    ref = torch.tensor([[[-1000.0, 0.0]]], dtype=torch.float64)
    result = evidence_loss(
        out, gold, torch.tensor([[0]]), reference_logits=ref, weights=ce_only(reference=1)
    )
    assert result["nll"].item() == pytest.approx(1000)
    assert result["reference_kl"].item() == pytest.approx(1000)
    assert result["total"].dtype == torch.float64
    result["total"].backward()
    assert torch.equal(out.logits.grad, torch.tensor([[[2.0, -2.0]]]))


def test_smoothing_is_opt_in_and_only_changes_ce():
    out = output([[[2.0, -1.0]]])
    gold = torch.tensor([[[0.7, 0.3]]], dtype=torch.float64)
    plain = evidence_loss(out, gold, torch.tensor([[0]]))
    smooth = evidence_loss(out, gold, torch.tensor([[0]]), label_smoothing=0.1)
    expected = -((0.9 * gold + 0.1 / 2) * out.log_probs()).sum()
    assert smooth["nll"].item() == pytest.approx(expected.item())
    assert plain["nll"].item() != smooth["nll"].item()
    assert torch.equal(plain["brier"], smooth["brier"])


def test_padding_not_counted_and_masked_nan_is_safe():
    out = output(
        [[[0.0, 0.0, math.nan], [math.nan, math.nan, math.nan]]],
        [[[True, True, False], [False, False, False]]],
    )
    gold = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float64)
    ref = torch.tensor(
        [[[0.0, 0.0, math.nan], [math.nan, math.nan, math.nan]]], dtype=torch.float64
    )
    result = evidence_loss(
        out, gold, torch.tensor([[0, -1]]), reference_logits=ref, weights=ce_only(reference=0.1)
    )
    assert result["nll"].item() == pytest.approx(math.log(2))
    assert result["reference_kl"].item() == 0
    result["total"].backward()
    assert torch.isfinite(out.logits.grad).all()
    assert (out.logits.grad[~out.candidate_mask] == 0).all()


def test_rps_uses_sorted_explicit_levels_and_preserves_soft_gold():
    probs = torch.tensor([[[0.2, 0.5, 0.3]]], dtype=torch.float64)
    out = output(probs.log().tolist())
    gold = torch.tensor([[[0.1, 0.2, 0.7]]], dtype=torch.float64)
    ordinals = torch.tensor([[[5.0, 0.0, 1.0]]], dtype=torch.float64)
    result = evidence_loss(out, gold, torch.tensor([[2]]), ordinals=ordinals)
    # Sorted probabilities [.5, .3, .2], gold [.2, .7, .1]. CDF errors .3, -.1.
    assert result["rps"].item() == pytest.approx((0.3**2 + 0.1**2) / 2)
    perm = torch.tensor([2, 0, 1])
    shuffled = replace(
        out, logits=out.logits[..., perm], candidate_mask=out.candidate_mask[..., perm]
    )
    rerun = evidence_loss(
        shuffled, gold[..., perm], torch.tensor([[2]]), ordinals=ordinals[..., perm]
    )
    for key in result:
        assert result[key].item() == pytest.approx(rerun[key].item())
    smoothed = evidence_loss(out, gold, torch.tensor([[2]]), ordinals=ordinals, label_smoothing=0.1)
    assert torch.equal(result["rps"], smoothed["rps"])


def test_variable_candidate_counts_rps_excludes_padding_thresholds():
    out = output(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, math.nan]]], [[[True, True, True], [True, True, False]]]
    )
    gold = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]], dtype=torch.float64)
    levels = torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, math.nan]]], dtype=torch.float64)
    result = evidence_loss(out, gold, torch.tensor([[2, 2]]), ordinals=levels)
    assert result["rps"].item() == pytest.approx(((1 / 9 + 4 / 9) / 2 + 1 / 4) / 2)


def test_toy_end_to_end_training_with_frozen_native_reference():
    torch.manual_seed(12)
    head = EvidenceResidualHead(8, dim=8, heads=2)
    mask = torch.tensor(
        [[[True, True, False, False], [True, True, True, False], [True, True, True, True]]]
    )
    inputs = {
        "native_logits": torch.zeros(1, 3, 4, requires_grad=True),
        "candidates": torch.randn(1, 3, 4, 8),
        "queries": torch.randn(1, 3, 8),
        "memory": torch.randn(1, 5, 8),
        "memory_mask": torch.ones(1, 5, dtype=torch.bool),
        "candidate_mask": mask,
        "question_mask": torch.ones(1, 3, dtype=torch.bool),
        "primitive": torch.tensor([[0, 1, 2]]),
    }
    gold = torch.tensor([[[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]])
    levels = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [3.0, 1.0, 4.0, 2.0]]])
    reference = inputs["native_logits"].detach().clone().requires_grad_()

    def loss():
        return evidence_loss(
            head(**inputs),
            gold,
            inputs["primitive"],
            ordinals=levels,
            reference_logits=reference,
            weights=EvidenceLossWeights(reference=0.05),
        )

    before = loss()["nll"].item()
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.03)
    for _ in range(12):
        optimizer.zero_grad()
        loss()["total"].backward()
        optimizer.step()
    after = loss()
    assert after["nll"].item() < before * 0.5
    assert after["reference_kl"].item() > 0
    assert reference.grad is None
    assert inputs["native_logits"].grad is None


@pytest.mark.parametrize(
    "bad",
    [
        "negative",
        "sum",
        "padding",
        "nan",
        "shape",
        "primitive",
        "empty",
        "reference_missing",
        "reference_shape",
        "reference_nan",
        "noul_count",
    ],
)
def test_invalid_loss_contracts_rejected(bad):
    out = output([[[0.0, 0.0, 0.0]]], [[[True, True, False]]])
    gold = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
    primitive = torch.tensor([[0]])
    kwargs = {}
    if bad == "negative":
        gold[0, 0, 0] = -1
    elif bad == "sum":
        gold[0, 0, 0] = 0.5
    elif bad == "padding":
        gold[0, 0] = torch.tensor([0.5, 0.0, 0.5])
    elif bad == "nan":
        gold[0, 0, 0] = math.nan
    elif bad == "shape":
        gold = gold[..., :2]
    elif bad == "primitive":
        primitive.fill_(4)
    elif bad == "empty":
        out = replace(
            out,
            candidate_mask=torch.zeros_like(out.candidate_mask),
            question_mask=torch.zeros_like(out.question_mask),
        )
    elif bad == "reference_missing":
        kwargs["weights"] = EvidenceLossWeights(reference=1)
    elif bad == "reference_shape":
        kwargs["reference_logits"] = torch.zeros(1, 1, 2)
    elif bad == "reference_nan":
        kwargs["reference_logits"] = torch.full_like(out.logits, math.nan)
    else:
        out = replace(out, candidate_mask=torch.ones_like(out.candidate_mask))
    with pytest.raises(ValueError):
        evidence_loss(out, gold, primitive, **kwargs)


@pytest.mark.parametrize("bad", ["missing", "duplicate", "nan", "shape", "single_candidate"])
def test_invalid_score_levels_rejected(bad):
    out = output([[[0.0, 0.0]]])
    gold = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    levels = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
    if bad == "missing":
        levels = None
    elif bad == "duplicate":
        levels.fill_(0)
    elif bad == "nan":
        levels[0, 0, 0] = math.nan
    elif bad == "shape":
        levels = levels[..., :1]
    else:
        out = replace(out, candidate_mask=torch.tensor([[[True, False]]]))
    with pytest.raises(ValueError):
        evidence_loss(out, gold, torch.tensor([[2]]), ordinals=levels)


@pytest.mark.parametrize("bad", [-0.1, 1.0, math.nan, math.inf, True])
def test_invalid_smoothing_rejected(bad):
    with pytest.raises(ValueError, match="label_smoothing"):
        evidence_loss(
            output([[[0.0, 0.0]]]),
            torch.tensor([[[1.0, 0.0]]]),
            torch.tensor([[0]]),
            label_smoothing=bad,
        )


@pytest.mark.parametrize("bad", [-1, math.nan, math.inf, True, "1"])
def test_invalid_weights_rejected(bad):
    with pytest.raises(ValueError, match="weights"):
        EvidenceLossWeights(reference=bad).validate()
