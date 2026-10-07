"""Ragged ordinal scans must preserve the loss and both input gradients."""

import pytest
import torch

from ayaka.losses import rps_loss
from ayaka.model.ragged import per_question_sum, ragged_log_softmax, ragged_max, seg_ids


def reference_rps(probs, targets, cu, ordinals, score_mask):
    losses = []
    for index, (start, end) in enumerate(zip(cu[:-1], cu[1:], strict=True)):
        if not score_mask[index] or end - start < 2:
            continue
        order = ordinals[start:end].argsort()
        predicted = probs[start:end][order].cumsum(0)[:-1]
        expected = targets[start:end][order].cumsum(0)[:-1]
        losses.append((predicted - expected).square().mean())
    return torch.stack(losses).mean() if losses else (probs.sum() + targets.sum()) * 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("lengths", [[], [1], [1, 2, 7, 3, 1, 5], [3, 9, 2, 6] * 128])
@pytest.mark.parametrize("selected", ["mixed", "all", "none"])
def test_rps_matches_reference_values_and_gradients(dtype, lengths, selected):
    rng = torch.Generator().manual_seed(81)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.long)
    count = sum(lengths)
    logits = torch.randn(count, generator=rng, dtype=dtype)
    target_logits = torch.randn(count, generator=rng, dtype=dtype)
    probs = ragged_log_softmax(logits, cu).exp().detach().requires_grad_()
    targets = ragged_log_softmax(target_logits, cu).exp().detach().requires_grad_()
    ordinals = torch.cat([torch.randperm(k, generator=rng) for k in lengths]) if lengths else cu[:0]
    mask = torch.tensor(
        [selected == "all" or (selected == "mixed" and i % 3 == 0) for i in range(len(lengths))],
        dtype=torch.bool,
    )
    actual = rps_loss(probs, targets, cu, ordinals, mask)
    expected = reference_rps(probs, targets, cu, ordinals, mask)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, (probs, targets))
    expected_grad = torch.autograd.grad(expected, (probs, targets))
    for got, want in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(
            got, want, atol=1e-8 if dtype == torch.float64 else 1e-6, rtol=1e-5
        )


def test_known_candidate_counts_preserve_ragged_reductions_with_empty_segments():
    cu = torch.tensor([0, 2, 2, 5])
    values = torch.tensor([1.0, 3.0, -2.0, 0.0, 2.0])
    assert torch.equal(seg_ids(cu), seg_ids(cu, output_size=values.numel()))
    torch.testing.assert_close(per_question_sum(values, cu), torch.tensor([4.0, 0.0, 0.0]))
    torch.testing.assert_close(ragged_max(values, cu), torch.tensor([3.0, -torch.inf, 2.0]))
    probs = ragged_log_softmax(values, cu).exp()
    torch.testing.assert_close(per_question_sum(probs, cu), torch.tensor([1.0, 0.0, 1.0]))
