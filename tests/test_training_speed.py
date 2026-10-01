import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ayaka.training.reasoning import trace_ce


@pytest.mark.parametrize("chunk", [1, 32, 128])
def test_batched_trace_projection_preserves_row_weighting_and_gradients(chunk):
    torch.manual_seed(3)
    projection = torch.nn.Linear(8, 19)
    model = SimpleNamespace(lm_logits=projection)
    hidden = torch.randn(3, 12, 8, requires_grad=True)
    items = [
        SimpleNamespace(reasoning_positions=[1, 2], reasoning_labels=[4, 5]),
        SimpleNamespace(reasoning_positions=None, reasoning_labels=None),
        SimpleNamespace(reasoning_positions=[2, 3, 4, 5, 6], reasoning_labels=[2, 3, 4, 5, 6]),
    ]
    expected = sum(
        F.cross_entropy(
            projection(hidden[row, item.reasoning_positions]), torch.tensor(item.reasoning_labels)
        )
        for row, item in enumerate(items)
        if item.reasoning_labels
    ) / len(items)
    reference = torch.autograd.grad(expected, (hidden, *projection.parameters()))
    actual = trace_ce(model, hidden, items, chunk)
    gradients = torch.autograd.grad(actual, (hidden, *projection.parameters()))
    torch.testing.assert_close(actual, expected)
    for a, b in zip(gradients, reference, strict=True):
        torch.testing.assert_close(a, b)


def test_proposal_batch_matches_isolated_rows_with_different_lengths_and_one_forward():
    from ayaka.checkpoint import apply_lora
    from ayaka.config import tiny_config
    from ayaka.data.candidate_v2 import candidate_curriculum
    from ayaka.model.electra import ElectraDecisionModel
    from ayaka.tokenization import ToyTokenizer
    from ayaka.training.candidates import proposal_ce, proposal_items

    torch.set_num_threads(1)
    cfg, tok = tiny_config(version=2, max_seq_len=4096), ToyTokenizer()
    model = ElectraDecisionModel.from_config(cfg, dtype=torch.float32)
    model.backbone.requires_grad_(False)
    apply_lora(model)
    model.eval()  # deterministic parity; gradients remain enabled
    items = [proposal_items(sample, tok, cfg)[0] for sample in candidate_curriculum("train", 2)]
    items += [replace(items[0], proposal_labels=None)]
    original = copy.deepcopy(model)
    expected = 0
    for item in items:
        if item.proposal_labels:
            hidden = original.backbone(
                input_ids=torch.tensor([item.proposal_input_ids]), use_cache=False
            ).last_hidden_state
            proxy = replace(
                item,
                reasoning_labels=item.proposal_labels,
                reasoning_positions=item.proposal_positions,
            )
            expected = expected + trace_ce(original, hidden, [proxy], 32)
    expected = expected / len(items)
    expected.backward()
    calls = []
    hook = model.backbone.register_forward_hook(lambda *args: calls.append(1))
    actual = proposal_ce(model, items, tok.pad_id)
    actual.backward()
    hook.remove()
    assert calls == [1]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for a, b in zip(model.parameters(), original.parameters(), strict=True):
        if b.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4)
