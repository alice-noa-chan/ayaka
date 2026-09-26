"""Sparse option pooling, legacy readout parity and length-specific training."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from ayaka.collate import encode_decision, full_rows
from ayaka.config import tiny_config
from ayaka.model.electra import ElectraDecisionModel, span_means
from ayaka.prompt import QuestionView
from ayaka.tokenization import ToyTokenizer


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("normalize", [False, True])
def test_sparse_pool_matches_independent_local_means_and_gradients(dtype, normalize):
    torch.manual_seed(4)
    hidden = torch.randn(3, 128, 16, dtype=dtype, requires_grad=True)
    rows = torch.tensor([0, 2, 0, 1, 2])
    spans = torch.tensor([[110, 117], [121, 128], [114, 119], [10, 10], [0, 128]])
    norm = torch.nn.RMSNorm(16).to(dtype) if normalize else None
    got = span_means(hidden, rows, spans, norm=norm)
    reference = []
    for row, (start, end) in zip(rows.tolist(), spans.tolist(), strict=True):
        selected = hidden[row, start:end]
        selected = norm(selected) if norm else selected
        selected = selected.to(torch.promote_types(dtype, torch.float32))
        reference.append(selected.sum(0) / max(end - start, 1))
    reference = torch.stack(reference)
    assert got.dtype == reference.dtype
    assert torch.allclose(got, reference, atol=2e-6, rtol=2e-6)
    weights = torch.randn_like(got)
    grad = torch.autograd.grad((got * weights).sum(), hidden, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((reference * weights).sum(), hidden)[0]
    assert torch.allclose(grad, expected_grad, atol=2e-6, rtol=2e-6)


def test_sparse_pool_does_not_lose_small_options_after_large_prefix():
    hidden = torch.full((1, 4096, 8), 10000.0)
    hidden[:, -3:] = 0.125
    got = span_means(hidden, torch.tensor([0]), torch.tensor([[4093, 4096]]))
    assert torch.equal(got, torch.full((1, 8), 0.125))
    assert span_means(
        hidden, torch.empty(0, dtype=torch.long), torch.empty(0, 2, dtype=torch.long)
    ).shape == (0, 8)
    assert torch.equal(
        span_means(hidden, torch.tensor([0]), torch.tensor([[4096, 4096]])), torch.zeros(1, 8)
    )


def model_and_batch():
    torch.manual_seed(1)
    model = ElectraDecisionModel.from_config(
        tiny_config(long_prompt_tokens=256), dtype=torch.float32
    )
    tok = ToyTokenizer()
    _, encoded = encode_decision(
        {"status": "received", "amount": 300},
        [QuestionView("choice", "Route?", ["finance", "support", "sales"])] * 2,
        tok,
        max_seq_len=512,
    )
    batch = replace(full_rows(encoded, tok.pad_id), seq_len=torch.tensor([255, 256]))
    return model, batch


def test_gate_legacy_load_preserves_both_length_buckets_and_logits():
    model, batch = model_and_batch()
    legacy = {
        **model.head_state_dict(),
        "gate": torch.tensor([0.1, 0.7, -0.2]),
        "temperature": torch.tensor([1.1, 1.3, 0.9]),
    }
    model.load_head_state_dict(legacy)
    assert torch.allclose(model.gate, torch.tensor([[0.1, 0.1], [0.7, 0.7], [-0.2, -0.2]]))
    model.eval()
    with torch.no_grad():
        top, spans, view = model.encode(batch)  # original normalized state contract
        # Reproduce the previous full-sequence cumulative-sum pooling.
        cumulative = torch.nn.functional.pad(spans.float().cumsum(1), (0, 0, 1, 0))
        starts, ends = view.cand_spans.unbind(1)
        legacy_pool = (
            cumulative[view.cand_question, ends] - cumulative[view.cand_question, starts]
        ) / (ends - starts).unsqueeze(1)
        reference_ptr = model.head(
            legacy_pool,
            top[torch.arange(2), view.answer_pos],
            view.cand_cu,
            view.cand_question,
        )
        got = model(batch, apply_temperature=True)
        reference = (
            got.label_logits + legacy["gate"][batch.primitive[batch.cand_question]] * reference_ptr
        ) / 1.3
    assert torch.allclose(got.logits, reference, atol=2e-6)


def test_long_gate_changes_only_long_question_and_both_buckets_train():
    model, batch = model_and_batch()
    model.gate.data[1] = torch.tensor([0.2, 0.8])
    model.eval()
    out = model(batch)
    short, long = out.logits[:3], out.logits[3:]
    assert torch.allclose(short, out.label_logits[:3] + 0.2 * out.pointer_logits[:3])
    assert torch.allclose(long, out.label_logits[3:] + 0.8 * out.pointer_logits[3:])
    assert torch.allclose(out.pointer_logits[:3], out.pointer_logits[3:], atol=1e-6)
    out.logits.square().sum().backward()
    assert model.gate.grad[1, 0].abs() > 0
    assert model.gate.grad[1, 1].abs() > 0
    assert model.gate.grad[[0, 2]].eq(0).all()


def test_zero_gate_eval_skips_pointer_but_train_grad_and_pointer_only_keep_it():
    model, batch = model_and_batch()
    model.eval()
    with (
        torch.no_grad(),
        patch.object(model.head, "forward", side_effect=AssertionError("unused pointer called")),
    ):
        skipped = model(batch)
    assert torch.equal(skipped.logits, skipped.label_logits)
    model.train()
    with patch.object(model.head, "forward", wraps=model.head.forward) as called:
        reference = model(batch)
        assert called.call_count == 1
    assert torch.equal(skipped.logits, reference.logits)
    reference.logits.square().sum().backward()
    assert model.gate.grad[1].ne(0).all()  # zero gates still receive learning signal
    model.eval()
    with patch.object(model.head, "forward", wraps=model.head.forward) as called:
        model(batch)  # eval with gradients enabled also retains pointer gradients
        assert called.call_count == 1
    pointer_only = replace(batch, has_label=torch.tensor([False, True]))
    with torch.no_grad(), patch.object(model.head, "forward", wraps=model.head.forward) as called:
        out = model(pointer_only)
        assert called.call_count == 1
    assert torch.equal(out.logits[:3], out.pointer_logits[:3])
    assert torch.equal(out.logits[3:], out.label_logits[3:])


def test_sparse_norm_forward_matches_full_norm_forward_and_gradients():
    model, batch = model_and_batch()
    model.gate.data.fill_(0.6)
    model.eval()
    got = model(batch)
    top, spans, view = model.encode(batch)
    reference = model.decide(top, view, span_hidden=spans)
    assert torch.allclose(got.logits, reference.logits, atol=2e-6)
    params = (
        model.text_model().norm.weight,
        model.head.in_r.weight,
        model.text_model().layers[0].self_attn.q_proj.weight,
    )
    got_grad = torch.autograd.grad(got.logits.square().sum(), params)
    reference_grad = torch.autograd.grad(reference.logits.square().sum(), params)
    for actual, expected in zip(got_grad, reference_grad, strict=True):
        assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
