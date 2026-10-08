"""The no-sync decode loop keeps exactly the tokens the per-token-sync loop keeps."""

import pytest
import torch
from test_checkpoint_serving_inputs import QUESTIONS, STATE, setup_checkpoint

from ayaka.eval import generation_bench as bench
from ayaka.model.fastpath import prefill_last
from ayaka.reasoning_pipeline import controlled_decision
from ayaka.serve import parse_question


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def eos_after(generator, model, tok, messages, steps):
    """Steer the real tiny model: a trace token, then a native EOS after ``steps`` calls."""
    ids, _ = generator.prepare(messages)
    trace_token = tok.hf.encode("~", add_special_tokens=False)[0]
    eos_token = tok.hf.encode("@", add_special_tokens=False)[0]
    text = model.text_model()
    embedding = text.get_input_embeddings().weight
    with torch.inference_mode():
        hidden, _ = prefill_last(text, torch.tensor([ids]))
        embedding[trace_token].copy_(hidden[0] * 100)
    generator.eos = {eos_token}
    calls = 0

    def hook(module, args, output):
        nonlocal calls
        calls += 1
        if calls == steps:
            with torch.inference_mode():
                embedding[eos_token].copy_(output.last_hidden_state[0, -1] * 10000)

    return text.register_forward_hook(hook), eos_token


@pytest.mark.parametrize("sync_every", [1, 5, 7, 64])
def test_nosync_matches_generate_trace_at_eos(tmp_path, sync_every):
    _, model, tok, _, _ = setup_checkpoint(tmp_path, "gemma")
    generator = controlled_decision(model, tok).generator
    messages = generator.messages_for(STATE, parse_question(QUESTIONS["pick"])[0])
    state = {name: t.clone() for name, t in model.state_dict().items()}

    handle, eos = eos_after(generator, model, tok, messages, 12)
    try:
        expected = generator.generate_trace(messages, 40).token_ids
    finally:
        handle.remove()
    model.load_state_dict(state)
    handle, _ = eos_after(generator, model, tok, messages, 12)
    try:
        got = bench.generate_nosync(generator, messages, 40, sync_every)
    finally:
        handle.remove()
    assert expected[-1] == eos and eos not in expected[:-1] and len(expected) > 7
    assert got == list(expected)


@pytest.mark.parametrize("budget", [1, 6, 16])
def test_nosync_matches_generate_trace_at_budget(tmp_path, budget):
    _, model, tok, _, _ = setup_checkpoint(tmp_path, "gemma")
    generator = controlled_decision(model, tok).generator
    generator.eos = set()
    messages = generator.messages_for(STATE, parse_question(QUESTIONS["judge"])[0])
    expected = generator.generate_trace(messages, budget).token_ids
    assert len(expected) == budget
    assert bench.generate_nosync(generator, messages, budget, 5) == list(expected)


def test_summary_reports_agreement_with_the_reference():
    reference = [{"seconds": 1.0, "tokens": [1, 2, 3, 4]}, {"seconds": 1.0, "tokens": [5, 6]}]
    other = [{"seconds": 0.5, "tokens": [1, 2, 9, 9]}, {"seconds": 0.5, "tokens": [5, 6]}]
    out = bench.summary(other, reference)
    assert out["exact_token_match"] == 1
    assert out["mean_common_prefix_fraction"] == pytest.approx((0.5 + 1.0) / 2)
    assert out["ms_per_token"] == pytest.approx(1000 * 1.0 / 6)
    assert "exact_token_match" not in bench.summary(reference)
