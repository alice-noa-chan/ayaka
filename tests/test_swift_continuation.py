"""Native generation cues may differ from canonical assistant history turns."""

from types import SimpleNamespace

import pytest
import torch

import ayaka.swift_continuation as continuation
from ayaka.primitives import QuestionSpec
from ayaka.reasoning import ReasoningSettings
from ayaka.reasoning_pipeline import ControlledDecision, Trace
from tests.test_reasoning_pipeline import Original


class NativeTemplate:
    def __init__(self, generation_channel=True, rewrite_user=False):
        self.generation_channel = generation_channel
        self.rewrite_user = rewrite_user

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        pieces = []
        for message in messages:
            role = "model" if message["role"] == "assistant" else message["role"]
            content = message["content"]
            if self.rewrite_user and len(messages) > 1 and not pieces:
                content = "Altered evidence"
            pieces.append(f"<{role}>{content}</{role}>\n")
        if add_generation_prompt:
            pieces.append("<model>" + ("<empty-thought>" if self.generation_channel else ""))
        text = "".join(pieces)
        return list(text.encode()) if tokenize else text


class Model:
    def __init__(self):
        self.calls = []

    def embed_weight(self):
        return torch.zeros(1)

    def __call__(self, batch, **kwargs):
        self.calls.append((batch, kwargs))
        return SimpleNamespace(logits=torch.tensor([0.0, 1.0]), cand_cu=torch.tensor([0, 2]))


def generator(monkeypatch, **kwargs):
    # The tokenizer identity contract has its own native serialization tests;
    # these fixtures exercise actual continuation token-prefix decisions.
    monkeypatch.setattr(continuation, "validate_tokenizer", lambda *args: None)
    result = continuation.SwiftTraceGenerator.__new__(continuation.SwiftTraceGenerator)
    result.tok = SimpleNamespace(hf=NativeTemplate(**kwargs), pad_id=0)
    result.encoding = {"chat_template_kwargs": {"enable_thinking": False}}
    result.contract = {}
    result.model = Model()
    result.max_context, result.apply_temperature = 8192, True
    result._tokens = lambda messages, letters: {
        "input_token_ids": result.tok.hf.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        ),
        "canonical_token_ids": {letter: [ord(letter)] for letter in letters},
    }
    return result


def trace(gen):
    messages = [{"role": "user", "content": "Original evidence"}]
    text = "Complete worked steps"
    result = Trace(
        text=text,
        input_ids=gen.prepare(messages)[0],
        token_ids=[*text.encode(), 999],
        finish_reason="eos",
    )
    result.messages = messages
    result.cache = object()
    return result


def spec():
    return QuestionSpec("choice", "Select one", ["Beta", "Alpha"], candidate_ids=["b", "a"])


@pytest.mark.parametrize(
    ("question", "labels"),
    [
        (spec(), [ord("A"), ord("B")]),
        (
            QuestionSpec("score", "Rate", ["Nine", "Seven"], [9, 7], ["9", "7"]),
            [ord("B"), ord("A")],
        ),
    ],
)
def test_generation_only_channel_uses_complete_canonical_read_and_original_order(
    monkeypatch, question, labels
):
    gen = generator(monkeypatch)
    observed = trace(gen)
    expected = gen.prepare(gen._final_messages(observed.messages, observed.text))[0]
    probabilities = gen.readout(observed, question)
    batch, kwargs = gen.model.calls[0]
    assert batch.input_ids[0].tolist() == expected
    assert kwargs == {"past_key_values": None, "apply_temperature": True}
    assert observed.cache is None and observed.readout_execution == "canonical_full_read"
    assert observed.readout_tokens == len(expected)
    assert observed.readout_label_ids == labels
    assert probabilities == pytest.approx(torch.softmax(torch.tensor([0.0, 1.0]), 0).tolist())


def test_unchanged_template_keeps_exact_cache_reuse(monkeypatch):
    gen = generator(monkeypatch, generation_channel=False)
    observed = trace(gen)
    cache = observed.cache
    prefix = len(observed.input_ids) + len(observed.token_ids) - 1
    expected = gen.prepare(gen._final_messages(observed.messages, observed.text))[0]
    gen.readout(observed, spec())
    batch, kwargs = gen.model.calls[0]
    assert kwargs["past_key_values"] is cache
    assert batch.input_ids[0].tolist() == expected[prefix:]
    assert observed.readout_execution == "cache_reuse"
    assert observed.readout_tokens == len(expected) - prefix


def test_rewritten_original_evidence_is_rejected_before_model_read(monkeypatch):
    gen = generator(monkeypatch, rewrite_user=True)
    observed = trace(gen)
    with pytest.raises(ValueError, match="original conversation"):
        gen.readout(observed, spec())
    assert gen.model.calls == []


def test_canonical_full_read_never_clips_an_overlong_final_chat(monkeypatch):
    gen = generator(monkeypatch)
    observed = trace(gen)
    gen.max_context = len(gen.prepare(gen._final_messages(observed.messages, observed.text))[0]) - 1
    with pytest.raises(ValueError, match="refuse truncation"):
        gen.readout(observed, spec())
    assert gen.model.calls == []


def test_explicit_full_read_usage_reaches_controlled_decision(monkeypatch):
    gen = generator(monkeypatch)
    gen.generate_trace = lambda messages, budget, reserve: trace(gen)
    gen.messages_for = lambda state, question: [{"role": "user", "content": "Original evidence"}]
    gen.reserve_tokens = lambda *args: 0
    result = ControlledDecision(Original(), gen).decide(
        "Original evidence", [spec()], reasoning=[ReasoningSettings(mode="on", effort="medium")]
    )[0]
    assert result.extras["reasoning"]["route"] == "reasoned"
    assert result.extras["reasoning"]["readout_execution"] == "canonical_full_read"
