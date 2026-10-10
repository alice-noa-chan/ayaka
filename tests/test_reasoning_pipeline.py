from decimal import Decimal
from types import SimpleNamespace

import pytest
import torch

from ayaka.collate import EncodedQuestion, full_rows
from ayaka.config import tiny_config
from ayaka.model.decision import PRIMITIVE_INDEX, AyakaDecisionModel
from ayaka.model.ragged import ragged_softmax
from ayaka.primitives import Decision, DecisionResult, QuestionSpec
from ayaka.reasoning import ReasoningSettings
from ayaka.reasoning_pipeline import (
    ControlledDecision,
    Trace,
    TraceFailure,
    TraceGenerator,
    readout_suffix,
    trace_messages,
)
from ayaka.serve import DecisionService
from ayaka.tokenization import ToyTokenizer


class Tok(ToyTokenizer):
    def decode(self, ids):
        return " ".join(map(str, ids))


class Original:
    def __init__(self):
        self.model = SimpleNamespace(cfg=tiny_config(version=2))
        self.tok, self.max_seq_len, self.calls = Tok(), 8192, []

    def decide(self, state, questions, device=None):
        self.calls.append([q.instruction for q in questions])
        return [
            DecisionResult(q.type, [0.99, 0.01], dict(zip(q.candidates, [0.99, 0.01], strict=True)))
            for q in questions
        ]


class Generator:
    def __init__(self, failure=False):
        self.budgets, self.failure = [], failure

    def generate_trace(self, messages, budget, reserve=0):
        self.budgets.append(budget)
        t = Trace(text="Complete worked steps", token_ids=[8, 9, 10], finish_reason="eos")
        if self.failure:
            t.error, t.finish_reason = "decode failed", "generation_error"
            raise TraceFailure(t)
        return t

    def readout(self, trace, spec):
        return [0.1, 0.9]


@pytest.mark.parametrize("mode", ["off", "on"])
def test_decimal_score_levels_survive_direct_and_reasoning_readouts(mode, monkeypatch):
    model = AyakaDecisionModel.from_config(tiny_config(version=2), dtype=torch.float32).eval()
    direct = Decision(model, Tok())
    monkeypatch.setattr(direct, "_run", lambda *args: [[0.1, 0.9]])
    spec = QuestionSpec(
        "score", "Rounded amount?", ["0.01", "0.02"], [Decimal("0.01"), Decimal("0.02")]
    )
    result = ControlledDecision(direct, Generator()).decide(
        "Amount", [spec], reasoning=[ReasoningSettings(mode=mode)]
    )[0]
    assert result.expected == pytest.approx(0.019)
    assert result.extras["reasoning"]["route"] == ("direct" if mode == "off" else "reasoned")


def test_forced_high_bypasses_router_confidence_and_numeric_gate():
    original, gen = Original(), Generator()

    class RejectRouter:
        def should_reason(self, *args):
            pytest.fail("forced requests must bypass the router")

    d = ControlledDecision(original, gen, RejectRouter())
    qs = [QuestionSpec("noul", "Is the greeting polite?", ["no", "yes"])]
    result = d.decide("Hello!", qs, reasoning=[ReasoningSettings(mode="on", effort="high")])[0]
    assert gen.budgets == [1024]
    assert original.calls == []
    assert result.probs == [0.1, 0.9]
    assert result.extras["reasoning"]["budget"] == 1024
    assert result.extras["reasoning"]["finish_reason"] == "eos"


@pytest.mark.parametrize(
    "setting",
    [ReasoningSettings(mode="off", effort="high"), ReasoningSettings(mode="on", max_tokens=0)],
)
def test_disabled_never_generates(setting):
    o, g = Original(), Generator()
    result = ControlledDecision(o, g).decide(
        "Hello", [QuestionSpec("noul", "Hi?", ["no", "yes"])], reasoning=[setting]
    )[0]
    assert not g.budgets
    assert result.extras["reasoning"]["generated_tokens"] == 0
    assert result.probs == [0.99, 0.01]


def test_http_inheritance_isolation_and_failed_tokens_are_counted():
    o, g = Original(), Generator(failure=True)
    service = DecisionService(ControlledDecision(o, g), "fake")
    body = {
        "state": "Hello",
        "options": {"reasoning": {"mode": "on", "effort": "high"}},
        "questions": {
            "a": {"type": "noul"},
            "b": {"type": "noul", "reasoning": {"max_tokens": 4}},
            "c": {"type": "noul", "reasoning": {"mode": "off"}},
        },
    }
    response = service.handle(body)
    assert g.budgets == [1024, 4]
    assert response["usage"]["reasoning_tokens"] == response["usage"]["output_tokens"] == 6
    assert response["reasoning"]["a"]["route"] == "fallback"
    assert response["reasoning"]["a"]["settings"]["effort"] == "high"
    service.handle({"state": "Hello", "questions": {"a": {"type": "noul"}}})
    assert g.budgets == [1024, 4]  # no request settings leak into default auto


def test_real_cache_continuation_equals_full_readout_and_is_permutation_invariant():
    torch.set_num_threads(1)
    model = AyakaDecisionModel.from_config(tiny_config(version=2), dtype=torch.float32).eval()
    tok, gen = Tok(), None
    gen = TraceGenerator(model, tok)
    gen.eos = set()
    q = QuestionSpec("choice", "Intent?", ["refund", "greeting", "complaint"])
    messages = trace_messages("A polite greeting", q)
    trace = gen.generate_trace(messages, 3, reserve=200)
    rendered = readout_suffix(tok, q)
    item = EncodedQuestion(trace.input_ids + trace.token_ids, rendered, PRIMITIVE_INDEX[q.type])
    with torch.no_grad():
        expected = ragged_softmax(
            model(full_rows([item], tok.pad_id), apply_temperature=True).logits,
            torch.tensor([0, 3]),
        ).tolist()
    assert gen.readout(trace, q) == pytest.approx(expected, abs=1e-5)
    perm = QuestionSpec("choice", q.instruction, list(reversed(q.candidates)))
    assert trace_messages("A polite greeting", perm) == messages
    other = gen.generate_trace(messages, 3, reserve=200)
    assert gen.readout(other, perm) == pytest.approx(list(reversed(expected)), abs=1e-5)


def test_context_failure_preserves_requested_budget_and_eos_count():
    torch.set_num_threads(1)
    model = AyakaDecisionModel.from_config(tiny_config(version=2), dtype=torch.float32).eval()
    gen = TraceGenerator(model, Tok(), max_context=8)
    with pytest.raises(TraceFailure) as exc:
        gen.generate_trace([{"role": "user", "content": "Hello"}], 1024)
    assert exc.value.trace.finish_reason == "context_limit"
    assert exc.value.trace.generated_tokens == 0
    gen.max_context = 4096
    trace = gen.generate_trace([{"role": "user", "content": "Hello"}], 1)
    gen.eos = {trace.token_ids[0]}
    ended = gen.generate_trace([{"role": "user", "content": "Hello"}], 10)
    assert ended.generated_tokens == 1 and ended.finish_reason == "eos" and ended.text == ""


def test_large_candidate_rerank_keeps_mass_and_separate_questions_have_separate_caches():
    torch.set_num_threads(1)
    model = AyakaDecisionModel.from_config(tiny_config(version=2), dtype=torch.float32).eval()
    gen = TraceGenerator(model, Tok())
    gen.eos = set()
    q = QuestionSpec("choice", "Which?", [f"candidate {i}" for i in range(28)])
    first = gen.generate_trace(trace_messages("State", q), 1, reserve=600)
    second = gen.generate_trace(trace_messages("Other state", q), 1, reserve=600)
    assert first.cache is not second.cache
    p = gen.readout(first, q)
    assert len(p) == 28 and sum(p) == pytest.approx(1, abs=1e-5)


def test_auto_router_receives_requested_budget_and_never_changes_it():
    o, g = Original(), Generator()
    seen = []

    class Router:
        def should_reason(self, state, spec, baseline, tok, budget):
            seen.append((baseline.probs, budget))
            return True

    d = ControlledDecision(o, g, Router())
    d.decide(
        "Hello",
        [QuestionSpec("noul", "Hi?", ["no", "yes"])],
        reasoning=[ReasoningSettings(mode="auto", effort="high", max_tokens=1000)],
    )
    assert seen == [([0.99, 0.01], 1000)]
    assert g.budgets == [1000]


def test_auto_without_validated_router_stays_direct_even_for_uncertain_calculation(monkeypatch):
    original, generator = Original(), Generator()
    spec = QuestionSpec("noul", "Does 1 + 1 equal 2?", ["no", "yes"])
    monkeypatch.setattr(
        original,
        "decide",
        lambda *args, **kwargs: [DecisionResult("noul", [0.5, 0.5], {"no": 0.5, "yes": 0.5})],
    )
    result = ControlledDecision(original, generator).decide("Calculate 1 + 1.", [spec])[0]
    assert generator.budgets == []
    assert result.extras["reasoning"]["budget"] == 384
    assert result.extras["reasoning"]["route"] == "direct"
    assert result.extras["reasoning"]["finish_reason"] == "no_validated_router"


def test_calibration_cannot_change_router_features_and_direct_uses_zero_budget():
    from ayaka.data.schema import Question, Sample
    from ayaka.eval.reasoning_v2 import evaluate_efforts

    seen = []

    class Calibration:
        def apply(self, probs, kind, route, budget):
            seen.append((route, budget))
            return [0.5, 0.5]

    decision = ControlledDecision(Original(), Generator(), calibration=Calibration())
    sample = Sample(
        "Hello", [Question.noul("q", "Hi?", 1)], {"source_example_id": "test", "split": "test"}
    )
    report = evaluate_efforts(decision, [sample], efforts=(), include_auto=True)
    assert report["rows"]["off"][0]["routing_features"][0] == 0.99
    assert report["rows"]["off"][0]["probs"] == [0.5, 0.5]
    assert seen == [("direct", 0), ("direct", 0)]
    assert report["rows"]["auto"][0]["finish_reason"] == "no_validated_router"


def test_always_types_reason_without_asking_the_router():
    asked = []

    class Router:
        def should_reason(self, state, spec, baseline, tok, budget):
            asked.append(spec.type)
            return False

    o, g = Original(), Generator()
    d = ControlledDecision(o, g, Router(), always_types=("noul",))
    noul, choice = (
        QuestionSpec("noul", "Hi?", ["no", "yes"]),
        QuestionSpec("choice", "Pick", ["a", "b"]),
    )
    results = d.decide("Hello", [noul, choice])
    assert asked == ["choice"]
    assert g.budgets == [384]
    assert [r.extras["reasoning"]["route"] for r in results] == ["reasoned", "direct"]
    assert results[0].extras["reasoning"]["router"] == "always"


@pytest.mark.parametrize("mode", ["off", "on"])
def test_always_types_never_override_an_explicit_mode(mode):
    o, g = Original(), Generator()
    d = ControlledDecision(o, g, always_types=("noul",))
    result = d.decide(
        "Hello",
        [QuestionSpec("noul", "Hi?", ["no", "yes"])],
        reasoning=[ReasoningSettings(mode=mode)],
    )[0]
    assert result.extras["reasoning"]["route"] == ("reasoned" if mode == "on" else "direct")


def test_direct_correction_can_stay_on_under_path_calibration():
    from ayaka.reasoning_pipeline import controlled_decision

    model = AyakaDecisionModel.from_config(tiny_config(version=2), dtype=torch.float32).eval()
    calibration = SimpleNamespace(apply=lambda probs, *args: probs)
    default = controlled_decision(model, Tok(), calibration=calibration)
    kept = controlled_decision(model, Tok(), calibration=calibration, direct_correction=True)
    assert default.original.apply_probability_calibration is False
    assert kept.original.apply_probability_calibration is True
