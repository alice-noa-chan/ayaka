import pytest

from ayaka.evidence_ids import id_messages
from ayaka.evidence_pipeline import EvidenceDecision
from ayaka.evidence_policy import EvidencePolicy
from ayaka.primitives import DecisionResult, QuestionSpec


class Original:
    def __init__(self, p):
        self.p = p
        self.calls = []

    def decide(self, state, questions, device=None):
        self.calls.append(state)
        return [
            DecisionResult(q.type, list(self.p), dict(zip(q.candidates, self.p, strict=True)))
            for q in questions
        ]


def test_gold_bookkeeping_does_not_enter_extractor_messages():
    request = {
        "state": "Amount 120 EUR.",
        "question": {"instructions": "Within limit?"},
        "labels": ["no", "yes"],
        "expected": "SECRET_GOLD",
        "oracle_id_plan": "SECRET_PROGRAM",
        "family": "SECRET_FAMILY",
    }
    messages = str(id_messages(request))
    assert all(
        token not in messages for token in ("SECRET_GOLD", "SECRET_PROGRAM", "SECRET_FAMILY")
    )


def test_invalid_plan_preserves_exact_original_probabilities():
    base = Original([0.6, 0.4])
    wrapper = EvidenceDecision(
        base, lambda _: '{"e":[888],"c":{}}', EvidencePolicy(baseline_cutoff=1)
    )
    out = wrapper.decide(
        "Amount 120 EUR; cap 150 EUR.",
        [QuestionSpec("noul", "Does the policy permit it?", ["false", "true"])],
    )[0]
    assert out.probs == [0.6, 0.4]
    assert len(base.calls) == 1
    assert out.extras["evidence"]["error"]


def test_explicit_policy_is_required_and_confident_baseline_skips_generation():
    base = Original([0.97, 0.03])

    def forbidden(_):
        raise AssertionError("extractor should not be called")

    wrapper = EvidenceDecision(base, forbidden, EvidencePolicy())
    out = wrapper.decide(
        "Amount 120 EUR; cap 150 EUR.",
        [QuestionSpec("noul", "Does the policy permit it?", ["false", "true"])],
    )[0]
    assert out.probs == [0.97, 0.03]


def test_grounded_predicate_is_fused_and_score_metadata_is_recomputed():
    base = Original([0.6, 0.4])
    plan = '{"e":[0],"c":{"question_holds":"le(n0,n1)"}}'
    wrapper = EvidenceDecision(
        base, lambda _: plan, EvidencePolicy(boolean_confidence=0.8, baseline_cutoff=1)
    )
    out = wrapper.decide(
        "Amount 120 EUR; cap 150 EUR.",
        [QuestionSpec("noul", "Does the policy permit it?", ["false", "true"])],
    )[0]
    assert out.probs == pytest.approx([0.4, 0.6])
    assert "COMPLETE SOURCE" not in base.calls[1]  # existing augmented-state protocol
    assert "Amount 120 EUR" in base.calls[1]
    assert out.extras["p_true"] == pytest.approx(0.6)
    score = wrapper.decide(
        "Amount 120 EUR; cap 150 EUR.",
        [QuestionSpec("score", "Rate authority.", ["low", "high"], ordinals=[2, 8])],
    )[0]
    assert score.expected == pytest.approx(4.4)


def test_calculation_gate_skips_non_quantitative_questions():
    base = Original([0.6, 0.4])

    def forbidden(_):
        raise AssertionError("extractor should not be called")

    policy = EvidencePolicy(baseline_cutoff=1, gate="calculation")
    wrapper = EvidenceDecision(base, forbidden, policy)
    out = wrapper.decide(
        "Clause 3 exception unless overridden; amounts 120, 150 and 30 EUR.",
        [QuestionSpec("noul", "Was the manager courteous?", ["false", "true"])],
    )[0]
    assert out.probs == [0.6, 0.4] and out.extras["evidence"]["route"] == "baseline"
    with pytest.raises(ValueError):
        EvidencePolicy(gate="everything")


class Reading(Original):
    """Baseline 0.6/0.4; with worked steps in the state 0.1/0.9."""

    def decide(self, state, questions, device=None):
        self.calls.append(state)
        p = [0.1, 0.9] if "<worked_steps>" in str(state) else list(self.p)
        return [
            DecisionResult(q.type, p, dict(zip(q.candidates, p, strict=True))) for q in questions
        ]


QUANT = QuestionSpec("noul", "Is the total within the 225 budget?", ["over", "within"])
SOURCE = "Subtotal 240.00, discount 15 percent, tax 8 percent after discount."


def test_reasoned_route_reads_worked_steps_and_records_them():
    from ayaka.evidence_pipeline import FROZEN_REASONING_POLICY

    base = Reading([0.6, 0.4])
    seen = []

    def reasoner(messages):
        seen.append(messages)
        return "discounted 204.00; taxed 220.32 <= 225"

    out = EvidenceDecision(base, reasoner, FROZEN_REASONING_POLICY).decide(SOURCE, [QUANT])[0]
    assert out.probs == [0.1, 0.9] and out.extras["evidence"]["route"] == "reasoned"
    assert "220.32" in out.extras["evidence"]["worked_steps"]
    assert len(seen) == 1 and "within" in str(seen[0])  # option text, not gold
    assert "<worked_steps>" in base.calls[-1]


def test_reasoned_route_skips_confident_or_empty_cases():
    from ayaka.evidence_pipeline import FROZEN_REASONING_POLICY

    def forbidden(_):
        raise AssertionError("confident baselines must not generate")

    confident = EvidenceDecision(Reading([0.95, 0.05]), forbidden, FROZEN_REASONING_POLICY)
    assert confident.decide(SOURCE, [QUANT])[0].probs == [0.95, 0.05]
    empty = EvidenceDecision(Reading([0.6, 0.4]), lambda _: "  ", FROZEN_REASONING_POLICY)
    out = empty.decide(SOURCE, [QUANT])[0]
    assert out.probs == [0.6, 0.4] and out.extras["evidence"]["route"] == "baseline"


def test_reasoning_decision_requires_an_unmerged_adapter():
    import torch

    from ayaka.config import tiny_config
    from ayaka.evidence_pipeline import reasoning_decision
    from ayaka.model.decision import AyakaDecisionModel
    from ayaka.tokenization import ToyTokenizer

    merged = AyakaDecisionModel.from_config(tiny_config(), dtype=torch.float32)
    with pytest.raises(ValueError):
        reasoning_decision(merged, ToyTokenizer())
