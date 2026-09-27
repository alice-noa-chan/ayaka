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
