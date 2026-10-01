import copy
import json

import pytest
from test_reasoning_pipeline import Generator, Original

from ayaka.candidates import policy_for, split_parent, validate_proposals
from ayaka.reasoning_pipeline import ControlledDecision, Trace, TraceFailure
from ayaka.serve import BadRequest, DecisionService

ROWS = [
    {"id": "delivery", "description": "Delivery issue", "excludes": "Refund and other issues"},
    {"id": "address", "description": "Address change", "excludes": "Delivery and refund issues"},
]


class Proposer(Generator):
    def __init__(self, text=None, failure=False):
        super().__init__()
        self.text = json.dumps(ROWS) if text is None else text
        self.fail = failure
        self.messages = []

    def generate_trace(self, messages, budget, reserve=0):
        self.messages.append(messages)
        self.budgets.append(budget)
        t = Trace(text=self.text, token_ids=[5, 6, 7], prefill_tokens=11, finish_reason="eos")
        if self.fail:
            t.error = "decode failed"
            raise TraceFailure(t)
        return t


class Scorer(Original):
    def decide(self, state, questions, device=None):
        from ayaka.primitives import DecisionResult

        self.calls.append(copy.deepcopy(questions))
        return [
            DecisionResult(q.type, [0.7, 0.3] if len(q.candidates) == 2 else [0.7, 0.2, 0.1], {})
            for q in questions
        ]


def request(mode="expand"):
    q = {
        "type": "choice",
        "instructions": "Classify the primary request.",
        "reasoning": {"mode": "auto"},
        "candidate_generation": {
            "mode": mode,
            "experimental": True,
            "scope": "The primary customer request",
        },
    }
    if mode == "expand":
        q["criteria"] = {"refund": "Refund request", "other": "Other primary request"}
        q["candidate_generation"]["other_id"] = "other"
    return {"state": "Where is my package?", "questions": {"intent": q}}


def service(proposer=None):
    original = Scorer()
    return DecisionService(ControlledDecision(original, proposer or Proposer()), "test"), original


def test_expand_preserves_siblings_and_mass_and_uses_clean_scoring():
    server, original = service()
    body = request()
    before = copy.deepcopy(body)
    result = server.handle(body)
    assert body == before
    assert result["answers"]["intent"]["probabilities"] == pytest.approx(
        {"refund": 0.7, "delivery": 0.21, "address": 0.06, "other": 0.03}
    )
    assert len(original.calls) == 2  # Neither scoring branch uses proposal cache.
    assert "Exclude all original sibling outcomes" in original.calls[1][0].instruction
    assert "Refund request" in original.calls[1][0].instruction
    assert original.calls[0][0].candidates == list(body["questions"]["intent"]["criteria"].values())
    assert result["usage"]["candidate_tokens"] == 3
    assert result["usage"]["reasoning_tokens"] == 0
    assert result["usage"]["output_tokens"] == 3
    assert result["candidate_generation"]["intent"]["calibration"] == "unvalidated"


def test_open_is_explicit_experimental_and_retains_residual():
    server, _ = service()
    result = server.handle(request("open"))
    assert set(result["answers"]["intent"]["probabilities"]) == {"delivery", "address", "__other__"}
    assert len(result["candidate_generation"]["intent"]["version"]) == 64


@pytest.mark.parametrize("mode", ["expand", "open"])
@pytest.mark.parametrize("settings", [{"mode": "off"}, {"mode": "on", "max_tokens": 0}])
def test_generation_disabled_never_proposes_or_scores(mode, settings):
    proposer = Proposer()
    server, original = service(proposer)
    body = request(mode)
    body["questions"]["intent"]["reasoning"] = settings
    with pytest.raises(BadRequest, match="disabled"):
        server.handle(body)
    assert not proposer.budgets and not original.calls


@pytest.mark.parametrize("failure", [False, True])
def test_failed_proposals_count_tokens_and_retain_parent(failure):
    server, _ = service(Proposer("not json", failure))
    result = server.handle(request())
    assert result["answers"]["intent"]["probabilities"] == {"refund": 0.7, "other": 0.3}
    assert result["candidate_generation"]["intent"]["status"] == "failed"
    assert result["usage"]["candidate_tokens"] == 3


def test_schema_duplicate_and_wrong_primitive_validation():
    policy = policy_for(request()["questions"]["intent"])
    rows = copy.deepcopy(ROWS)
    rows[1]["description"] = "  DELIVERY issue  "
    with pytest.raises(ValueError, match="duplicate"):
        validate_proposals(json.dumps(rows), policy, {})
    q = request()["questions"]["intent"]
    q["type"] = "score"
    with pytest.raises(ValueError, match="Choice only"):
        policy_for(q)
    with pytest.raises(ValueError):
        split_parent({"other": 0.3}, {"x": float("nan")}, "other")


def test_fixed_unchanged_and_settings_do_not_leak():
    server, _ = service()
    server.handle(request())
    body = request()
    q = body["questions"]["intent"]
    q["candidate_generation"] = {"mode": "fixed"}
    q["reasoning"] = {"mode": "off"}
    result = server.handle(body)
    assert result["answers"]["intent"]["probabilities"] == {"refund": 0.7, "other": 0.3}
    assert result["usage"]["candidate_tokens"] == 0
    second = copy.deepcopy(q)
    body["questions"]["second"] = second
    original = server.decision.original
    original.calls.clear()
    batched = server.handle(body)
    assert len(original.calls) == 1 and len(original.calls[0]) == 2
    assert batched["answers"]["intent"] == batched["answers"]["second"]


def test_generated_partition_cannot_use_fixed_calibration_or_router():
    class Forbidden:
        def apply(self, *args, **kwargs):
            pytest.fail("fixed calibration on generated candidates")

        def should_reason(self, *args, **kwargs):
            pytest.fail("fixed router on generated candidates")

    server, _ = service()
    server.decision.calibration = Forbidden()
    server.decision.router = Forbidden()
    result = server.handle(request("open"))
    assert result["candidate_generation"]["intent"]["status"] == "completed"
    assert server.decision.calibration is not None and server.decision.router is not None
    assert getattr(server.decision.original, "apply_temperature", True) is True


def test_unvalidated_partition_bypasses_native_text_temperature():
    import torch
    from test_reasoning_pipeline import Tok

    from ayaka.config import tiny_config
    from ayaka.model.electra import ElectraDecisionModel
    from ayaka.primitives import QuestionSpec
    from ayaka.reasoning import ReasoningSettings
    from ayaka.reasoning_pipeline import controlled_decision

    model = ElectraDecisionModel.from_config(tiny_config(version=2), dtype=torch.float32).eval()
    ordinary = controlled_decision(model, Tok())
    experimental = ordinary.for_unvalidated_partition()
    spec = [QuestionSpec("noul", "A request?", ["no", "yes"])]
    settings = [ReasoningSettings(mode="off")]
    before = experimental.decide("Hello", spec, reasoning=settings)[0].probs
    with torch.no_grad():
        model.temperature.fill_(7)
    after = experimental.decide("Hello", spec, reasoning=settings)[0].probs
    assert before == pytest.approx(after)
    assert ordinary.original.apply_temperature is True
    assert ordinary.generator.apply_temperature is True


def test_forced_high_preserved_separately_from_proposal_budget():
    class ForcedProposer(Proposer):
        def readout(self, trace, spec):
            return [0.7, 0.3] if len(spec.candidates) == 2 else [0.7, 0.2, 0.1]

    proposer = ForcedProposer()
    server, original = service(proposer)
    body = request()
    body["questions"]["intent"].pop("reasoning")
    body["options"] = {"reasoning": {"mode": "on", "effort": "high"}}
    result = server.handle(body)
    assert proposer.budgets == [1024, 384, 1024]
    assert not original.calls
    assert result["usage"]["reasoning_tokens"] == 6
    assert result["usage"]["candidate_tokens"] == 3
    assert result["usage"]["output_tokens"] == 9
    judgments = result["candidate_generation"]["intent"]["judgments"]
    assert [j["reasoning"]["budget"] for j in judgments] == [1024, 1024]
