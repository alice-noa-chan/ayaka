"""CPU-only candidate proposals and canonical readout; no model loading."""

import copy
import io
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import replace

import pytest

from ayaka.swift.candidates import (
    CandidateGenerationError,
    ProposalResult,
    VLLMCandidateGenerator,
    split_parent,
)
from ayaka.swift.policy import Policy
from ayaka.swift.prompt import InvalidQuestion, render_question
from ayaka.swift.readers import FakeReader, VLLMChatReader
from ayaka.swift.router import FEATURES
from ayaka.swift.server import DecisionService, serve


def rows(count=2):
    return [
        {"id": f"child-{i}", "description": f"Outcome {i}", "excludes": f"All outcomes except {i}"}
        for i in range(count)
    ]


class FakeGenerator:
    def __init__(self, text=None, *, error=None):
        self.text = json.dumps(rows()) if text is None else text
        self.error = error
        self.calls = []

    def generate(self, messages, max_tokens):
        self.calls.append((messages, max_tokens))
        if self.error:
            raise self.error
        return ProposalResult(self.text, 17, min(23, max_tokens), "eos")


def request(mode="open", *, namespace=True, **settings):
    policy = {"mode": mode, "experimental": True, "scope": "One primary request", **settings}
    question = {"type": "choice", "instructions": "Classify the primary request"}
    if mode == "expand":
        question["criteria"] = {"refund": "Refund request", "other": "Any request besides refund"}
        policy.setdefault("other_id", "other")
    body = {"state": {"message": "Where is my package?"}, "questions": {"q": question}}
    if namespace:
        body["ayaka"] = {"questions": {"q": {"candidate_generation": policy}}}
    else:
        question["candidate_generation"] = policy
    return body


@pytest.fixture
def services():
    active = []

    def make(reader=None, generator=None, policy=None, **kwargs):
        reader = reader or FakeReader()
        generator = generator or FakeGenerator()
        service = DecisionService(reader, "fake", policy, candidate_generator=generator, **kwargs)
        active.append(service)
        return service, reader, generator

    yield make
    for service in active:
        service.close()


@pytest.mark.parametrize("namespace", [True, False])
def test_fixed_response_bytes_and_calls_unchanged(services, namespace):
    service, reader, generator = services()
    body = {
        "state": "original state",
        "questions": {
            "q": {"type": "choice", "criteria": {"a": "First", "b": "Second"}},
            "n": {"type": "noul"},
            "s": {"type": "score", "criteria": ["Low", "High"]},
        },
    }
    baseline = json.dumps(service.handle(body), ensure_ascii=False).encode()
    original_calls = copy.deepcopy(reader.calls)
    fixed = copy.deepcopy(body)
    if namespace:
        fixed["ayaka"] = {
            "questions": {
                name: {"candidate_generation": {"mode": "fixed"}} for name in body["questions"]
            }
        }
    else:
        for question in fixed["questions"].values():
            question["candidate_generation"] = {"mode": "fixed"}
    assert json.dumps(service.handle(fixed), ensure_ascii=False).encode() == baseline
    assert len(reader.calls) == 2 * len(body["questions"])
    assert reader.calls[len(original_calls) :] == original_calls
    assert generator.calls == []


@pytest.mark.parametrize("namespace", [True, False])
def test_open_freezes_list_scores_original_state_and_accounts_usage(services, namespace):
    service, reader, generator = services(FakeReader([{"A": 0.7, "B": 0.2, "C": 0.1}]))
    body = request(namespace=namespace, max_tokens=64)
    before = copy.deepcopy(body)
    result = service.handle(body)
    answer = result["answers"]["q"]
    assert set(answer) == {"type", "choice", "probabilities", "confidence", "ayaka"}
    assert answer["choice"] == "child-0"
    assert answer["probabilities"] == pytest.approx(
        {"child-0": 0.7, "child-1": 0.2, "__other__": 0.1}
    )
    assert sum(answer["probabilities"].values()) == pytest.approx(1)
    candidates, diagnostics = answer["ayaka"]["candidates"], answer["ayaka"]["diagnostics"]
    assert [row["id"] for row in candidates["items"]] == list(answer["probabilities"])
    assert candidates["items"][-1]["id"] == "__other__"
    assert candidates["mode"] == "open" and candidates["parent_id"] is None
    assert candidates["status"] == "completed" and len(candidates["hash"]) == 64
    assert diagnostics["calibration"] == "unvalidated_generated_partition"
    assert diagnostics["validation_outcome"] == "accepted"
    assert [stage["stage"] for stage in diagnostics["stages"]] == [
        "proposal",
        "generated_partition",
    ]
    assert len(generator.calls) == len(reader.calls) == 1
    assert generator.calls[0][1] == 64
    child = {
        "type": "choice",
        "instructions": body["questions"]["q"]["instructions"]
        + "\nRestrict this classification to: One primary request",
        "criteria": {row["id"]: row["description"] for row in candidates["items"]},
    }
    expected_messages, _ = render_question(body["state"], child)
    assert reader.calls[0][0] == expected_messages
    assert "Return ONLY" not in json.dumps(reader.calls)
    assert result["usage"] == {"input_tokens": 27, "output_tokens": 24}
    assert result["ayaka"]["usage"] == {
        "proposal_input_tokens": 17,
        "proposal_output_tokens": 23,
        "scoring_input_tokens": 10,
        "scoring_output_tokens": 1,
    }
    assert body == before


def test_expand_keeps_siblings_exactly_and_conserves_parent_mass(services):
    parent = {"A": 0.7, "B": 0.3}
    body = request("expand")
    fixed_service, _, _ = services(FakeReader([parent]))
    baseline = fixed_service.handle({"state": body["state"], "questions": body["questions"]})[
        "answers"
    ]["q"]
    service, reader, generator = services(FakeReader([parent, {"A": 0.7, "B": 0.2, "C": 0.1}]))
    result = service.handle(body)
    answer = result["answers"]["q"]
    probabilities = answer["probabilities"]
    assert probabilities["refund"] == baseline["probabilities"]["refund"]
    assert probabilities == pytest.approx(
        {"refund": 0.7, "child-0": 0.21, "child-1": 0.06, "other": 0.03}
    )
    assert sum(probabilities.values()) == pytest.approx(1)
    candidates = answer["ayaka"]["candidates"]
    diagnostics = answer["ayaka"]["diagnostics"]
    assert candidates["parent_id"] == candidates["residual_id"] == "other"
    assert candidates["items"][-1]["id"] == "other"
    assert candidates["items"][0] == {"id": "refund", "description": "Refund request"}
    assert diagnostics["parent_probabilities"] == baseline["probabilities"]
    assert diagnostics["parent_candidates"] == body["questions"]["q"]["criteria"]
    assert [stage["stage"] for stage in diagnostics["stages"]] == [
        "original",
        "proposal",
        "generated_partition",
    ]
    prompt = reader.calls[1][0][-1]["content"]
    assert "Condition on this parent outcome: Any request besides refund" in prompt
    assert '"refund": "Refund request"' in prompt
    assert len(generator.calls) == 1 and len(reader.calls) == 2
    assert result["usage"] == {"input_tokens": 37, "output_tokens": 25}


def test_split_bookkeeping_example_and_invalid_mass():
    probabilities = split_parent(
        {"sibling": 0.7, "other": 0.3}, {"a": 0.7, "b": 0.2, "other": 0.1}, "other"
    )
    assert probabilities == {"sibling": 0.7, "a": 0.21, "b": 0.06, "other": 0.03}
    for children in ({"a": 0.7}, {"sibling": 1}, {"a": float("nan")}, {"a": -0.1, "other": 1.1}):
        with pytest.raises(ValueError):
            split_parent({"sibling": 0.7, "other": 0.3}, children, "other")


def malformed_proposals():
    valid = rows()
    return [
        "not JSON",
        "```json\n[]\n```",
        "{}",
        "[]",
        json.dumps(valid[:1]),
        json.dumps([*valid, *rows(3)]),
        json.dumps([valid[0], valid[0]]),
        json.dumps([valid[0], {**valid[1], "id": " CHILD-0 "}]),
        json.dumps([valid[0], {**valid[1], "id": "ｃｈｉｌｄ-０"}]),
        json.dumps([valid[0], {**valid[1], "description": " OUTCOME   0 "}]),
        json.dumps([valid[0], {**valid[1], "description": "Ｏｕｔｃｏｍｅ ０"}]),
        json.dumps([valid[0], {**valid[1], "id": "__other__"}]),
        json.dumps([valid[0], {**valid[1], "id": " __OTHER__ "}]),
        json.dumps([valid[0], {**valid[1], "id": "x" * 81}]),
        json.dumps([valid[0], {**valid[1], "description": "x" * 2001}]),
        json.dumps([valid[0], {**valid[1], "excludes": "x" * 2001}]),
        json.dumps([valid[0], {**valid[1], "excludes": " "}]),
        json.dumps([valid[0], {**valid[1], "description": 42}]),
        json.dumps([valid[0], {"id": "new", "description": "Missing exclusion"}]),
        json.dumps([valid[0], {**valid[1], "unexpected": "extra"}]),
        '[{"id":"first","id":"second","description":"One","excludes":"Two"},'
        '{"id":"third","description":"Three","excludes":"Four"}]',
        json.dumps([valid[0], {**valid[1], "description": "\ud800"}]),
    ]


@pytest.mark.parametrize("text", malformed_proposals())
def test_invalid_open_proposals_fail_once_with_usage(services, text):
    service, reader, generator = services(generator=FakeGenerator(text))
    with pytest.raises(
        CandidateGenerationError, match="open candidate generation failed"
    ) as caught:
        service.handle(request())
    assert len(generator.calls) == 1 and reader.calls == []
    response = caught.value.response
    assert response["usage"] == {"input_tokens": 17, "output_tokens": 23}
    diagnostics = response["ayaka"]["questions"]["q"]["diagnostics"]
    assert diagnostics["validation_outcome"] == "rejected"
    assert diagnostics["stages"][0]["status"] == "failed"


@pytest.mark.parametrize(
    "text", [value for value in malformed_proposals() if value != json.dumps(rows()[:1])]
)
def test_invalid_expand_proposals_return_original_distribution(services, text):
    service, reader, generator = services(generator=FakeGenerator(text))
    body = request("expand")
    baseline_service, _, _ = services()
    baseline = baseline_service.handle({"questions": body["questions"]})["answers"]["q"]
    result = service.handle(body)
    answer = result["answers"]["q"]
    assert {k: v for k, v in answer.items() if k != "ayaka"} == {
        k: v for k, v in baseline.items() if k != "ayaka"
    }
    assert answer["ayaka"]["candidates"]["status"] == "expansion_failed"
    assert len(reader.calls) == len(generator.calls) == 1
    assert result["usage"] == {"input_tokens": 27, "output_tokens": 24}


@pytest.mark.parametrize(
    "proposal",
    [
        {"id": "refund", "description": "New", "excludes": "Other"},
        {"id": " Refund ", "description": "New", "excludes": "Other"},
        {"id": "other", "description": "New", "excludes": "Other"},
        {"id": "new", "description": " REFUND  REQUEST ", "excludes": "Other"},
        {"id": "new", "description": "Any request besides refund", "excludes": "Other"},
    ],
)
def test_replacing_original_expansion_falls_back_explicitly(services, proposal):
    reader = FakeReader([{"A": 0.7, "B": 0.3}, {"A": 0.7, "B": 0.3}])
    service, _, generator = services(reader, FakeGenerator(json.dumps([proposal])))
    body = request("expand")
    baseline = service.handle({"state": body["state"], "questions": body["questions"]})
    result = service.handle(body)
    answer = result["answers"]["q"]
    assert {k: v for k, v in answer.items() if k != "ayaka"} == {
        k: v for k, v in baseline["answers"]["q"].items() if k != "ayaka"
    }
    assert answer["ayaka"]["candidates"]["status"] == "expansion_failed"
    assert answer["ayaka"]["candidates"]["items"] == [
        {"id": k, "description": v} for k, v in body["questions"]["q"]["criteria"].items()
    ]
    assert answer["ayaka"]["diagnostics"]["validation_outcome"] == "rejected"
    assert len(reader.calls) == 2 and len(generator.calls) == 1
    assert result["usage"] == {"input_tokens": 27, "output_tokens": 24}


@pytest.mark.parametrize("mode", ["open", "expand"])
def test_backend_generation_failure_is_explicit(services, mode):
    service, reader, generator = services(
        generator=FakeGenerator(error=RuntimeError("generator unavailable"))
    )
    if mode == "open":
        with pytest.raises(CandidateGenerationError, match="generator unavailable") as caught:
            service.handle(request(mode))
        assert caught.value.response["usage"] == {"input_tokens": 0, "output_tokens": 0}
        assert reader.calls == []
    else:
        result = service.handle(request(mode))
        answer = result["answers"]["q"]
        assert answer["ayaka"]["candidates"]["status"] == "expansion_failed"
        assert answer["ayaka"]["diagnostics"]["validation_outcome"] == "generation_failed"
        assert len(reader.calls) == 1
    assert len(generator.calls) == 1


@pytest.mark.parametrize(
    "patch",
    [
        {"experimental": False},
        {"scope": ""},
        {"scope": " "},
        {"scope": "x" * 4001},
        {"scope": 12},
        {"max_new": 0},
        {"max_new": 9},
        {"max_new": True},
        {"max_new": 1},
        {"max_tokens": 0},
        {"max_tokens": 1025},
        {"max_tokens": True},
        {"max_tokens": 1.5},
        {"mode": "unknown"},
        {"mode": []},
        {"unexpected": True},
        {"other_id": "other"},
    ],
)
def test_generation_policy_validation_before_all_calls(services, patch):
    service, reader, generator = services()
    body = request(**patch)
    body["questions"]["fixed"] = {"type": "noul"}
    with pytest.raises(InvalidQuestion):
        service.handle(body)
    assert reader.calls == generator.calls == []


@pytest.mark.parametrize("kind", ["noul", "score"])
@pytest.mark.parametrize("mode", ["open", "expand"])
def test_generation_rejects_other_primitives(services, kind, mode):
    service, reader, generator = services()
    body = request(mode)
    body["questions"]["q"]["type"] = kind
    with pytest.raises(InvalidQuestion, match="Choice only"):
        service.handle(body)
    assert reader.calls == generator.calls == []


@pytest.mark.parametrize("placement", ["options", "question"])
def test_explicit_reasoning_still_rejected_before_calls(services, placement):
    service, reader, generator = services()
    body = request()
    layer = body.setdefault("options", {}) if placement == "options" else body["questions"]["q"]
    layer["reasoning"] = {"mode": "on", "max_tokens": 32}
    with pytest.raises(InvalidQuestion, match="reasoning not supported"):
        service.handle(body)
    assert reader.calls == generator.calls == []


@pytest.mark.parametrize("settings", [{"mode": "off"}, {"mode": "on", "max_tokens": 0}])
def test_proposal_budget_independent_of_reasoning(services, settings):
    service, _, generator = services()
    body = request(max_tokens=1)
    body["options"] = {"reasoning": settings}
    result = service.handle(body)
    assert result["answers"]["q"]["ayaka"]["candidates"]["status"] == "completed"
    assert generator.calls[0][1] == 1
    assert result["ayaka"]["usage"]["proposal_output_tokens"] == 1


@pytest.mark.parametrize("mode,trace_count", [("off", 0), ("on", 1)])
def test_mixed_generated_and_fixed_questions_honor_each_reasoning_override(
    services, mode, trace_count
):
    from test_swift_reasoning import TraceReader, always_router

    reader = TraceReader()
    service, _, generator = services(
        reader, policy=Policy(reasoning_route=always_router()), diagnostic=True
    )
    body = request()
    body["options"] = {"reasoning": {"mode": mode, "effort": "high"}}
    body["questions"]["q"]["reasoning"] = {"mode": "off"}
    body["questions"]["fixed"] = {"type": "noul"}
    result = service.handle(body)
    assert result["answers"]["fixed"]["ayaka"]["route"] == ("reasoned" if trace_count else "direct")
    assert len(reader.trace_calls) == trace_count
    if trace_count:
        assert reader.trace_calls[0][1] == 1024
    assert len(reader.calls) == 2 and len(generator.calls) == 1
    assert service.policy.reasoning_route == always_router()


def test_alias_agreement_and_conflict_are_atomic(services):
    service, reader, generator = services()
    body = request()
    policy = body["ayaka"]["questions"]["q"]["candidate_generation"]
    body["questions"]["q"]["candidate_generation"] = {**policy, "max_new": 8}
    with pytest.raises(InvalidQuestion, match="conflicting"):
        service.handle(body)
    assert reader.calls == generator.calls == []
    body["questions"]["q"]["candidate_generation"] = policy.copy()
    assert service.handle(body)["answers"]["q"]["ayaka"]["candidates"]["status"] == "completed"
    assert len(generator.calls) == len(reader.calls) == 1


def test_alias_conflict_distinguishes_boolean_and_integer(services):
    service, reader, generator = services()
    body = request()
    policy = body["ayaka"]["questions"]["q"]["candidate_generation"]
    body["questions"]["q"]["candidate_generation"] = {**policy, "experimental": 1}
    with pytest.raises(InvalidQuestion, match="conflicting"):
        service.handle(body)
    assert reader.calls == generator.calls == []


def test_limits_accept_boundaries_and_one_expansion_proposal(services):
    service, reader, generator = services(generator=FakeGenerator(json.dumps(rows(1))))
    result = service.handle(request("expand", max_new=1, max_tokens=1024, scope="s" * 4000))
    assert len(result["answers"]["q"]["probabilities"]) == 3
    assert generator.calls[0][1] == 1024 and len(reader.calls) == 2


@pytest.mark.parametrize("policy", [None, [], {}, {"mode": "fixed", "max_tokens": 1}])
def test_policy_must_be_object_and_fixed_accepts_only_mode(services, policy):
    service, reader, generator = services()
    body = request()
    body["ayaka"]["questions"]["q"]["candidate_generation"] = policy
    with pytest.raises(InvalidQuestion):
        service.handle(body)
    assert reader.calls == generator.calls == []


def test_generated_media_rejected_before_all_calls(services):
    service, reader, generator = services()
    body = request()
    body["media"] = []
    with pytest.raises(InvalidQuestion, match="text states only"):
        service.handle(body)
    assert reader.calls == generator.calls == []


@pytest.mark.parametrize(
    "extension",
    [None, [], {"questions": []}, {"questions": {"missing": {}}}, {"questions": {"q": []}}],
)
def test_namespace_shape_rejected(services, extension):
    service, reader, generator = services()
    body = request()
    body["ayaka"] = extension
    with pytest.raises(InvalidQuestion):
        service.handle(body)
    assert reader.calls == generator.calls == []


@pytest.mark.parametrize(
    "question_patch",
    [{"instructions": ""}, {"instructions": None}, {"criteria": {}}, {"criteria": ["a", "b"]}],
)
def test_open_requires_instructions_and_absent_criteria(services, question_patch):
    service, reader, generator = services()
    body = request()
    body["questions"]["q"].update(question_patch)
    with pytest.raises(InvalidQuestion):
        service.handle(body)
    assert reader.calls == generator.calls == []


@pytest.mark.parametrize(
    "criteria,other_id",
    [
        (None, "other"),
        (["a", "other"], "other"),
        ({"a": "A", "other": "B"}, "missing"),
        ({"a": "A", "other": ""}, "other"),
        ({"a": "A", "other": {}}, "other"),
        ({"a": "A", "other": "B"}, []),
    ],
)
def test_expand_requires_explicit_original_definitions(services, criteria, other_id):
    service, reader, generator = services()
    body = request("expand", other_id=other_id)
    body["questions"]["q"]["criteria"] = criteria
    with pytest.raises(InvalidQuestion):
        service.handle(body)
    assert reader.calls == generator.calls == []


def test_generated_bias_and_reasoning_route_disabled_without_policy_mutation(services):
    route = {
        "features": FEATURES.copy(),
        "means": [0.0] * len(FEATURES),
        "scales": [1.0] * len(FEATURES),
        "weights": [0.0] * len(FEATURES),
        "intercept": 0.0,
        "threshold": 0.0,
        "rate_cap": 0.1,
        "max_tokens": 384,
    }
    policy = Policy(t_choice=2, letter_bias={"choice": {"3": [10, -5, -5]}}, reasoning_route=route)
    service, reader, _ = services(
        FakeReader([{"A": 0.64, "B": 0.25, "C": 0.11}]), policy=policy, diagnostic=True
    )
    original_policy = copy.deepcopy(service.policy)
    answer = service.handle(request())["answers"]["q"]
    expected = Policy(t_choice=2).decide(
        "choice", {"child-0": 0.64, "child-1": 0.25, "__other__": 0.11}
    )
    assert answer["probabilities"] == pytest.approx(expected["probabilities"])
    assert service.policy == original_policy
    assert len(reader.calls) == 1


def test_expand_original_uses_fitted_bias_but_children_use_temperature_only(services):
    policy = Policy(t_choice=2, letter_bias={"choice": {"2": [2, -2], "3": [10, -5, -5]}})
    body = request("expand")
    fixed, _, _ = services(FakeReader([{"A": 0.7, "B": 0.3}]), policy=policy, diagnostic=True)
    baseline = fixed.handle({"questions": body["questions"]})["answers"]["q"]
    service, _, _ = services(
        FakeReader([{"A": 0.7, "B": 0.3}, {"A": 0.64, "B": 0.25, "C": 0.11}]),
        policy=policy,
        diagnostic=True,
    )
    answer = service.handle(body)["answers"]["q"]
    assert answer["probabilities"]["refund"] == baseline["probabilities"]["refund"]
    conditional = answer["ayaka"]["diagnostics"]["conditional_probabilities"]
    assert conditional == pytest.approx(
        Policy(t_choice=2).apply("choice", {"child-0": 0.64, "child-1": 0.25, "other": 0.11})
    )


def test_large_original_partition_groups_and_generated_children_stay_conditional(services):
    body = request("expand", max_new=8)
    original = {f"original-{i}": f"Original outcome {i}" for i in range(29)}
    original["other"] = "Remaining original outcomes"
    body["questions"]["q"]["criteria"] = original
    fixed, _, _ = services()
    baseline = fixed.handle({"questions": body["questions"]})["answers"]["q"]["probabilities"]
    service, reader, generator = services(generator=FakeGenerator(json.dumps(rows(8))))
    answer = service.handle(body)["answers"]["q"]
    probabilities = answer["probabilities"]
    assert len(probabilities) == 38 and sum(probabilities.values()) == pytest.approx(1)
    assert {k: probabilities[k] for k in original if k != "other"} == {
        k: v for k, v in baseline.items() if k != "other"
    }
    stages = answer["ayaka"]["diagnostics"]["stages"]
    assert stages[0]["readout"] == "grouped_approx" and stages[0]["passes"] == 3
    assert stages[-1]["passes"] == 1
    assert len(reader.calls) == 4 and len(generator.calls) == 1
    assert all(len(letters) <= 26 for _, letters in reader.calls)


def test_mixed_fixed_and_generated_questions_usage_and_namespaces(services):
    service, reader, generator = services()
    body = request()
    body["questions"]["n"] = {"type": "noul"}
    result = service.handle(body)
    assert result["answers"]["n"] == {
        "type": "noul",
        "noul": 0.5,
        "ayaka": {"route": "direct", "calibration": "unfitted"},
    }
    assert "ayaka" in result["answers"]["q"]
    assert len(reader.calls) == 2 and len(generator.calls) == 1
    assert result["usage"] == {"input_tokens": 37, "output_tokens": 25}


def test_list_hash_binds_order_scope_and_parent_definitions(services):
    def get_hash(text, body):
        service, _, _ = services(generator=FakeGenerator(text))
        return service.handle(body)["answers"]["q"]["ayaka"]["candidates"]["hash"]

    body = request("expand")
    base = get_hash(json.dumps(rows()), body)
    assert get_hash(json.dumps(rows()), body) == base
    assert get_hash(json.dumps(list(reversed(rows()))), body) != base
    assert get_hash(json.dumps(rows()), request("expand", scope="Another scope")) != base
    changed = copy.deepcopy(body)
    changed["questions"]["q"]["criteria"]["other"] = "Changed parent definition"
    assert get_hash(json.dumps(rows()), changed) != base


def test_vllm_proposals_use_same_model_and_separate_greedy_budget(monkeypatch):
    reader = VLLMChatReader(
        "http://fake/v1", "frozen-model", 7, chat_template_kwargs={"enable_thinking": True}
    )
    seen = []

    def urlopen(request, timeout):
        seen.append((request.full_url, json.loads(request.data), timeout))
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {"message": {"content": json.dumps(rows())}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 17, "completion_tokens": 23},
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    messages = [{"role": "user", "content": "Propose"}]
    result = VLLMCandidateGenerator(reader).generate(messages, 64)
    assert result == ProposalResult(json.dumps(rows()), 17, 23, "eos")
    assert seen == [
        (
            reader.url,
            {
                "model": reader.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 64,
                "ignore_eos": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            7,
        )
    ]
    assert reader.chat_template_kwargs["enable_thinking"] is True


def test_vllm_service_default_generator_then_original_state_raw_read(monkeypatch):
    reader = VLLMChatReader("http://fake/v1", "frozen-model")
    reader.describe = FakeReader().describe
    seen = []

    def urlopen(wire_request, timeout):
        body = json.loads(wire_request.data)
        seen.append(body)
        if body["max_tokens"] == 64:
            payload = {
                "choices": [{"message": {"content": json.dumps(rows())}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 17, "completion_tokens": 23},
            }
        else:
            payload = {
                "prompt_token_ids": reader.describe(body["messages"], ["A", "B", "C"])[
                    "input_token_ids"
                ],
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "token": "token_id:65",
                                    "top_logprobs": [
                                        {"token": "token_id:65", "logprob": 2},
                                        {"token": "token_id:66", "logprob": 1},
                                        {"token": "token_id:67", "logprob": 0},
                                    ],
                                }
                            ]
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    service = DecisionService(reader, "fake")
    try:
        result = service.handle(request(max_tokens=64))
    finally:
        service.close()
    assert [body["max_tokens"] for body in seen] == [64, 1]
    assert all(body["model"] == "frozen-model" for body in seen)
    assert all(
        body["temperature"] == 0 and body["chat_template_kwargs"]["enable_thinking"] is False
        for body in seen
    )
    assert seen[1]["logprob_token_ids"] == [65, 66, 67]
    assert [message["role"] for message in seen[1]["messages"]] == ["system", "user"]
    assert "Where is my package?" in seen[1]["messages"][-1]["content"]
    assert "Return ONLY" not in json.dumps(seen[1])
    assert result["usage"] == {"input_tokens": 18, "output_tokens": 24}


@pytest.mark.parametrize(
    "choice",
    [None, [], [None], [{"message": None}], [{"message": {"content": "valid but missing finish"}}]],
)
def test_malformed_vllm_content_still_accounts_reported_tokens(services, monkeypatch, choice):
    reader = VLLMChatReader("http://fake", "frozen-model")
    generator = VLLMCandidateGenerator(reader)
    payload = {"usage": {"prompt_tokens": 17, "completion_tokens": 23}, "choices": choice}
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode())
    )
    service, fake_reader, _ = services(generator=generator)
    with pytest.raises(CandidateGenerationError) as caught:
        service.handle(request())
    assert caught.value.response["usage"] == {"input_tokens": 17, "output_tokens": 23}
    assert fake_reader.calls == []


def test_backend_budget_violation_reports_actual_spend(services):
    generator = FakeGenerator()
    generator.generate = lambda messages, max_tokens: ProposalResult(
        json.dumps(rows()), 17, 385, "length"
    )
    service, reader, _ = services(generator=generator)
    with pytest.raises(
        CandidateGenerationError, match="exceeds the requested token budget"
    ) as caught:
        service.handle(request())
    assert caught.value.response["usage"] == {"input_tokens": 17, "output_tokens": 385}
    assert reader.calls == []


@pytest.mark.parametrize("valid", [False, True])
def test_length_capped_proposal_has_one_attempt_and_actual_usage(services, valid):
    generator = FakeGenerator()
    generator.generate = lambda messages, max_tokens: ProposalResult(
        json.dumps(rows()) if valid else "[truncated", 17, max_tokens, "length"
    )
    service, reader, _ = services(generator=generator)
    if valid:
        result = service.handle(request())
        assert result["answers"]["q"]["ayaka"]["diagnostics"]["finish_reason"] == "length"
        assert len(reader.calls) == 1
    else:
        with pytest.raises(CandidateGenerationError) as caught:
            service.handle(request())
        result = caught.value.response
        assert reader.calls == []
    assert result["ayaka"]["usage"]["proposal_output_tokens"] == 384


def test_expand_confidence_uses_final_joint_distribution(services):
    service, _, _ = services(
        FakeReader([{"A": 0.5, "B": 0.5}, {"A": 0.6, "B": 0.4}]), FakeGenerator(json.dumps(rows(1)))
    )
    answer = service.handle(request("expand"))["answers"]["q"]
    assert answer["probabilities"] == pytest.approx({"refund": 0.5, "child-0": 0.3, "other": 0.2})
    assert answer["confidence"] == pytest.approx(0.25)
    assert "confidence" not in answer["ayaka"]


def test_production_backend_guard_does_not_affect_fixed_readout():
    reader = FakeReader()
    service = DecisionService(reader, "fake")
    try:
        with pytest.raises(InvalidQuestion, match="frozen vLLM"):
            service.handle(request())
        assert reader.calls == []
        assert service.handle({"questions": {"n": {"type": "noul"}}})["answers"]["n"]["noul"] == 0.5
    finally:
        service.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("output_tokens", 385),
        ("output_tokens", -1),
        ("input_tokens", True),
        ("text", None),
        ("finish_reason", "tool_calls"),
    ],
)
def test_invalid_generator_results_are_not_successes(services, field, value):
    generator = FakeGenerator()
    generator.generate = lambda messages, max_tokens: replace(
        ProposalResult(json.dumps(rows()), 17, 23, "eos"), **{field: value}
    )
    service, reader, _ = services(generator=generator)
    with pytest.raises(CandidateGenerationError):
        service.handle(request())
    assert reader.calls == []


def test_http_422_failure_usage_and_reasoning_then_open_success(services):
    service, reader, generator = services(generator=FakeGenerator("malformed"))
    server = serve(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post(body):
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/systemone",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            return response.status, json.load(response)

    try:
        status, response = post(request())
        assert status == 422 and "open candidate generation failed" in response["error"]
        assert response["usage"] == {"input_tokens": 17, "output_tokens": 23}
        body = request()
        body["options"] = {"reasoning": {"mode": "on"}}
        assert post(body)[0] == 422
        assert len(generator.calls) == 1 and reader.calls == []
        generator.text = json.dumps(rows())
        status, response = post(request())
        assert status == 200 and "ayaka" in response["answers"]["q"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_candidate_imports_block_model_stack():
    code = """
import importlib.abc
import sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers'} or fullname in {'ayaka.candidates', 'ayaka.reasoning_pipeline'}:
            raise AssertionError('unexpected model dependency: ' + fullname)
sys.meta_path.insert(0, BlockModels())
import ayaka.swift.server
import ayaka.swift.candidates
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
