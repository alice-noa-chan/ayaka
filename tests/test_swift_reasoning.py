"""CPU trace-generator and HTTP fakes; no model loading."""

import io
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ayaka.swift.collect import adapt_jevbench, collect, load_reads
from ayaka.swift.policy import Policy
from ayaka.swift.prompt import parse_question, render_question
from ayaka.swift.readers import FakeReader, VLLMChatReader
from ayaka.swift.reasoning import FINAL_INSTRUCTION, TRACE_INSTRUCTION, TraceResult, reasoned_read
from ayaka.swift.router import FEATURES, validate_pairs
from ayaka.swift.server import DecisionService


def test_hf_equivalent_with_fake_generator_and_real_raw_gather():
    import torch

    from ayaka.swift.readers import HFReader

    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            calls.append((messages, kwargs))
            if not tokenize:
                return "prompt"
            if kwargs.get("return_tensors"):
                return {"input_ids": torch.tensor([[1, 2]])}
            return [1, 2]

        def encode(self, text, add_special_tokens=False):
            return [1, 2] + ([ord(text[-1])] if text[-1] in "AB" else [])

        def decode(self, ids, skip_special_tokens=False):
            return "PRIVATE HF STEPS"

    class Model:
        generation_config = SimpleNamespace(eos_token_id=9)

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False and kwargs["max_new_tokens"] == 384
            return torch.tensor([[1, 2, 7, 8, 9]])

        def __call__(self, **kwargs):
            logits = torch.zeros(1, 2, 67)
            logits[0, -1, 66] = 2
            return SimpleNamespace(logits=logits)

    reader = HFReader("already-local-fake")
    reader.tokenizer, reader.model = Tokenizer(), Model()
    result, metadata = reasoned_read(reader, "12", parse_question({"type": "noul"}))
    assert result.raw_probs["true"] > 0.8
    assert result.input_tokens == 4 and result.output_tokens == 4
    assert metadata["trace_tokens"] == 3 and metadata["finish_reason"] == "eos"
    assert all(kwargs["enable_thinking"] is False for _, kwargs in calls)


class TraceReader(FakeReader):
    def __init__(self, results=None, *, capped=False, latency=0.02):
        super().__init__(results)
        self.trace_calls = []
        self.capped = capped
        self.trace_latency = latency

    def generate_trace(self, messages, max_tokens):
        self.trace_calls.append((messages, max_tokens))
        return TraceResult(
            "PRIVATE WORKED STEPS",
            11,
            max_tokens if self.capped else 3,
            "length" if self.capped else "eos",
            self.trace_latency,
        )


def always_router():
    return {
        "features": FEATURES.copy(),
        "means": [0.0] * len(FEATURES),
        "scales": [1.0] * len(FEATURES),
        "weights": [0.0] * len(FEATURES),
        "intercept": 0.0,
        "threshold": 0.0,
        "rate_cap": 0.1,
        "max_tokens": 384,
    }


def test_http_internal_route_and_explicit_request_still_422():
    import threading
    import urllib.error
    import urllib.request

    from ayaka.swift.server import serve

    reader = TraceReader()
    service = DecisionService(
        reader, "fake", Policy(reasoning_route=always_router()), diagnostic=True
    )
    server = serve(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    body = {"state": "12", "questions": {"q": {"type": "noul"}}}

    def request(value):
        return urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/systemone",
            data=json.dumps(value).encode(),
            headers={"Content-Type": "application/json"},
        )

    try:
        with urllib.request.urlopen(request(body), timeout=5) as response:
            payload = json.load(response)
        assert payload["usage"] == {"input_tokens": 31, "output_tokens": 5}
        assert "PRIVATE" not in json.dumps(payload)
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(
                request({**body, "options": {"reasoning": {"mode": "on"}}}), timeout=5
            )
        assert caught.value.code == 422
        caught.value.close()
        assert len(reader.trace_calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        service.close()


@pytest.mark.parametrize("variant", ["min", "cygnet", "rules", "labeled"])
@pytest.mark.parametrize("capped", [False, True])
def test_two_call_structure_and_capped_trace_still_read(variant, capped):
    reader = TraceReader(capped=capped)
    question = parse_question(
        {"type": "choice", "instructions": "calculate", "criteria": {"cat": "one", "dog": "two"}}
    )
    original, _ = render_question("12", question, prompt_variant=variant)
    read, metadata = reasoned_read(reader, "12", question, prompt_variant=variant)
    first, budget = reader.trace_calls[0]
    assert first == [
        {"role": "user", "content": original[-1]["content"] + "\n\n" + TRACE_INSTRUCTION}
    ]
    assert budget == 384
    assert reader.calls[0][0] == [
        *first,
        {"role": "assistant", "content": "PRIVATE WORKED STEPS"},
        {"role": "user", "content": FINAL_INSTRUCTION},
    ]
    assert metadata["finish_reason"] == ("length" if capped else "eos")
    assert metadata["length_capped"] is capped
    assert read.input_tokens == 21
    assert read.output_tokens == (385 if capped else 4)
    assert read.latency_s == pytest.approx(0.03)


def test_internal_service_route_usage_and_trace_never_leaks():
    reader = TraceReader([{"A": 0.6, "B": 0.4}, {"A": 0.1, "B": 0.9}])
    policy = Policy(reasoning_route=always_router())
    with pytest.raises(ValueError, match="adoption gate"):
        DecisionService(reader, "fake", policy)
    service = DecisionService(reader, "fake", policy, diagnostic=True)
    body = {"state": "12", "questions": {"q": {"type": "choice", "criteria": ["first", "second"]}}}
    try:
        result = service.handle(body)
        assert result["answers"]["q"]["choice"] == "second"
        assert result["usage"] == {"input_tokens": 31, "output_tokens": 5}
        assert "PRIVATE" not in json.dumps(result)
        assert len(reader.calls) == 2 and len(reader.trace_calls) == 1
        from ayaka.swift.prompt import InvalidQuestion

        with pytest.raises(InvalidQuestion, match="reasoning not supported"):
            service.handle({**body, "options": {"reasoning": {"mode": "on", "max_tokens": 20}}})
        assert len(reader.calls) == 2
    finally:
        service.close()


def test_default_server_has_no_trace_generation():
    reader = TraceReader()
    service = DecisionService(reader, "fake")
    try:
        service.handle({"questions": {"q": {"type": "noul"}}})
        assert not reader.trace_calls
    finally:
        service.close()


def test_vllm_trace_uses_bounded_greedy_eos_then_exact_raw_gather(monkeypatch):
    reader = VLLMChatReader("http://fake", "frozen", revision="a" * 40)
    fake = FakeReader()
    reader.describe = fake.describe
    bodies = []

    def urlopen(request, timeout):
        body = json.loads(request.data)
        bodies.append(body)
        description = fake.describe(body["messages"], ["A", "B"])
        if len(bodies) == 1:
            response = {
                "choices": [
                    {"message": {"content": "two plus two is four"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 6},
                "prompt_token_ids": description["input_token_ids"],
            }
        else:
            response = {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "token": "token_id:65",
                                    "top_logprobs": [
                                        {"token": "token_id:65", "logprob": 1},
                                        {"token": "token_id:66", "logprob": 0},
                                    ],
                                }
                            ]
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "prompt_token_ids": description["input_token_ids"],
            }
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    result, metadata = reasoned_read(reader, "4", parse_question({"type": "noul"}))
    assert bodies[0]["temperature"] == 0 and bodies[0]["max_tokens"] == 384
    assert bodies[0]["chat_template_kwargs"]["enable_thinking"] is False
    assert bodies[0]["ignore_eos"] is False
    assert bodies[1]["max_tokens"] == 1
    assert bodies[1]["logprob_token_ids"] == [65, 66]
    assert result.output_tokens == 7 and result.input_tokens == 2
    assert metadata["finish_reason"] == "eos"


def test_reasoned_collection_candidate_binding_resume_and_budget_guard(tmp_path):
    records = []
    for index, state in enumerate(["plain", "plain", "on 2026-10-04"]):
        records.append(
            adapt_jevbench(
                {
                    "id": str(index),
                    "state": state,
                    "source": "private",
                    "public": False,
                    "split": "calibration",
                    "labels": ["a", "b"],
                    "expected": "a",
                    "question": {"type": "choice", "criteria": ["a", "b"]},
                }
            )
        )
    direct_reader = TraceReader(
        [{"A": 0.99, "B": 0.01}, {"A": 0.9, "B": 0.1}, {"A": 0.99, "B": 0.01}]
    )
    direct_reader.backend, direct_reader.logprobs_mode = "hf", "raw_logits"
    direct_path = tmp_path / "direct.jsonl"
    collect(iter(records), direct_reader, direct_path, model="frozen", revision="a" * 40)
    direct = load_reads([direct_path])
    reader = TraceReader(capped=True)
    reader.backend, reader.logprobs_mode = "hf", "raw_logits"
    path = tmp_path / "reasoned.jsonl"
    assert (
        collect(
            iter(records),
            reader,
            path,
            model="frozen",
            revision="a" * 40,
            reasoned=True,
            direct_reads=direct,
        )
        == 2
    )
    rows = load_reads([path])
    assert [r["id"] for r in rows] == ["1", "2"]
    paired = validate_pairs(direct, rows)
    assert all(r["length_capped"] for r in paired.values())
    assert (
        collect(
            iter(records),
            reader,
            path,
            model="frozen",
            revision="a" * 40,
            reasoned=True,
            direct_reads=direct,
        )
        == 0
    )
    assert len(reader.trace_calls) == 2
    with pytest.raises(ValueError, match="budget/recipe"):
        collect(
            iter(records),
            reader,
            path,
            model="frozen",
            revision="a" * 40,
            reasoned=True,
            direct_reads=direct,
            trace_max_tokens=32,
        )
    assert len(reader.trace_calls) == 2
    with pytest.raises(ValueError, match="cached read differs"):
        collect(
            iter([*records[:2], replace(records[2], state="changed")]),
            reader,
            tmp_path / "fresh.jsonl",
            model="frozen",
            revision="a" * 40,
            reasoned=True,
            direct_reads=direct,
        )
    assert len(reader.trace_calls) == 2
