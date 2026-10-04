"""v1 Jev contract and HTTP admission, without model downloads."""

import ast
import json
import threading
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from ayaka.http_transport import BackendOverloaded, ServiceConfig
from ayaka.jev_api import build_answer, choice_confidence, score_confidence
from ayaka.primitives import DecisionResult
from ayaka.serve import BadRequest, DecisionService, parse_question, serve
from ayaka.tokenization import ToyTokenizer


class UniformDecision:
    tok = ToyTokenizer()
    max_seq_len = 512

    def __init__(self):
        self.calls = []

    def decide(self, state, questions):
        self.calls.append((state, questions))
        return [
            DecisionResult(q.type, [1 / len(q.candidates)] * len(q.candidates), {})
            for q in questions
        ]


def questions(structured=False):
    instruction = {"task": ["decide", {"strict": True}]} if structured else "decide"
    return {
        "n": {
            "type": "noul",
            "instructions": instruction,
            "criteria": {"false": {"holds": False}, "true": ["holds", True]}
            if structured
            else {"false": "no", "true": "yes"},
        },
        "c": {
            "type": "choice",
            "instructions": ["select", {"one": True}] if structured else "select one",
            "criteria": {"a": {"label": "alpha"}, "b": ["beta"], "c": None}
            if structured
            else {"a": "alpha", "b": "beta", "c": None},
        },
        "s": {
            "type": "score",
            "instructions": instruction,
            "criteria": [["low", {"rule": True}], {"level": 1}, "high"]
            if structured
            else ["low", "medium", "high"],
        },
    }


@contextmanager
def running_server(decision, *, key="test-key", model_id="ayaka-test-1.0.0", config=None):
    server = serve(
        decision,
        "ayaka-test",
        host="127.0.0.1",
        port=0,
        config=config or ServiceConfig(api_key=key, max_inflight=1),
        model_id=model_id,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.fixture
def api_server():
    decision = UniformDecision()
    with running_server(decision) as base:
        yield base, decision


def request(base, body=None, *, path="/v1/systemone", key="test-key", raw=None):
    headers = {"Content-Type": "application/json"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    data = raw if raw is not None else json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        response = urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return response.status, json.loads(response.read()), response.headers


@pytest.mark.parametrize("structured", [False, True])
def test_required_wire_fields_and_reserved_namespace(api_server, structured):
    base, decision = api_server
    body = {"state": {"x": [1]}, "questions": questions(structured), "ayaka": {}}
    status, result, headers = request(base, body)
    assert status == 200 and result["model"] == "ayaka-test-1.0.0"
    assert set(result) == {"model", "answers", "usage", "ayaka"}
    assert set(result["ayaka"]) == {"latency_ms"}
    assert uuid.UUID(headers["x-typesafe-request-id"]).version == 4
    assert set(result["answers"]["n"]) == {"type", "noul"}
    assert result["answers"]["n"]["noul"] == 0.5
    for name in ("c", "s"):
        assert result["answers"][name]["confidence"] == pytest.approx(0)
        assert sum(result["answers"][name]["probabilities"].values()) == pytest.approx(1)
    assert result["answers"]["s"]["legend"] == dict(
        zip(("0", "1", "2"), body["questions"]["s"]["criteria"], strict=True)
    )
    assert result["usage"]["input_tokens"] > 0 and result["usage"]["output_tokens"] == 0
    _, specs = decision.calls[-1]
    assert specs[0].instruction == (
        '{"task":["decide",{"strict":true}]}' if structured else "decide"
    )
    assert specs[1].candidates == (
        ['{"label":"alpha"}', '["beta"]', "c"] if structured else ["alpha", "beta", "c"]
    )
    # Even names resembling v2 controls have no effect in v1's reserved object.
    ignored = {**body, "ayaka": {"reasoning": "on", "media": [1], "questions": "unknown"}}
    _, again, _ = request(base, ignored)
    assert again["answers"] == result["answers"] and again["usage"] == result["usage"]


def test_models_auth_and_request_ids(api_server):
    base, _ = api_server
    for key in (None, "wrong"):
        assert request(base, {"questions": questions()}, key=key)[0] == 401
        assert request(base, path="/v1/models", key=key)[0] == 401
    status, listing, first = request(base, path="/v1/models")
    assert status == 200
    assert {m["name"] for m in listing["models"]} == {
        "jev-latest",
        "jev-preview",
        "ayaka-test",
        "ayaka-test-1.0.0",
    }
    assert all(set(m) == {"name", "description", "release_date"} for m in listing["models"])
    _, health, second = request(base, path="/health", key=None)
    assert health["status"] == "ok"
    assert first["x-typesafe-request-id"] != second["x-typesafe-request-id"]
    for alias in ("jev-latest", "jev-preview", "ayaka-test", "ayaka-test-1.0.0"):
        assert (
            request(base, {"model": alias, "questions": questions()})[1]["model"]
            == "ayaka-test-1.0.0"
        )


@pytest.mark.parametrize(
    "body,field",
    [
        ({"questions": {}}, "questions"),
        ({"questions": []}, "questions"),
        ({"questions": questions(), "state": 7}, "state"),
        ({"questions": questions(), "model": "unknown"}, "model"),
        ({"questions": questions(), "ayaka": []}, "ayaka"),
        ({"questions": {"q": 1}}, "questions.q"),
        ({"questions": {"q": {"type": "rank"}}}, "questions.q.type"),
        (
            {"questions": {"q": {"type": "choice", "criteria": {"a": 1, "b": "b"}}}},
            "questions.q.criteria.a",
        ),
        ({"questions": {"q": {"type": "choice", "criteria": {"a": None}}}}, "questions.q.criteria"),
        (
            {
                "questions": {
                    "q": {"type": "choice", "criteria": {str(i): None for i in range(256)}}
                }
            },
            "questions.q.criteria",
        ),
        ({"questions": {"q": {"type": "score", "criteria": ["x"] * 11}}}, "questions.q.criteria"),
        ({"questions": {"q": {"type": "score", "criteria": ["x"]}}}, "questions.q.criteria"),
        (
            {"questions": {"q": {"type": "noul", "criteria": {"true": None}}}},
            "questions.q.criteria.true",
        ),
        ({"questions": {"q": {"type": "noul", "instructions": 1}}}, "questions.q.instructions"),
        (
            {"questions": {"q": {"type": "noul", "instructions": "a", "instruction": "b"}}},
            "questions.q.instructions",
        ),
    ],
)
def test_validation_fields_before_inference(api_server, body, field):
    base, decision = api_server
    status, error, headers = request(base, body)
    assert status == 422 and error["field"] == field
    assert headers["x-typesafe-request-id"]
    assert not decision.calls


@pytest.mark.parametrize("raw", [b"{", b"null", b"[]", b"\xff"])
def test_invalid_json_and_body(api_server, raw):
    status, body, _ = request(api_server[0], raw=raw)
    assert status == 422 and body["field"] == "body"


def test_bounded_admission_and_recovery(api_server, monkeypatch):
    base, decision = api_server
    entered, release = threading.Event(), threading.Event()
    original = decision.decide

    def blocked(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)

    monkeypatch.setattr(decision, "decide", blocked)
    result = []
    thread = threading.Thread(
        target=lambda: result.append(request(base, {"questions": questions()}))
    )
    thread.start()
    try:
        assert entered.wait(5)
        status, _, headers = request(base, {"questions": questions()})
        assert status == 429 and headers["retry-after"] == "1"
    finally:
        release.set()
        thread.join(10)
    assert result[0][0] == 200
    assert request(base, {"questions": questions()})[0] == 200


@pytest.mark.parametrize(
    "exception",
    [BackendOverloaded("private"), urllib.error.HTTPError("fixture", 503, "private", {}, None)],
)
def test_backend_overload_and_recovery(api_server, monkeypatch, exception):
    base, decision = api_server
    original = decision.decide

    def fail(*args):
        raise exception

    monkeypatch.setattr(decision, "decide", fail)
    status, body, headers = request(base, {"questions": questions()})
    assert status == 529 and headers["retry-after"] == "1"
    assert "private" not in json.dumps(body)
    monkeypatch.setattr(decision, "decide", original)
    assert request(base, {"questions": questions()})[0] == 200


def test_documented_confidence_examples_and_first_max():
    assert choice_confidence({"a": 0.5, "b": 0.3, "c": 0.2}) == pytest.approx(0.25)
    assert score_confidence([0, 0.5, 0.5]) == 0.25
    assert score_confidence([0.5, 0, 0.5]) == 0
    for confidence in (choice_confidence, score_confidence):
        assert confidence([0.5, 0.5]) == 0
        assert confidence([1, 0]) == 1
    assert "confidence" not in build_answer("noul", {"false": 0.2, "true": 0.8})


def test_legacy_aliases_and_limits():
    spec, labels = parse_question({"type": "choice", "instruction": "i", "criteria": ["a", "b"]})
    assert labels == spec.candidates == ["a", "b"] and spec.instruction == "i"
    spec, labels = parse_question(
        {"type": "score", "criteria": {"10": "high", "-2": "low", "other": "medium"}}
    )
    assert labels == ["10", "-2", "other"] and spec.ordinals == [10, -2, 2]
    assert (
        len(parse_question({"type": "choice", "criteria": {str(i): None for i in range(255)}})[1])
        == 255
    )
    assert len(parse_question({"type": "score", "criteria": ["x"] * 10})[1]) == 10
    with pytest.raises(BadRequest):
        parse_question({"type": "choice", "criteria": ["a", "a"]})


def test_key_environment_and_unauthenticated_default(monkeypatch):
    monkeypatch.delenv("AYAKA_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_KEY", "secret")
    assert ServiceConfig(api_key_env="CUSTOM_KEY").key == "secret"
    with running_server(UniformDecision(), config=ServiceConfig(api_key_env="CUSTOM_KEY")) as base:
        assert request(base, {"questions": questions()}, key=None)[0] == 401
        assert request(base, {"questions": questions()}, key="secret")[0] == 200
        monkeypatch.setenv("CUSTOM_KEY", "changed")
        assert request(base, {"questions": questions()}, key="secret")[0] == 200
    with running_server(UniformDecision(), key=None) as base:
        assert request(base, {"questions": questions()}, key=None)[0] == 200
    with pytest.raises(ValueError, match="positive"):
        ServiceConfig(max_inflight=0)
    assert (
        DecisionService(UniformDecision(), "ayaka-small").handle({"questions": questions()})[
            "model"
        ]
        == "ayaka-small-1.0.0"
    )


def test_confidence_implementation_stays_in_shared_module():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "ayaka" / "serve.py", root / "ayaka" / "http_transport.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.FunctionDef)
            and node.name in ("choice_confidence", "score_confidence")
            for node in ast.walk(tree)
        )
