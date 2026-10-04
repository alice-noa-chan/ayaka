"""CPU contract and service checks; no SDK, model downloads or GPU required."""

import ast
import base64
import io
import json
import socket
import threading
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from ayaka.http_transport import BackendOverloaded, ServiceConfig
from ayaka.jev_api import (
    ModelCatalog,
    ValidationError,
    build_answer,
    choice_confidence,
    normalize_request,
    parse_question,
    score_confidence,
)
from ayaka.swift.readers import FakeReader


def request(base, body=None, *, path="/v1/systemone", key="test-key"):
    headers = {"Content-Type": "application/json"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        base + path, headers=headers, data=None if body is None else json.dumps(body).encode()
    )
    try:
        result = urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as exc:
        result = exc
    with result:
        data = result.read()
        return result.status, data if path == "/metrics" else json.loads(data), result.headers


def make_service(kind):
    if kind == "swift":
        from ayaka.swift.server import DecisionService

        return DecisionService(FakeReader(), "ayaka-test", model_id="ayaka-test-1.0.0")
    from ayaka.serve import DecisionService

    class StubDecision:
        tok = SimpleNamespace(encode=lambda value: list(value))

        def decide(self, state, specs):
            return [
                SimpleNamespace(probs=[1 / len(s.candidates)] * len(s.candidates), extras={})
                for s in specs
            ]

    service = DecisionService(StubDecision(), "ayaka-test", model_id="ayaka-test-1.0.0")
    service._count_tokens = lambda state, parsed: 12
    return service


@pytest.fixture(params=["swift", "v1"])
def api_server(request):
    service = make_service(request.param)
    module = __import__(
        "ayaka.swift.server" if request.param == "swift" else "ayaka.serve",
        fromlist=["make_handler"],
    )
    from ayaka.http_transport import server_for

    config = ServiceConfig(api_key="test-key", max_inflight=1, request_timeout_s=0.25)
    handler = module.make_handler(service, config=config)
    server = server_for(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", service, handler.runtime
    server.shutdown()
    thread.join()
    server.server_close()
    if hasattr(service, "close"):
        service.close()


def questions(structured=False):
    instruction = {"task": ["judge", "only evidence"]} if structured else "Judge"
    description = ["low", {"rule": True}] if structured else "low"
    return {
        "n": {
            "type": "noul",
            "instructions": instruction,
            "criteria": {"true": description, "false": {"no": []} if structured else "no"},
        },
        "c": {
            "type": "choice",
            "instructions": instruction,
            "criteria": {"a": description, "b": None, "c": {"rule": 3} if structured else "other"},
        },
        "s": {
            "type": "score",
            "instructions": instruction,
            "criteria": [description, {"level": 1} if structured else "high"],
        },
    }


@pytest.mark.parametrize("structured", [False, True])
def test_required_wire_fields_and_extensions(api_server, structured):
    base, _, _ = api_server
    status, body, headers = request(
        base,
        {
            "state": {"x": [1]},
            "model": "jev-latest",
            "questions": questions(structured),
            "ayaka": {"reasoning": {"mode": "off"}},
        },
    )
    assert status == 200
    assert body["model"] == "ayaka-test-1.0.0"
    assert set(body) == {"model", "usage", "answers", "ayaka"}
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}
    assert all(type(v) is int and v >= 0 for v in body["usage"].values())
    assert uuid.UUID(headers["x-typesafe-request-id"]).version == 4
    assert set(body["answers"]["n"]) == {"type", "noul", "ayaka"}
    for name in ("c", "s"):
        answer = body["answers"][name]
        assert 0 <= answer["confidence"] <= 1
        assert sum(answer["probabilities"].values()) == pytest.approx(1)
        assert answer["ayaka"]["calibration"] == "unfitted"
        assert answer["ayaka"]["route"] == "direct"
    assert body["answers"]["s"]["legend"] == {
        "0": ["low", {"rule": True}] if structured else "low",
        "1": {"level": 1} if structured else "high",
    }


def test_models_auth_and_request_ids(api_server):
    base, _, _ = api_server
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
    for alias in ("jev-preview", "ayaka-test", "ayaka-test-1.0.0"):
        assert (
            request(base, {"model": alias, "questions": questions()})[1]["model"]
            == "ayaka-test-1.0.0"
        )
    status, error, _ = request(base, {"model": "unknown", "questions": questions()})
    assert status == 422 and error["field"] == "model" and "unknown" in error["error"]


@pytest.mark.parametrize(
    "body,field",
    [
        ({"state": 7, "questions": questions()}, "state"),
        (
            {
                "questions": {
                    "bad": {"type": "choice", "criteria": {str(i): None for i in range(256)}}
                }
            },
            "questions.bad.criteria",
        ),
        (
            {"questions": {"bad": {"type": "score", "criteria": ["x"] * 11}}},
            "questions.bad.criteria",
        ),
        (
            {"questions": {"bad": {"type": "noul", "criteria": {"true": None}}}},
            "questions.bad.criteria.true",
        ),
        (
            {"questions": {"bad": {"type": "choice", "criteria": {"a": 1, "b": "b"}}}},
            "questions.bad.criteria.a",
        ),
        (
            {
                "questions": questions(),
                "ayaka": {"reasoning": {"mode": "off"}},
                "options": {"reasoning": {"mode": "on"}},
            },
            "ayaka.reasoning",
        ),
        ({"questions": questions(), "ayaka": {"media": []}, "media": [1]}, "ayaka.media"),
        (
            {
                "questions": questions(),
                "ayaka": {"questions": {"n": {"reasoning": {"mode": "off"}}}},
                "options": [],
            },
            "options",
        ),
    ],
)
def test_validation_fields(api_server, body, field):
    status, error, _ = request(api_server[0], body)
    assert status == 422 and error["field"] == field


def test_deadline_preserves_backpressure_and_metrics(api_server):
    base, service, runtime = api_server
    entered, release = threading.Event(), threading.Event()
    original = service.handle

    def blocked(body):
        entered.set()
        assert release.wait(5)
        return original(body)

    service.handle = blocked
    result = []
    thread = threading.Thread(
        target=lambda: result.append(
            request(base, {"state": "private-state", "questions": questions()})
        )
    )
    thread.start()
    try:
        assert entered.wait(5)
        status, _, headers = request(base, {"questions": questions()})
        assert status == 429 and headers["retry-after"] == "1"
        thread.join(3)
        assert result[0][0] == 504
        assert request(base, {"questions": questions()})[0] == 429
    finally:
        release.set()
        thread.join(5)
    # Wait on the bounded worker itself, without arbitrary sleeps.
    runtime.pool.submit(lambda: None).result(timeout=5)
    assert request(base, {"questions": questions()})[0] == 200
    status, metrics, _ = request(base, path="/metrics", key=None)
    assert status == 200
    text = metrics.decode()
    assert 'status="429"' in text and 'status="504"' in text
    assert 'type="choice",route="direct"' in text
    assert "ayaka_request_latency_seconds_bucket" in text and "ayaka_tokens_total" in text
    assert "private-state" not in text and "test-key" not in text


@pytest.mark.parametrize(
    "exception,status", [(BackendOverloaded("private"), 529), (RuntimeError("private"), 502)]
)
def test_backend_failures(api_server, exception, status):
    base, service, _ = api_server

    def fail(body):
        raise exception

    service.handle = fail
    actual, body, headers = request(base, {"questions": questions()})
    assert actual == status and "private" not in json.dumps(body)
    if status == 529:
        assert headers["retry-after"] == "1"


def test_nonfinite_backend_response_returns_502(api_server):
    base, service, _ = api_server
    service.handle = lambda body: {
        "answers": {"n": {"type": "noul", "noul": float("nan")}},
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    assert request(base, {"questions": {"n": questions()["n"]}})[0] == 502


def test_slow_body_deadline(api_server):
    base, _, _ = api_server
    port = int(base.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
        client.sendall(
            b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer test-key\r\nContent-Length: 100\r\n\r\n{"
        )
        result = client.recv(4096)
    assert b"504" in result


def test_doc_confidence_examples_and_first_max():
    assert score_confidence([0, 0.5, 0.5]) == 0.25
    assert score_confidence([0.5, 0, 0.5]) == 0
    assert choice_confidence({"a": 0.5, "b": 0.3, "c": 0.2}) == pytest.approx(0.25)
    assert choice_confidence([0.6, 0.3, 0.1]) == pytest.approx(0.4)
    assert choice_confidence([0.6, 0.2, 0.2]) == pytest.approx(0.4)
    assert score_confidence([0, 0.57, 0.43]) == pytest.approx(0.355)
    for confidence in (choice_confidence, score_confidence):
        assert confidence([0.5, 0.5]) == 0
        assert confidence([1, 0]) == 1
    assert "confidence" not in build_answer("noul", {"false": 0.2, "true": 0.8})


def test_limits_rendering_and_conflicts():
    assert (
        len(
            parse_question(
                {"type": "choice", "criteria": {str(i): None for i in range(255)}}
            ).labels
        )
        == 255
    )
    assert len(parse_question({"type": "score", "criteria": ["x"] * 10}).labels) == 10
    body = {
        "questions": questions(),
        "options": {"reasoning": {"mode": "off"}},
        "ayaka": {"reasoning": {"mode": "off"}},
    }
    assert normalize_request(body)["options"] == body["options"]
    assert "ayaka" in body  # Input remains untouched.
    with pytest.raises(ValidationError, match="conflicting"):
        normalize_request(
            {
                "questions": {"q": {"type": "noul", "reasoning": {"mode": "off"}}},
                "ayaka": {"questions": {"q": {"reasoning": {"mode": "on"}}}},
            }
        )
    assert ModelCatalog("ayaka", "ayaka-1.0").resolve() == "ayaka-1.0"


def test_confidence_computation_lives_only_in_jev_api():
    paths = [Path("ayaka/serve.py"), *Path("ayaka/swift").glob("*.py")]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.name not in ("choice_confidence", "score_confidence"), path
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == "confidence"
                    ):
                        assert isinstance(node.value, ast.Call) and isinstance(
                            node.value.func, ast.Name
                        )
                        assert node.value.func.id in ("choice_confidence", "score_confidence"), path


def test_bind_key_environment(monkeypatch):
    monkeypatch.delenv("AYAKA_API_KEY", raising=False)
    for host in ("127.0.0.1", "::1", "localhost"):
        ServiceConfig().validate_bind(host)
    with pytest.raises(ValueError, match="API key"):
        ServiceConfig().validate_bind("0.0.0.0")
    ServiceConfig(allow_no_key=True).validate_bind("0.0.0.0")
    monkeypatch.setenv("CUSTOM_KEY", "secret")
    config = ServiceConfig(api_key_env="CUSTOM_KEY")
    assert config.key == "secret"
    config.validate_bind("0.0.0.0")


def image_media():
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (2, 3), "red").save(out, format="PNG")
    return [
        {
            "type": "image",
            "mime_type": "image/png",
            "data": base64.b64encode(out.getvalue()).decode(),
        }
    ]


def test_swift_images_forwarded_and_text_policy_isolated():
    from ayaka.swift.policy import Policy
    from ayaka.swift.server import DecisionService

    reader = FakeReader(lambda messages, letters: {"A": 0.8, "B": 0.2})
    policy = Policy(t_choice=3, t_noul=3, commit_margin=0, fitted_on="text")
    service = DecisionService(reader, "fake", policy, diagnostic=True)
    try:
        body = {
            "state": "invoice",
            "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
            "ayaka": {"media": image_media()},
        }
        result = service.handle(body)
        answer = result["answers"]["q"]
        assert answer["probabilities"] == pytest.approx({"a": 0.8, "b": 0.2})
        assert answer["confidence"] == pytest.approx(0.6)
        assert answer["ayaka"]["calibration"] == "unvalidated_image"
        parts = reader.calls[0][0][-1]["content"]
        assert parts[0]["type"] == "text"
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
        legacy = {**body, "media": body["ayaka"]["media"]}
        assert service.handle(legacy)["answers"] == result["answers"]
        text = service.handle({k: v for k, v in body.items() if k != "ayaka"})
        assert text["answers"]["q"]["probabilities"] != answer["probabilities"]
        assert service.policy.t_choice == policy.t_choice and service.policy.letter_bias is None
    finally:
        service.close()


@pytest.mark.parametrize(
    "media",
    [
        [],
        [{"type": "image", "mime_type": "image/png", "data": "bad"}],
        [{"type": "image", "url": "https://example.com"}],
    ],
)
def test_invalid_image_before_backend(media):
    from ayaka.swift.server import DecisionService

    service = DecisionService(FakeReader(), "fake")
    try:
        with pytest.raises(ValidationError):
            service.handle({"questions": questions(), "ayaka": {"media": media}})
        assert not service.reader.calls
    finally:
        service.close()


def test_images_over_http(api_server):
    base, service, _ = api_server
    status, response, _ = request(
        base, {"state": "invoice", "questions": questions(), "ayaka": {"media": image_media()}}
    )
    if hasattr(service, "reader"):
        assert status == 200
        assert all(
            a["ayaka"]["calibration"] == "unvalidated_image" for a in response["answers"].values()
        )
        assert response["answers"]["c"]["confidence"] == pytest.approx(0)
    else:
        assert status == 422 and "image backend" in response["error"]


def test_reasoned_and_grouped_confidence_and_routes(monkeypatch):
    from ayaka.swift import server
    from ayaka.swift.grouping import QuestionRead

    service = server.DecisionService(FakeReader(), "fake")
    try:
        monkeypatch.setattr(
            server,
            "system_read",
            lambda *a, **kw: QuestionRead({"a": 0.5, "b": 0.3, "c": 0.2}, 10, 2, 0.01, passes=2),
        )
        answer = service.handle({"questions": {"q": questions()["c"]}})["answers"]["q"]
        assert answer["ayaka"]["route"] == "reasoned"
        assert answer["confidence"] == pytest.approx(0.25)
    finally:
        service.close()
    service = server.DecisionService(FakeReader(), "fake")
    monkeypatch.undo()
    try:
        answer = service.handle(
            {"questions": {"q": {"type": "choice", "criteria": {str(i): None for i in range(255)}}}}
        )["answers"]["q"]
        assert answer["ayaka"]["route"] == "grouped"
        assert answer["confidence"] == choice_confidence(answer["probabilities"])
    finally:
        service.close()


def test_vllm_image_parts_expanded_usage_and_overload(monkeypatch):
    import math

    from test_swift_readers import StubTokenizer

    from ayaka.swift.media import ImageReader, image_parts
    from ayaka.swift.readers import VLLMChatReader

    payload = {
        "prompt_token_ids": [7] * 25,
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "token_id:65",
                            "top_logprobs": [
                                {"token": "token_id:65", "logprob": math.log(0.6)},
                                {"token": "token_id:66", "logprob": math.log(0.4)},
                            ],
                        }
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 25, "completion_tokens": 1},
    }
    seen = []

    def backend(request, timeout):
        seen.append(json.loads(request.data))
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", backend)
    reader = VLLMChatReader("http://fixture", "fake", tokenizer=StubTokenizer())
    result = ImageReader(reader, image_parts(image_media())).read(
        [{"role": "user", "content": "image"}], ["A", "B"]
    )
    assert result.input_tokens == 25 and result.input_token_ids == payload["prompt_token_ids"]
    assert seen[0]["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert result.letter_probs == pytest.approx({"A": 0.6, "B": 0.4})

    def overloaded(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 503, "overloaded", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", overloaded)
    with pytest.raises(BackendOverloaded):
        reader.read([{"role": "user", "content": "text"}], ["A", "B"])


def test_image_mime_animation_and_count_limits():
    from ayaka.swift.media import image_parts

    sample = image_media()
    for media in (sample * 5, [{**sample[0], "mime_type": "image/jpeg"}]):
        with pytest.raises(ValidationError):
            image_parts(media)
    from PIL import Image

    out = io.BytesIO()
    first, second = Image.new("RGB", (2, 2), "red"), Image.new("RGB", (2, 2), "blue")
    first.save(out, format="PNG", save_all=True, append_images=[second])
    with pytest.raises(ValidationError, match="single frame"):
        image_parts([{**sample[0], "data": base64.b64encode(out.getvalue()).decode()}])
