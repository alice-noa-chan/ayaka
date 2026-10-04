import json
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import pytest

from ayaka.swift.policy import Policy
from ayaka.swift.readers import FakeReader
from ayaka.swift.server import DecisionService, serve


@pytest.fixture
def running_server():
    reader = FakeReader(
        lambda messages, letters: {letter: 1 if i == 0 else 3 for i, letter in enumerate(letters)}
    )
    service = DecisionService(reader, "fake", Policy(), max_questions=4)
    server = serve(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", reader, service
    server.shutdown()
    thread.join()
    server.server_close()
    service.close()


def post(base, body):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    request = urllib.request.Request(
        base + "/v1/systemone", data=data, headers={"Content-Type": "application/json"}
    )
    try:
        response = urllib.request.urlopen(request, timeout=5)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return response.status, json.load(response)


def test_http_roundtrip_and_gets(running_server):
    base, reader, _ = running_server
    status, response = post(
        base,
        {
            "state": {"x": 1},
            "questions": {
                "n": {"type": "noul"},
                "c": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
                "s": {"type": "score", "criteria": ["low", "high"]},
            },
        },
    )
    assert status == 200
    assert response == {
        "model": "fake",
        "answers": {
            "n": {"type": "noul", "noul": 0.75},
            "c": {"type": "choice", "choice": "b", "probabilities": {"a": 0.25, "b": 0.75}},
            "s": {"type": "score", "score": 0.75, "probabilities": {"0": 0.25, "1": 0.75}},
        },
        "usage": {"input_tokens": 30, "output_tokens": 3},
    }
    assert len(reader.calls) == 3
    for route, expected in [
        ("/health", {"status": "ok", "model": "fake", "prompt_variant": "min"}),
        ("/v1/models", {"data": [{"id": "fake"}]}),
    ]:
        with urllib.request.urlopen(base + route, timeout=5) as result:
            assert json.load(result) == expected


@pytest.mark.parametrize(
    "request_reasoning,question,accepted",
    [
        ({"mode": "on", "effort": "high"}, None, False),
        (None, {"mode": "on", "effort": "high"}, False),
        ({"mode": "auto", "max_tokens": 1}, None, False),
        ({"mode": "on", "effort": "high"}, {"mode": "off"}, True),
        ({"mode": "on", "effort": "high"}, {"max_tokens": 0}, True),
        ({"mode": "off"}, None, True),
        ({"mode": "auto", "max_tokens": 0}, None, True),
        ({"mode": "bad"}, None, False),
        ({"max_tokens": True}, None, False),
        ({"max_tokens": 1025}, None, False),
        ({"unknown": 1}, None, False),
        ([], None, False),
        ({"mode": "off"}, {"effort": "bad"}, False),
    ],
)
def test_reasoning_budget_validation_before_reader(
    running_server, request_reasoning, question, accepted
):
    base, reader, _ = running_server
    body = {"questions": {"q": {"type": "noul"}}}
    if request_reasoning is not None:
        body["options"] = {"reasoning": request_reasoning}
    if question is not None:
        body["questions"]["q"]["reasoning"] = question
    status, response = post(base, body)
    assert status == (200 if accepted else 422)
    assert len(reader.calls) == int(accepted)
    if not accepted and (request_reasoning or question) in (
        {"mode": "on", "effort": "high"},
        {"mode": "auto", "max_tokens": 1},
    ):
        assert response["error"] == "reasoning not supported by this readout"


@pytest.mark.parametrize(
    "body",
    [
        {"options": {"reasoning": None}, "questions": {"q": {"type": "noul"}}},
        {"options": [], "questions": {"q": {"type": "noul"}}},
        {"questions": {"ok": {"type": "noul"}, "bad": {"type": "noul", "reasoning": None}}},
        {
            "questions": {
                "ok": {"type": "noul"},
                "bad": {"type": "noul", "reasoning": {"mode": "on"}},
            }
        },
    ],
)
def test_malformed_or_unsupported_reasoning_is_atomic(running_server, body):
    base, reader, _ = running_server
    assert post(base, body)[0] == 422
    assert not reader.calls


@pytest.mark.parametrize(
    "criteria", [{"0": "low", "1": "mid", "100": "high"}, {"2": "low", "4": "high"}]
)
def test_score_api_rejects_nonuniform_and_noncontiguous_ordinals(running_server, criteria):
    base, reader, _ = running_server
    status, response = post(
        base,
        {"questions": {"ok": {"type": "noul"}, "bad": {"type": "score", "criteria": criteria}}},
    )
    assert status == 422 and "uniform and contiguous" in response["error"]
    assert not reader.calls


def test_score_api_sorts_contiguous_ordinals(running_server):
    base, reader, _ = running_server
    status, response = post(
        base, {"questions": {"s": {"type": "score", "criteria": {"3": "high", "2": "low"}}}}
    )
    assert status == 200
    assert response["answers"]["s"]["score"] == pytest.approx(2.75)
    assert list(response["answers"]["s"]["probabilities"]) == ["2", "3"]
    assert len(reader.calls) == 1


def test_policy_variant_mismatch_refused_and_force_renders_requested_variant(tmp_path, monkeypatch):
    from ayaka.swift import server as module
    from ayaka.swift.prompt import RULES_SYSTEM

    reader = FakeReader()
    policy = Policy(prompt_variant="cygnet")
    path = tmp_path / "policy.json"
    policy.save(path)
    with pytest.raises(ValueError, match="--force-variant"):
        DecisionService(reader, "fake", Policy.load(path), prompt_variant="rules")
    monkeypatch.setattr(module, "reader_from_args", lambda args: reader)
    with pytest.raises(SystemExit):
        module.main(["--policy", str(path), "--prompt-variant", "rules"])
    assert not reader.calls
    service = DecisionService(reader, "fake", policy, prompt_variant="rules", force_variant=True)
    try:
        service.handle({"questions": {"n": {"type": "noul"}}})
        assert reader.calls[0][0][0]["content"] == RULES_SYSTEM
    finally:
        service.close()
    service = DecisionService(reader, "fake", prompt_variant="rules")
    assert service.policy.prompt_variant == "rules"
    service.close()


@pytest.mark.parametrize(
    "body,status",
    [
        (b"{", 400),
        (b"", 400),
        ([], 422),
        ({}, 422),
        ({"questions": {}}, 422),
        ({"questions": {"x": {"type": "choice", "criteria": ["a"]}}}, 422),
        ({"questions": {str(i): {"type": "noul"} for i in range(5)}}, 422),
        ({"questions": {"x": {"type": "choice", "criteria": [str(i) for i in range(521)]}}}, 422),
    ],
)
def test_http_validation(running_server, body, status):
    base, reader, _ = running_server
    actual, response = post(base, body)
    assert actual == status
    assert "error" in response
    assert not reader.calls


def test_backend_failure_returns_502(running_server):
    base, _, service = running_server

    class BrokenReader:
        def read(self, messages, letters):
            raise RuntimeError("secret backend details")

    service.reader = BrokenReader()
    status, response = post(base, {"questions": {"x": {"type": "noul"}}})
    assert status == 502
    assert response == {"error": "backend failure: RuntimeError"}


def test_200_sequential_raw_socket_error_responses(running_server):
    base, _, service = running_server

    class BrokenReader:
        def read(self, messages, letters):
            raise RuntimeError("backend failure")

    service.reader = BrokenReader()
    address = urlsplit(base)
    valid = json.dumps({"questions": {"x": {"type": "noul"}}}).encode()
    cases = [
        ("/v1/systemone", b"{", "1", 400),
        ("/v1/systemone", b"{}", "2", 422),
        ("/v1/systemone", valid, str(len(valid)), 502),
        ("/missing", b"unread body" * 1000, "11000", 404),
    ]
    for index in range(200):
        path, body, length, status = cases[index % len(cases)]
        version = "1.0" if index % 2 else "1.1"
        headers = (
            f"POST {path} HTTP/{version}\r\nHost: localhost\r\n"
            f"Content-Length: {length}\r\nConnection: close\r\n\r\n"
        ).encode()
        with socket.create_connection((address.hostname, address.port), timeout=5) as client:
            # Separate sends expose closes while client body data is in flight.
            # Do not half-close or retry: the server must deliver a clean EOF.
            client.sendall(headers)
            client.sendall(body)
            received = bytearray()
            try:
                while chunk := client.recv(65536):
                    received.extend(chunk)
            except OSError as exc:
                pytest.fail(
                    f"request {index} ({path}, {length}): {exc}; received {received[:300]!r}"
                )
        head, separator, payload = bytes(received).partition(b"\r\n\r\n")
        assert separator, (index, received)
        lines = head.decode("ascii").split("\r\n")
        assert int(lines[0].split()[1]) == status, index
        fields = dict(line.lower().split(": ", 1) for line in lines[1:])
        assert fields["connection"] == "close"
        assert len(payload) == int(fields["content-length"]), index
        assert "error" in json.loads(payload), index


def test_questions_run_concurrently():
    barrier = threading.Barrier(3, timeout=5)

    class ConcurrentReader:
        def read(self, messages, letters):
            barrier.wait()
            return FakeReader().read(messages, letters)

    service = DecisionService(ConcurrentReader(), "fake", max_parallel=3)
    try:
        result = service.handle({"questions": {str(i): {"type": "noul"} for i in range(3)}})
        assert len(result["answers"]) == 3
    finally:
        service.close()


def test_stdlib_path_imports_without_torch():
    code = """
import importlib.abc
import sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers'}:
            raise AssertionError('unexpected model dependency: ' + fullname)
sys.meta_path.insert(0, BlockModels())
import ayaka.swift.server
import ayaka.swift.policy
import ayaka.swift.collect
import ayaka.swift.fit
import ayaka.swift.evaluate
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
