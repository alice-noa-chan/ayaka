import concurrent.futures
import json
import urllib.error
import urllib.request

import pytest

from ayaka.http_transport import CacheFull, RequestCache, RequestConflict, request_bytes, server_for
from ayaka.serve import make_handler


def test_concurrent_replay_is_byte_identical_and_computes_once():
    cache, calls = RequestCache(), []

    def compute():
        calls.append(1)
        return 200, b'{"usage":{"reasoning_tokens":1024}}'

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        responses = list(pool.map(lambda _: cache.execute("key", b"body", compute), range(8)))
    assert len(calls) == 1 and all(r == responses[0] for r in responses)
    with pytest.raises(RequestConflict):
        cache.execute("key", b"other", compute)


def test_cache_does_not_evict_live_inference_and_large_results_are_not_recomputed():
    cache = RequestCache(capacity=1, max_response_bytes=2)
    assert cache.execute("a", b"body", lambda: (200, b"long"))[0] == 503
    assert cache.execute("a", b"body", lambda: pytest.fail("recomputed"))[0] == 503
    with pytest.raises(CacheFull):
        cache.execute("b", b"body", lambda: pytest.fail("computed"))


def test_benchmark_sized_replay_cache_is_bounded_by_total_response_memory():
    cache, calls = RequestCache(), []
    for i in range(300):

        def compute(i=i):
            calls.append(i)
            return 200, b"response" * 100

        cache.execute(str(i), str(i).encode(), compute)
    assert cache.execute("0", b"0", lambda: pytest.fail("evicted first benchmark result"))[0] == 200
    assert len(calls) == 300
    small = RequestCache(capacity=10, max_response_bytes=200, max_bytes=400)
    small.execute("a", b"a", lambda: (200, b"a" * 150))
    small.execute("b", b"b", lambda: (200, b"b" * 150))
    with pytest.raises(CacheFull, match="memory"):
        small.execute("c", b"c", lambda: pytest.fail("inference ran without replay capacity"))


def test_post_reset_after_inference_replays_without_generation(monkeypatch):
    cache, calls = RequestCache(), []

    def open_request(req, timeout):
        result = cache.execute(req.get_header("Idempotency-key"), req.data, compute)
        if len(calls) == 1 and not getattr(open_request, "failed", False):
            open_request.failed = True
            raise ConnectionResetError("response lost after inference")

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return result[1]

        return Response()

    def compute():
        calls.append(1)
        return 200, b'{"reasoning_tokens":8}'

    monkeypatch.setattr(urllib.request, "urlopen", open_request)
    request = urllib.request.Request("http://ayaka", data=b"body", headers={"Idempotency-Key": "a"})
    assert json.loads(request_bytes(request))["reasoning_tokens"] == 8
    assert len(calls) == 1

    def http_error(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("http://ayaka", 400, "invalid", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", http_error)
    with pytest.raises(urllib.error.HTTPError):
        request_bytes(
            urllib.request.Request("http://ayaka", data=b"body", headers={"Idempotency-Key": "a"})
        )
    assert len(calls) == 2


def test_unkeyed_post_and_http_errors_are_not_retried(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise ConnectionResetError()

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(ConnectionResetError):
        request_bytes(urllib.request.Request("http://ayaka", data=b"body"))
    assert len(calls) == 1


def test_native_ipv6_health_and_key_conflict():
    import threading

    class Service:
        model_name = "diagnostic"
        calls = 0

        def handle(self, body):
            self.calls += 1
            return {"count": self.calls}

    service = Service()
    server = server_for(("::1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://[::1]:{server.server_address[1]}/v1/systemone"
    try:
        req = urllib.request.Request(url, data=b"{}", headers={"Idempotency-Key": "one"})
        assert request_bytes(req) == request_bytes(req)
        assert service.calls == 1
        req.data = b'{"different":true}'
        with pytest.raises(urllib.error.HTTPError) as error:
            request_bytes(req)
        assert error.value.code == 409 and service.calls == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_oversized_keyed_response_preserves_spent_usage_on_every_replay(monkeypatch):
    import threading

    from ayaka import serve

    monkeypatch.setattr(serve, "RequestCache", lambda: RequestCache(max_response_bytes=200))

    class Service:
        model_name = "diagnostic"
        calls = 0

        def handle(self, body):
            self.calls += 1
            return {"large": "x" * 400, "usage": {"reasoning_tokens": 1024}}

    service = Service()
    server = server_for(("::1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://[::1]:{server.server_address[1]}/v1/systemone"
    responses = []
    try:
        request = urllib.request.Request(url, data=b"{}", headers={"Idempotency-Key": "large"})
        for _ in range(2):
            with pytest.raises(urllib.error.HTTPError) as error:
                request_bytes(request)
            assert error.value.code == 503
            responses.append(error.value.read())
        assert responses[0] == responses[1] and service.calls == 1
        assert json.loads(responses[0])["usage"]["reasoning_tokens"] == 1024
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
