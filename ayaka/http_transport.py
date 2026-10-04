"""Bounded, process-local idempotency for safe inference transport retries."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import http.client
import ipaddress
import json
import os
import socket
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from threading import BoundedSemaphore, Lock

from .jev_api import ModelCatalog, ValidationError, error_body, wire_response


class RequestConflict(ValueError):
    pass


class CacheFull(RuntimeError):
    pass


class RequestCache:
    """Never evict a live key: retrying it must not spend inference twice.

    The guarantee covers this server process and the advertised retention period.
    A deployment with multiple workers needs a shared idempotency store.
    """

    def __init__(
        self,
        capacity=8192,
        retention=3600,
        max_response_bytes=2 * 1024 * 1024,
        max_bytes=64 * 1024 * 1024,
    ):
        self.capacity, self.retention = capacity, retention
        self.max_response_bytes = max_response_bytes
        self.max_bytes = max_bytes
        self.entries = {}
        self.lock = Lock()

    def execute(self, key, raw, compute):
        if not isinstance(key, str) or not 1 <= len(key) <= 128 or not key.isascii():
            raise RequestConflict("Idempotency-Key must be 1–128 ASCII characters")
        if any(ord(c) < 33 or ord(c) > 126 for c in key):
            raise RequestConflict("Idempotency-Key contains invalid characters")
        digest = hashlib.sha256(raw).hexdigest()
        with self.lock:
            now = time.monotonic()
            self.entries = {k: v for k, v in self.entries.items() if now < v[0]}
            if key in self.entries:
                _, previous, response = self.entries[key]
                if digest != previous:
                    raise RequestConflict("Idempotency-Key already belongs to a different body")
                return response
            if len(self.entries) >= self.capacity:
                raise CacheFull("idempotency cache full; retry later with the same key")
            used = sum(len(entry[2][1]) for entry in self.entries.values())
            if used + max(self.max_response_bytes, 128) > self.max_bytes:
                raise CacheFull("idempotency response memory full; retry later with the same key")
            response = compute()
            if len(response[1]) > self.max_response_bytes:
                response = (503, b'{"error":"response exceeds idempotency cache limit"}')
            self.entries[key] = (time.monotonic() + self.retention, digest, response)
            return response


def request_bytes(request, timeout=300, attempts=3):
    """Retry GET or explicitly idempotent POST, never HTTP/application errors.

    Only clients targeting an Ayaka server should supply Idempotency-Key. Server
    restarts and external servers do not inherit the process-local guarantee.
    """
    safe = request.get_method() == "GET" or request.get_header("Idempotency-key") is not None
    for attempt in range(attempts if safe else 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError:
            raise
        except (OSError, http.client.HTTPException):
            if not safe or attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (attempt + 1))


def server_for(address, handler):
    from http.server import ThreadingHTTPServer

    runtime = getattr(handler, "runtime", None)
    if runtime is not None:
        runtime.config.validate_bind(address[0])

    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET

        def server_close(self):
            super().server_close()
            runtime = getattr(handler, "runtime", None)
            if runtime is not None:
                runtime.close()

    return Server(address, handler)


class BackendOverloaded(RuntimeError):
    """The inference backend has no queue capacity."""


@dataclass(frozen=True)
class ServiceConfig:
    api_key: str | None = None
    api_key_env: str = "AYAKA_API_KEY"
    allow_no_key: bool = False
    max_inflight: int = 32
    request_timeout_s: float = 60.0
    retry_after: int = 1

    def __post_init__(self):
        import math

        if (
            self.max_inflight < 1
            or not math.isfinite(self.request_timeout_s)
            or self.request_timeout_s <= 0
        ):
            raise ValueError("max-inflight and request-timeout-s must be positive")

    @property
    def key(self):
        return self.api_key if self.api_key is not None else os.environ.get(self.api_key_env)

    def validate_bind(self, host):
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.lower() == "localhost"
        if not loopback and not self.key and not self.allow_no_key:
            raise ValueError("non-loopback serving requires an API key or --allow-no-key")


def add_service_arguments(parser):
    parser.add_argument("--api-key-env", default="AYAKA_API_KEY")
    parser.add_argument("--allow-no-key", action="store_true")
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--request-timeout-s", type=float, default=60)
    parser.add_argument("--model-id", help="actual versioned Ayaka model id returned in responses")
    parser.add_argument("--model-description", default=ModelCatalog.description)
    parser.add_argument("--model-release-date", default=ModelCatalog.release_date)


def config_from_args(args):
    return ServiceConfig(
        api_key_env=args.api_key_env,
        allow_no_key=args.allow_no_key,
        max_inflight=args.max_inflight,
        request_timeout_s=args.request_timeout_s,
    )


class Metrics:
    """Bounded-label counters. Never retain state, instructions or question IDs."""

    buckets = (0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60)

    def __init__(self):
        self.lock = Lock()
        self.requests = Counter()
        self.answers = Counter()
        self.tokens = Counter()
        self.latencies = Counter()
        self.elapsed = 0.0

    def observe(self, status, route, elapsed):
        with self.lock:
            self.requests[(status, route)] += 1
            self.elapsed += elapsed
            self.latencies["count"] += 1
            for bound in self.buckets:
                if elapsed <= bound:
                    self.latencies[bound] += 1

    def inference(self, response):
        with self.lock:
            for kind in ("input_tokens", "output_tokens"):
                self.tokens[kind] += response.get("usage", {}).get(kind, 0)
            for answer in response.get("answers", {}).values():
                kind = answer.get("type")
                if kind not in ("noul", "choice", "score"):
                    continue
                route = answer.get("ayaka", {}).get("route", "direct")
                if route not in ("direct", "reasoned", "grouped", "generated", "image"):
                    route = "other"
                self.answers[(kind, route)] += 1

    def render(self):
        with self.lock:
            lines = ["# TYPE ayaka_requests_total counter"]
            lines += [
                f'ayaka_requests_total{{status="{s}",route="{r}"}} {v}'
                for (s, r), v in sorted(self.requests.items())
            ]
            lines += ["# TYPE ayaka_answers_total counter"]
            lines += [
                f'ayaka_answers_total{{type="{t}",route="{r}",status="200"}} {v}'
                for (t, r), v in sorted(self.answers.items())
            ]
            lines += ["# TYPE ayaka_tokens_total counter"]
            lines += [
                f'ayaka_tokens_total{{type="{t}"}} {self.tokens[t]}'
                for t in ("input_tokens", "output_tokens")
            ]
            lines += ["# TYPE ayaka_request_latency_seconds histogram"]
            lines += [
                f'ayaka_request_latency_seconds_bucket{{le="{b}"}} {self.latencies[b]}'
                for b in self.buckets
            ]
            lines += [
                f'ayaka_request_latency_seconds_bucket{{le="+Inf"}} {self.latencies["count"]}',
                f"ayaka_request_latency_seconds_count {self.latencies['count']}",
                f"ayaka_request_latency_seconds_sum {self.elapsed}",
            ]
            return "\n".join(lines) + "\n"


class RequestRuntime:
    def __init__(self, config):
        self.config = config
        self.key = config.key  # Freeze credentials for this server's lifetime.
        self.slots = BoundedSemaphore(config.max_inflight)
        self.pool = ThreadPoolExecutor(max_workers=config.max_inflight)
        self.metrics = Metrics()

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def make_api_handler(
    service, *, config=None, cache=None, max_http_bytes=24 * 1024 * 1024, validation_errors=()
):
    """Shared deadline/admission/auth transport; backends keep their own inference."""
    runtime = RequestRuntime(config or ServiceConfig())
    catalog = getattr(service, "catalog", ModelCatalog(service.model_name))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.started = time.monotonic()
            self.request_id = str(uuid.uuid4())
            self.connection.settimeout(runtime.config.request_timeout_s)

        def finish(self):
            # Bounded drain preserves error responses on Windows when a rejected
            # client's body is still in flight. Always close each connection.
            try:
                self.wfile.flush()
                deadline = time.monotonic() + 0.10
                while (remaining := deadline - time.monotonic()) > 0:
                    self.connection.settimeout(remaining)
                    try:
                        chunk = self.rfile.read1(64 * 1024)
                    except OSError:
                        # A body deadline poisons BufferedReader for subsequent
                        # reads. Drain late TCP input directly in that case.
                        chunk = self.connection.recv(64 * 1024)
                    if not chunk:
                        break
            except OSError:
                pass
            finally:
                with contextlib.suppress(OSError):
                    self.connection.shutdown(socket.SHUT_WR)
                super().finish()

        def _send(self, status, body, *, content_type="application/json", retry=False):
            data = (
                body
                if isinstance(body, bytes)
                else json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.send_header("x-typesafe-request-id", self.request_id)
            if retry or status in (429, 529):
                self.send_header("retry-after", str(runtime.config.retry_after))
            self.end_headers()
            route = self.path.rstrip("/")
            route = {
                "/v1/systemone": "systemone",
                "/v1/models": "models",
                "/health": "health",
                "/v1/health": "health",
                "/metrics": "metrics",
            }.get(route, "other")
            runtime.metrics.observe(status, route, time.monotonic() - self.started)
            self.close_connection = True
            with contextlib.suppress(OSError):
                self.wfile.write(data)

        def _authorized(self):
            if not runtime.key:
                return True
            header = self.headers.get("Authorization", "")
            scheme, _, token = header.partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                token.encode(), runtime.key.encode()
            ):
                self._send(401, error_body("missing or invalid API key"))
                return False
            return True

        def do_GET(self):  # noqa: N802
            path = self.path.rstrip("/")
            if path in ("/health", "/v1/health"):
                health = {"status": "ok", "model": catalog.actual_id}
                if hasattr(service, "prompt_variant"):
                    health["prompt_variant"] = service.prompt_variant
                self._send(200, health)
            elif path == "/v1/models":
                if self._authorized():
                    self._send(200, catalog.listing())
            elif path == "/metrics":
                self._send(
                    200,
                    runtime.metrics.render().encode(),
                    content_type="text/plain; version=0.0.4; charset=utf-8",
                )
            else:
                self._send(404, error_body("not found"))

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/systemone":
                self._send(404, error_body("not found"))
                return
            if not self._authorized():
                return
            if not runtime.slots.acquire(blocking=False):
                self._send(429, error_body("maximum in-flight requests reached"))
                return
            submitted = False
            try:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise ValidationError("invalid Content-Length", "Content-Length") from exc
                if not 0 <= length <= max_http_bytes:
                    raise ValidationError(
                        f"request exceeds the {max_http_bytes // (1024 * 1024)} MiB body limit",
                        "body",
                    )
                chunks, received = [], 0
                while received < length:
                    remaining = runtime.config.request_timeout_s - (time.monotonic() - self.started)
                    if remaining <= 0:
                        raise TimeoutError("request deadline exceeded")
                    self.connection.settimeout(remaining)
                    chunk = self.rfile.read1(min(64 * 1024, length - received))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    received += len(chunk)
                raw = b"".join(chunks)
                if len(raw) != length:
                    raise ValidationError("incomplete request body", "body")
                body = json.loads(raw)
                key = self.headers.get("Idempotency-Key")

                def compute():
                    try:
                        response = service.handle(body)
                        runtime.metrics.inference(response)
                        code, out = 200, wire_response(response)
                        if "answers" in out and isinstance(body, dict):
                            missing = set(body.get("questions", {})) - set(out["answers"])
                            if missing:
                                failed = next(iter(missing))
                                code = 422
                                out.update(
                                    error_body(
                                        "candidate generation produced no answer",
                                        f"ayaka.questions.{failed}.candidate_generation",
                                    )
                                )
                    except (ValidationError, *validation_errors) as exc:
                        code, out = (
                            422,
                            getattr(
                                exc,
                                "response",
                                error_body(str(exc), getattr(exc, "field", "questions")),
                            ),
                        )
                    except BackendOverloaded:
                        code, out = 529, error_body("backend queue overloaded")
                    except urllib.error.HTTPError as exc:
                        code = 529 if exc.code in (429, 503, 529) else 502
                        out = error_body(
                            "backend queue overloaded"
                            if code == 529
                            else "backend failure: HTTPError"
                        )
                    except Exception as exc:
                        code, out = 502, error_body(f"backend failure: {type(exc).__name__}")
                    if code != 200 and "answers" not in out:
                        runtime.metrics.inference(out)
                    try:
                        data = json.dumps(out, ensure_ascii=False, allow_nan=False).encode("utf-8")
                    except (TypeError, ValueError):
                        code = 502
                        data = json.dumps(error_body("backend failure: invalid response")).encode()
                    if (
                        key is not None
                        and cache is not None
                        and len(data) > cache.max_response_bytes
                    ):
                        code, data = (
                            503,
                            json.dumps(
                                {
                                    "error": "response exceeds idempotency cache limit",
                                    "usage": out.get("usage", {}),
                                }
                            ).encode(),
                        )
                    return code, data

                def work():
                    return (
                        cache.execute(key, raw, compute)
                        if key is not None and cache is not None
                        else compute()
                    )

                remaining = runtime.config.request_timeout_s - (time.monotonic() - self.started)
                if remaining <= 0:
                    raise TimeoutError("request deadline exceeded")
                future = runtime.pool.submit(work)
                submitted = True
                future.add_done_callback(lambda _: runtime.slots.release())
                try:
                    status, data = future.result(timeout=remaining)
                except TimeoutError:
                    future.cancel()
                    self._send(504, error_body("request deadline exceeded"))
                else:
                    self._send(status, data)
            except TimeoutError:
                self._send(504, error_body("request deadline exceeded"))
            except RequestConflict as exc:
                self._send(409, error_body(str(exc)))
            except CacheFull as exc:
                self._send(529, error_body(str(exc)))
            except (ValidationError, *validation_errors) as exc:
                self._send(422, error_body(str(exc), getattr(exc, "field", "body")))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send(422, error_body(str(exc), "body"))
            finally:
                if not submitted:
                    runtime.slots.release()

        def log_message(self, *args):
            pass

    Handler.runtime = runtime
    return Handler
