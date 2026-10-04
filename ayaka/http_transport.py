"""Jev HTTP admission and authentication, independent of v1 inference."""

from __future__ import annotations

import contextlib
import hmac
import json
import os
import socket
import time
import urllib.error
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from threading import BoundedSemaphore

from .jev_api import ModelCatalog, ValidationError, error_body, wire_response


class BackendOverloaded(RuntimeError):
    """The inference backend has no capacity; callers may retry later."""


@dataclass(frozen=True)
class ServiceConfig:
    api_key: str | None = None
    api_key_env: str = "AYAKA_API_KEY"
    max_inflight: int = 32
    retry_after: int = 1

    def __post_init__(self):
        if self.max_inflight < 1 or self.retry_after < 1:
            raise ValueError("max-inflight and retry-after must be positive")

    @property
    def key(self):
        return self.api_key if self.api_key is not None else os.environ.get(self.api_key_env)


def add_service_arguments(parser):
    parser.add_argument(
        "--api-key-env", default="AYAKA_API_KEY", help="Bearer API key environment variable"
    )
    parser.add_argument(
        "--max-inflight", type=int, default=32, help="maximum active/queued HTTP decisions"
    )
    parser.add_argument("--model-id", help="versioned response id (default: <served-name>-1.0.0)")
    parser.add_argument("--model-description", default=ModelCatalog.description)
    parser.add_argument("--model-release-date", default=ModelCatalog.release_date)


def config_from_args(args):
    return ServiceConfig(api_key_env=args.api_key_env, max_inflight=args.max_inflight)


def make_api_handler(service, *, config=None, max_http_bytes=24 * 1024 * 1024):
    config = config or ServiceConfig()
    key = config.key  # Freeze credentials for this server's lifetime.
    slots = BoundedSemaphore(config.max_inflight)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def finish(self):
            # Bounded drain preserves responses on Windows when a rejected
            # client's request body is still arriving. Close every connection.
            try:
                self.wfile.flush()
                deadline = time.monotonic() + 0.10
                while (remaining := deadline - time.monotonic()) > 0:
                    self.connection.settimeout(remaining)
                    if not self.rfile.read1(64 * 1024):
                        break
            except OSError:
                pass
            finally:
                with contextlib.suppress(OSError):
                    self.connection.shutdown(socket.SHUT_WR)
                super().finish()

        def _send(self, status, body):
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.send_header("x-typesafe-request-id", str(uuid.uuid4()))
            if status in (429, 529):
                self.send_header("retry-after", str(config.retry_after))
            self.end_headers()
            self.close_connection = True
            with contextlib.suppress(OSError):
                self.wfile.write(data)

        def _authorized(self):
            if not key:
                return True
            scheme, _, token = self.headers.get("Authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode(), key.encode()):
                self._send(401, error_body("missing or invalid API key"))
                return False
            return True

        def do_GET(self):  # noqa: N802
            path = self.path.rstrip("/")
            if path in ("/health", "/v1/health"):
                self._send(200, {"status": "ok", "model": service.catalog.actual_id})
            elif path == "/v1/models":
                if self._authorized():
                    self._send(200, service.catalog.listing())
            else:
                self._send(404, error_body("not found"))

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/systemone":
                self._send(404, error_body("not found"))
                return
            if not self._authorized():
                return
            if not slots.acquire(blocking=False):
                self._send(429, error_body("maximum in-flight requests reached"))
                return
            try:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise ValidationError("invalid Content-Length", "Content-Length") from exc
                if not 0 <= length <= max_http_bytes:
                    raise ValidationError("request body exceeds the 24 MiB limit", "body")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValidationError("incomplete request body", "body")
                body = json.loads(raw or b"{}")
                started = time.perf_counter()
                out = service.handle(body)
                out["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                self._send(200, wire_response(out))
            except ValidationError as exc:
                self._send(422, error_body(str(exc), exc.field))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send(422, error_body(str(exc), "body"))
            except BackendOverloaded:
                self._send(529, error_body("backend queue overloaded"))
            except urllib.error.HTTPError as exc:
                status = 529 if exc.code in (429, 503, 529) else 502
                self._send(
                    status,
                    error_body("backend queue overloaded" if status == 529 else "backend failure"),
                )
            except Exception as exc:  # never leak backend details to callers
                self._send(500, error_body(f"internal error: {type(exc).__name__}"))
            finally:
                slots.release()

        def log_message(self, *args):
            pass

    return Handler
