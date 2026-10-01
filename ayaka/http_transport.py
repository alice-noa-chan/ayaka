"""Bounded, process-local idempotency for safe inference transport retries."""

from __future__ import annotations

import hashlib
import http.client
import socket
import time
import urllib.error
import urllib.request
from threading import Lock


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

    class IPv6Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6

    cls = IPv6Server if ":" in address[0] else ThreadingHTTPServer
    return cls(address, handler)
