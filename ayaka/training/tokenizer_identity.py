"""Bounded serialization reuse during synchronous, in-memory CPU preparation.

Serialize a native tokenizer once on entry and once on successful exit. This
scope must end before prepared artifacts are written or optimizer work starts.
It is not a persistent cache or an attestation against concurrent mutation.
"""

from __future__ import annotations

import hashlib
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

from ..eval.read_artifact import fingerprint

_SCOPES = ContextVar("ayaka_preparation_tokenizer_scopes", default=())


def _native(tokenizer):
    return getattr(tokenizer, "hf", tokenizer)


def configuration_fingerprint(tokenizer):
    """Template and decode settings absent from backend.to_str(), including EOS."""
    native = _native(tokenizer)
    return fingerprint(
        {
            "is_fast": getattr(native, "is_fast", False),
            "chat_template": getattr(native, "chat_template", None),
            "special_tokens": getattr(native, "special_tokens_map", None),
            "bos": getattr(native, "bos_token_id", None),
            "pad": getattr(native, "pad_token_id", None),
            "eos": getattr(native, "eos_token_id", None),
            "padding_side": getattr(native, "padding_side", None),
            "truncation_side": getattr(native, "truncation_side", None),
            "clean_up_tokenization_spaces": getattr(native, "clean_up_tokenization_spaces", None),
            "split_special_tokens": getattr(native, "split_special_tokens", None),
            "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output": getattr(
                native,
                "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output",
                None,
            ),
        }
    )


def _serialize(backend):
    text = backend.to_str()
    return {
        "raw_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "json_string_sha256": fingerprint(text),
    }


@dataclass
class _Snapshot:
    native: object
    backend: object
    configuration_sha256: str
    fingerprints: dict
    active: bool = True

    def check(self, *, serialized=False):
        if (
            not self.active
            or getattr(self.native, "backend_tokenizer", None) is not self.backend
            or configuration_fingerprint(self.native) != self.configuration_sha256
            or serialized
            and _serialize(self.backend) != self.fingerprints
        ):
            raise ValueError("tokenizer changed during scoped preparation; discard prepared rows")


def backend_fingerprints(tokenizer):
    """Retain both existing hash conventions without changing artifact bytes."""
    native = _native(tokenizer)
    backend = getattr(native, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("exact fast-tokenizer serialization is required")
    for snapshot in reversed(_SCOPES.get()):
        if snapshot.active and snapshot.native is native:
            snapshot.check()
            return dict(snapshot.fingerprints)
    return _serialize(backend)


@contextmanager
def tokenizer_identity_scope(tokenizer):
    """Reuse only this object's serialization; discard results on mutation.

    Nested same-tokenizer scopes share the outer validation. A different
    tokenizer gets a separate snapshot. Nothing survives scope exit; ordinary
    calls continue to serialize the current tokenizer on every invocation.
    """
    native = _native(tokenizer)
    backend = getattr(native, "backend_tokenizer", None)
    if backend is None:  # The existing segmented ToyTokenizer needs no serialization.
        yield
        return
    for existing in reversed(_SCOPES.get()):
        if existing.active and existing.native is native:
            existing.check()
            yield
            return
    snapshot = _Snapshot(native, backend, configuration_fingerprint(native), _serialize(backend))
    token = _SCOPES.set((*_SCOPES.get(), snapshot))
    try:
        yield
        snapshot.check(serialized=True)
    finally:
        snapshot.active = False
        _SCOPES.reset(token)


def scoped_tokenizer_preparation(function):
    """For pure preparation functions taking a `tok` or `tokenizer` argument."""
    signature = inspect.signature(function)
    name = "tok" if "tok" in signature.parameters else "tokenizer"
    if name not in signature.parameters:
        raise ValueError("scoped preparation requires a tokenizer argument")

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        with tokenizer_identity_scope(bound.arguments[name]):
            return function(*args, **kwargs)

    return wrapped
