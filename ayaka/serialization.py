"""Typed structured-state serialization (docs.md section 10, addendum A2).

A JSON-like state is serialized into a typed token stream instead of being
flattened to plain text, so that primitive types stay distinguishable at the
token level::

    {"user": {"age": 31, "verified": True}, "tags": ["premium"]}
    ->
    <obj> <key> user <obj> <key> age <num> 31 <key> verified <bool_true>
    <end_obj> <key> tags <array> <str> premium <end_array> <end_obj>

Invariants:

- ``42`` (number) != ``"42"`` (string) != ``true`` at the token level.
- Object keys are emitted in canonical (sorted) order; array order is
  preserved.
- Canonical serialization doubles as the state cache key input
  (docs.md section 21.3).
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata

JsonValue = dict | list | str | int | float | bool | None
State = JsonValue | str

SERIALIZATION_VERSION = "v1"


def _number_to_text(value: int | float) -> str:
    if isinstance(value, bool):  # bool is an int subclass; guarded by caller
        raise TypeError("bool is not a number here")
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("non-finite numbers are not representable in state")
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return str(value)


def serialize_typed(value: JsonValue) -> str:
    """Serialize a JSON value into the typed token stream."""
    if isinstance(value, dict):
        parts = ["<obj>"]
        for key in sorted(value.keys(), key=lambda k: str(k)):
            parts.append("<key>")
            parts.append(str(key))
            parts.append(serialize_typed(value[key]))
        parts.append("<end_obj>")
        return " ".join(parts)
    if isinstance(value, (list, tuple)):
        parts = ["<array>"]
        parts.extend(serialize_typed(item) for item in value)
        parts.append("<end_array>")
        return " ".join(parts)
    if isinstance(value, bool):
        return "<bool_true>" if value else "<bool_false>"
    if value is None:
        return "<null>"
    if isinstance(value, (int, float)):
        return f"<num> {_number_to_text(value)}"
    if isinstance(value, str):
        return f"<str> {value}"
    raise TypeError(f"unsupported state value type: {type(value)!r}")


def serialize_state(state: State) -> str:
    """Serialize a full state.

    Structured (dict/list/scalar) values get the typed treatment; a plain
    string state is wrapped in the text-state markers.
    """
    if isinstance(state, str):
        return f"<doc> {state} </doc>"
    return f"<state> {serialize_typed(state)} </state>"


def canonical_state(state: State) -> str:
    """Canonical serialization used for cache keys and dedup.

    For string states the text is normalized; for structured states the
    value is round-tripped through the typed serializer (which already
    sorts object keys and normalizes number formatting).
    """
    if isinstance(state, str):
        return f"<doc> {normalize_text(state)} </doc>"
    return serialize_state(_canonical_json(state))


def _canonical_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        pairs = [
            (normalize_text(str(k)), _canonical_json(v))
            for k, v in value.items()
            if normalize_text(str(k))
        ]
        pairs.sort(key=lambda kv: (kv[0], json.dumps(kv[1], sort_keys=True, default=str)))
        return dict(pairs)
    if isinstance(value, (list, tuple)):
        return [_canonical_json(v) for v in value]
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def normalize_text(text: str) -> str:
    """Whitespace/case normalization for dedup keys."""
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.casefold().split())


def canonical_json_dumps(state: JsonValue) -> str:
    """Stable JSON dump of the canonicalized value (for manifests)."""
    return json.dumps(
        _canonical_json(state), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def state_cache_key(
    state: State,
    *,
    model_version: str,
    tokenizer_version: str,
    serialization_version: str = SERIALIZATION_VERSION,
) -> str:
    """Cross-request state cache key (docs.md section 21.3)."""
    material = "\x1f".join(
        [model_version, tokenizer_version, serialization_version, canonical_state(state)]
    )
    return _sha256(material)


def dedup_key(state: State, instruction: str, candidates: list[str]) -> str:
    """Cross-dataset dedup key (docs.md section 32 / 43.5).

    Candidate order is canonicalized away — options form an unordered
    semantic set, so ``[A, B]`` and ``[B, A]`` dedup to the same key.
    """
    material = "\x1f".join(
        [
            canonical_state(state),
            normalize_text(instruction),
            "\x1e".join(sorted(normalize_text(c) for c in candidates)),
        ]
    )
    return _sha256(material)
