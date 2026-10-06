"""Verified checkpoint snapshots and untruncated v1 context checks, offline first."""

from __future__ import annotations

import hashlib
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from ..config import ElectraConfig
from ..evidence import reasoning_messages
from ..evidence_generation import chat_ids
from ..evidence_pipeline import decision_request
from ..prompt import render_prefix, render_question
from .matched_contract import (
    BACKBONE_REVISION,
    CHECKPOINT_REPO,
    CHECKPOINT_REVISION,
    fingerprint,
    validate_protocol,
)

CONTEXT_VERSION = "ayaka-checked-v1-context-1"


def consumed_files(root):
    """Pin presence as well as bytes: adding a preferred file cannot change loading."""
    root = Path(root)
    files = {
        name
        for name in ("ayaka_config.json", "electra_config.json", "meta.json")
        if (root / name).is_file()
    }
    if (
        not files & {"ayaka_config.json", "electra_config.json"}
        or not (root / "head.safetensors").is_file()
    ):
        raise ValueError("checked v1 requires config and safetensors head, no fallback")
    files.add("head.safetensors")
    adapter = root / "adapter"
    if not adapter.is_dir() or adapter.is_symlink():
        raise ValueError("checked v1 requires a real unmerged adapter directory")
    files.update(p.relative_to(root).as_posix() for p in adapter.rglob("*") if p.is_file())
    if not {"adapter/adapter_config.json", "adapter/adapter_model.safetensors"} <= files:
        raise ValueError("checked v1 requires adapter config and safetensors weights")
    if any(name.endswith((".bin", ".pt")) for name in files):
        raise ValueError("checked v1 refuses alternate pickle adapter weights")
    return files


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint_files(root, hashes):
    if consumed_files(root) != set(hashes):
        raise ValueError("checkpoint consumed-file inventory differs from external pins")
    for name, expected in hashes.items():
        parsed = PurePosixPath(name)
        if parsed.is_absolute() or ".." in parsed.parts or "\\" in name or ":" in name:
            raise ValueError("checkpoint pins require safe relative POSIX paths")
        if _file_hash(Path(root) / name) != expected:
            raise ValueError(f"{name}: actual checkpoint bytes differ from external pins")


def checkpoint_config(root):
    root = Path(root)
    configs = [
        json.loads((root / name).read_bytes())
        for name in ("ayaka_config.json", "electra_config.json")
        if (root / name).is_file()
    ]
    if any(fingerprint(c) != fingerprint(configs[0]) for c in configs):
        raise ValueError("preferred and legacy checkpoint config disagree")
    c = configs[0]
    if (
        c.get("backbone") != "google/gemma-4-12B-it"
        or c.get("backbone_revision") != BACKBONE_REVISION
        or c.get("readout", "hybrid") != "hybrid"
        or c.get("version", 1) != 1
        or c.get("lora_r", 64) != 64
    ):
        raise ValueError("checkpoint is not the pinned published v1 hybrid config")
    adapter = json.loads((root / "adapter/adapter_config.json").read_bytes())
    if (
        adapter.get("base_model_name_or_path") != c["backbone"]
        or adapter.get("revision") not in (None, BACKBONE_REVISION)
        or adapter.get("r") != 64
    ):
        raise ValueError("checkpoint adapter differs from the pinned v1 recipe")
    cfg = ElectraConfig(**{**c, "lora_targets": tuple(c.get("lora_targets", ()))})
    from ..input_contract import read_contract

    if read_contract(str(root), cfg) is not None:
        raise ValueError("published v1 must retain its legacy segmented input contract")
    return cfg


@contextmanager
def checkpoint_snapshot(receipt, protocol):
    """Copy while hashing; only verified copied files can be passed to a loader.

    HF snapshot file symlinks are allowed: their resolved bytes are copied, and
    the resulting temporary checkpoint contains no symlinks or hard links.
    """
    protocol = validate_protocol(protocol)
    hashes = protocol["checkpoint_source_sha256"]
    if (
        receipt.get("checkpoint_repo") != CHECKPOINT_REPO
        or receipt.get("checkpoint_revision") != CHECKPOINT_REVISION
        or receipt.get("base_revision") != BACKBONE_REVISION
        or receipt.get("source_sha256") != hashes
    ):
        raise ValueError("checkpoint receipt differs from externally pinned inventory")
    original = Path(receipt["checkpoint_path"])
    verify_checkpoint_files(original, hashes)
    with tempfile.TemporaryDirectory(prefix="ayaka-checked-v1-") as temp:
        snapshot = Path(temp)
        for name, expected in sorted(hashes.items()):
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with (original / name).open("rb") as source, target.open("xb") as dest:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    h.update(chunk)
                    dest.write(chunk)
            if h.hexdigest() != expected:
                raise ValueError("checkpoint changed while copying its consumed bytes")
        verify_checkpoint_files(original, hashes)
        verify_checkpoint_files(snapshot, hashes)
        cfg = checkpoint_config(snapshot)
        yield snapshot, cfg
        # No successful execution receipt may describe a mutated snapshot/source.
        verify_checkpoint_files(snapshot, hashes)
        verify_checkpoint_files(original, hashes)


def full_decision_context(state, specs, tok, *, limit=8192, max_labels=26):
    prefix = render_prefix(state, tok)  # deliberately no truncation parameter
    suffixes = [render_question(spec.view(), tok, max_labels).suffix_ids for spec in specs]
    counts = [len(prefix) + len(suffix) for suffix in suffixes]
    if not counts or any(n > limit for n in counts):
        raise ValueError("full v1 decision context exceeds limit; refuse truncation")
    return {
        "state_sha256": fingerprint(state),
        "questions_sha256": fingerprint([spec.view().__dict__ for spec in specs]),
        "input_tokens": counts,
        "input_token_ids_sha256": [fingerprint(prefix + suffix) for suffix in suffixes],
        "context_limit": limit,
    }


def context_preflight(items, tok, cfg):
    from scripts.swift.v1_on_runner import jevbench_record

    from .jevbench import record_to_item

    rows = {}
    for item in items:
        bench = record_to_item(jevbench_record(item))
        direct = full_decision_context(
            bench.state, [bench.spec], tok, max_labels=cfg.max_label_candidates
        )
        extraction = len(
            chat_ids(tok, reasoning_messages(decision_request(bench.state, bench.spec)))
        )
        if extraction + 384 > 12288:
            raise ValueError("full v1 extraction context cannot reserve frozen 384-token budget")
        rows[item.id] = {
            "direct": direct,
            "extraction_input_tokens": extraction,
            "extraction_context_limit": 12288,
            "extraction_reserved_tokens": 384,
        }
    return rows


class FullContextDecision:
    """Check original AND actual worked-step readouts before each model forward."""

    def __init__(self, inner):
        self.inner = inner
        self.contexts = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def decide(self, state, specs, **kwargs):
        value = full_decision_context(
            state,
            specs,
            self.inner.tok,
            limit=self.inner.max_seq_len,
            max_labels=self.inner.max_labels,
        )
        self.contexts.append(value)
        return self.inner.decide(state, specs, **kwargs)


def validate_context_rows(rows, preflight=None):
    for row in rows:
        c = row.get("checked_context") or {}
        values = c.get("decisions")
        if c.get("version") != CONTEXT_VERSION or not isinstance(values, list) or not values:
            raise ValueError("v1 resume/scoring requires checked full-context execution rows")
        if c.get("sha256") != fingerprint({k: v for k, v in c.items() if k != "sha256"}):
            raise ValueError("v1 context receipt fingerprint mismatch")
        if set(c) != {"version", "decisions", "extraction_input_tokens", "sha256"}:
            raise ValueError("v1 context receipt has unknown fields")
        from .jevbench import record_to_item

        b, q = row["binding"], row["binding"]["question"]
        bench = record_to_item(
            {
                "id": row["id"],
                "state": b["state"],
                "labels": q["labels"],
                "expected": b["gold"]
                if isinstance(b["gold"], str)
                else max(b["gold"], key=b["gold"].get),
                "question": {
                    "type": q["type"],
                    "instructions": q["instruction"],
                    "criteria": dict(zip(q["labels"], q["descriptions"], strict=True)),
                },
            }
        )
        question_sha = fingerprint([bench.spec.view().__dict__])
        if values[0].get("state_sha256") != fingerprint(row["binding"]["state"]):
            raise ValueError("v1 baseline context differs from bound input")
        for value in values:
            counts = value.get("input_tokens")
            hashes = value.get("input_token_ids_sha256")
            if (
                set(value)
                != {
                    "state_sha256",
                    "questions_sha256",
                    "input_tokens",
                    "input_token_ids_sha256",
                    "context_limit",
                }
                or value.get("questions_sha256") != question_sha
                or not isinstance(hashes, list)
                or len(hashes) != 1
            ):
                raise ValueError("v1 context receipt differs from full question coordinates")
            from .matched_contract import _sha

            _sha(value.get("state_sha256"))
            _sha(hashes[0])
            if (
                value.get("context_limit") != 8192
                or not isinstance(counts, list)
                or len(counts) != 1
                or any(type(n) is not int or not 0 < n <= 8192 for n in counts)
            ):
                raise ValueError("v1 context receipt reports clipped/invalid input")
        if (
            type(c.get("extraction_input_tokens")) is not int
            or not 0 < c["extraction_input_tokens"] + 384 <= 12288
        ):
            raise ValueError("v1 extraction context receipt violates frozen reserve")
        if preflight is not None:
            expected = preflight.get(row["id"])
            if (
                expected is None
                or fingerprint(values[0]) != fingerprint(expected["direct"])
                or c["extraction_input_tokens"] != expected["extraction_input_tokens"]
            ):
                raise ValueError(
                    "v1 resume/scoring context differs from fresh full-input preflight"
                )
