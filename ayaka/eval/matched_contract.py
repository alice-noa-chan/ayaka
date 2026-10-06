"""Fail-closed inputs for the exposed v1-on / Swift-off comparison.

Pins must be declared in a separate protocol before collecting v1 rows. Neither
the protocol nor these checks attest fresh data, checkpoint origin or quality.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from ..evidence_pipeline import FROZEN_REASONING_POLICY
from ..swift.binding import validate_bound_reads
from ..swift.policy import Policy
from ..swift.prompt import render_question
from .read_artifact import fingerprint

VERSION = "ayaka-matched-comparison-1"
SCOPE = "matched comparison on an exposed cohort; not a fresh independent confirmation"
CHECKPOINT_REPO = "alice-noa-chan/ayaka-large"
CHECKPOINT_REVISION = "267605ee22f2b5f934d81e5fbee691952d2e6f55"
BACKBONE_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
COHORT = {
    "procedural": "abf80932e93b1236296af3e79c399b6ce1b517bbe3ffe7f4eda6bbb31db4c92d",
    "hard_calibration": "01f17bc9a454e69f771f4afb0d3d73cf415b09609f6dded36517da36b261b58c",
    "hard_dev": "8a237769ee38d51adfa89bf6c6fedb5d49e966196497f165dbf5ca2072c3f7ee",
}
SWIFT_IMPLEMENTATION = {
    "readers.py": "0234f36ea1c1787de1facac32102c9d75b5210890efa98fa90d4e493320f7926",
    "prompt.py": "2fa3444e430982097843ebe636f3cddac11d5225fd0667a6bf10f39e71b8a995",
    "grouping.py": "7fde0ebacc94d494690d7dfae998f613f0303bbc288f49a7687746a008e57b44",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _sha(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("expected a lowercase SHA-256 anchor")
    return value


def _hashes(values):
    if not isinstance(values, dict) or not values:
        raise ValueError("protocol needs a nonempty external file hash inventory")
    for name, value in values.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("hash inventory names must be nonempty strings")
        _sha(value)
    return copy.deepcopy(values)


def v1_run(checkpoint_hashes):
    return {
        "runner": "v1-on-runner-2",
        "checkpoint_repo": CHECKPOINT_REPO,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_files_sha256": fingerprint(_hashes(checkpoint_hashes)),
        "backbone_revision": BACKBONE_REVISION,
        "max_seq_len": 8192,
        "max_new_tokens": 384,
        "reasoner_adapter": "off",
        "policy": asdict(FROZEN_REASONING_POLICY),
    }


def make_protocol(*, checkpoint_hashes, policy_sha256, fit_input_hashes, fit_source_hashes):
    """Construct a declaration; callers pin its bytes BEFORE observing v1 results."""
    return {
        "version": VERSION,
        "scope": SCOPE,
        "cohort_sha256": dict(COHORT),
        "checkpoint_source_sha256": _hashes(checkpoint_hashes),
        "v1_run": v1_run(checkpoint_hashes),
        "policy_sha256": _sha(policy_sha256),
        "policy_fit_inputs_sha256": _hashes(fit_input_hashes),
        "policy_fit_source_sha256": _hashes(fit_source_hashes),
    }


def validate_protocol(protocol):
    if not isinstance(protocol, dict):
        raise ValueError("comparison protocol must be an object")
    try:
        expected = make_protocol(
            checkpoint_hashes=protocol["checkpoint_source_sha256"],
            policy_sha256=protocol["policy_sha256"],
            fit_input_hashes=protocol["policy_fit_inputs_sha256"],
            fit_source_hashes=protocol["policy_fit_source_sha256"],
        )
    except KeyError as exc:
        raise ValueError("incomplete comparison protocol") from exc
    if fingerprint(protocol) != fingerprint(expected):
        raise ValueError("comparison protocol differs from the frozen recipe")
    return expected


def pinned_bytes(path, expected_sha256):
    """Hash and consume one read; never reopen a path between verification and parse."""
    data = Path(path).read_bytes()
    if digest(data) != _sha(expected_sha256):
        raise ValueError(f"{Path(path).name}: bytes differ from the external anchor")
    return data


def load_protocol(path, expected_sha256):
    return validate_protocol(json.loads(pinned_bytes(path, expected_sha256)))


def policy_from_bytes(data, protocol):
    protocol = validate_protocol(protocol)
    if digest(data) != protocol["policy_sha256"]:
        raise ValueError("policy bytes differ from the external anchor")
    values = json.loads(data)
    if "commit_margin" not in values and values.get("noul_commit") is True:
        values["commit_margin"] = 0.0
    policy = Policy(**values)
    if (
        policy.prompt_variant != "min"
        or policy.promotable is not True
        or policy.fitted_on == "unfitted"
    ):
        raise ValueError("comparison requires the pinned fitted production min policy")
    return replace(policy, reasoning_route=None)


def load_policy(path, protocol):
    return policy_from_bytes(Path(path).read_bytes(), protocol)


def item_binding(item):
    return {
        "state": item.state,
        "question": asdict(item.question),
        "gold": item.gold,
        "gold_distribution": item.gold_distribution,
        "source": item.source,
        "tier": item.tier,
        "public": item.public,
        "case_id": item.case_id or item.cluster_id or item.id,
        "cluster_id": item.cluster_id or item.case_id or item.id,
    }


def _probabilities(row, key, labels):
    probs = row.get(key)
    if (
        not isinstance(probs, dict)
        or list(probs) != labels
        or any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in probs.values())
        or not math.isclose(math.fsum(probs.values()), 1.0, abs_tol=1e-6, rel_tol=0)
    ):
        raise ValueError(f"{row.get('id')}: invalid ordered {key}")


def validate_rows(rows, items, protocol, *, system, complete=True):
    """Tie every scored/provenance field to canonical input, not just nested hashes."""
    protocol = validate_protocol(protocol)
    rows, items = copy.deepcopy(list(rows)), list(items)
    if system not in {"v1", "v2"}:
        raise ValueError("unknown comparison system")
    expected = {}
    for item in items:
        if item.id in expected or item.public is not False:
            raise ValueError("cohort contains duplicate or public inputs")
        expected[item.id] = item
    ids = [row.get("id") for row in rows]
    if (
        len(ids) != len(set(ids))
        or set(ids) - set(expected)
        or (complete and set(ids) != set(expected))
    ):
        raise ValueError("rows must cover the frozen cohort exactly once")
    if system == "v2":
        validate_bound_reads(rows)
    for row in rows:
        item = expected[row["id"]]
        want = item_binding(item)
        top = {
            k: want[k]
            for k in (
                "gold",
                "gold_distribution",
                "source",
                "tier",
                "public",
                "case_id",
                "cluster_id",
            )
        }
        top.update(type=item.question.type, labels=list(item.question.labels))
        if any(fingerprint(row.get(k)) != fingerprint(v) for k, v in top.items()):
            raise ValueError(f"{item.id}: scored fields differ from canonical input")
        binding = row.get("binding") or {}
        if any(fingerprint(binding.get(k)) != fingerprint(v) for k, v in want.items()):
            raise ValueError(f"{item.id}: nested binding differs from canonical input")
        _probabilities(row, "raw_probs", top["labels"])
        if system == "v1":
            if (
                fingerprint(row.get("run")) != fingerprint(protocol["v1_run"])
                or row.get("readout") != "v1_native_route"
            ):
                raise ValueError(f"{item.id}: v1 run differs from exact frozen policy/checkpoint")
            if fingerprint(binding) != fingerprint(want):
                raise ValueError(f"{item.id}: extra or altered v1 binding")
            _probabilities(row, "baseline_probs", top["labels"])
            if row.get("candidate_log_masses") is not None:
                raise ValueError(
                    "v1 scorer must consume recorded probabilities, not alternate masses"
                )
            continue
        runtime = binding["runtime"]
        if (
            row.get("prompt_variant") != "min"
            or row.get("model") != "google/gemma-4-12B-it"
            or row.get("revision") != BACKBONE_REVISION
            or row.get("tokenizer_revision") != BACKBONE_REVISION
            or type(row.get("passes")) is not int
            or row["passes"] != 1
            or row.get("diagnostic") is not False
            or any(k in row for k in ("reasoned_read", "trace", "generated_tokens"))
            or runtime.get("adapter_sha256") is not None
            or runtime.get("implementation_sha256") != SWIFT_IMPLEMENTATION
            or runtime.get("chat_template_kwargs", {}).get("enable_thinking") is not False
        ):
            raise ValueError(f"{item.id}: not the frozen Swift min direct recipe")
        messages, _ = render_question(
            item.state, item.question, state_format=runtime["state_format"], prompt_variant="min"
        )
        if binding["messages"] != messages:
            raise ValueError("native messages differ from canonical question and state")
        gathered = row["pass_bindings"][0]
        letters = [chr(65 + i) for i in range(len(top["labels"]))]
        if (
            list(binding["token_inputs"][0]["canonical_token_ids"]) != letters
            or list(gathered["canonical_token_ids"]) != letters
            or gathered.get("letters") != letters
            or gathered.get("labels") != top["labels"]
        ):
            raise ValueError("canonical token/display order differs from candidate coordinates")
        payload = {k: v for k, v in gathered.items() if k != "binding_sha256"}
        if gathered.get("binding_sha256") != fingerprint(
            {"parent_binding_sha256": binding["binding_sha256"], **payload}
        ):
            raise ValueError("gathered pass fingerprint differs")
        if gathered.get("messages") != messages:
            raise ValueError("gathered pass messages differ")
        if (
            type(row.get("input_tokens")) is not int
            or not 0 < row["input_tokens"] < 16384
            or row["input_tokens"] != len(binding["token_inputs"][0]["input_token_ids"])
        ):
            raise ValueError("native input token count differs")
        if type(row.get("output_tokens")) is not int or row["output_tokens"] != 1:
            raise ValueError("canonical letter read must account for one output token")
    return rows
