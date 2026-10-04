"""Separate holdout storage with opaque commitments in development bundles.

This is data separation and a declared frozen-selection contract, not filesystem
access control, cryptographic secrecy, execution attestation or a promotion gate.
Preparation can inspect original gold; the training runner never opens it.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..data.schema import Sample
from ..eval.read_artifact import fingerprint
from .direct_distillation import _sha
from .prepare_v2 import LINEAGE_KEYS, canonical, sha256

VERSION = "ayaka-direct-holdout-1"
DEVELOPMENT_SPLITS = ("train", "router_train", "dev", "calibration")


def signatures(sample):
    data = sample.to_json()
    content = {
        "state": sample.state,
        "questions": [
            {k: v for k, v in q.items() if k not in {"target_distribution", "id"}}
            for q in data["questions"]
        ],
        "media": sample.metadata.get("media"),
    }
    fields = (
        *LINEAGE_KEYS,
        "source_example_id",
        "translation_of",
        "derived_from",
        "case_facts_sha256",
    )
    return {
        "state_sha256": fingerprint(sample.state),
        "content_sha256": sha256(canonical(content)),
        "lineage_sha256": {
            key: fingerprint(sample.metadata[key])
            for key in fields
            if sample.metadata.get(key) is not None
        },
    }


def make_commitment(samples, *, context, counts):
    if not samples or any(s.metadata.get("split") != "test" for s in samples):
        raise ValueError("holdout storage requires nonempty test samples")
    raw = b"\n".join(canonical(s.to_json()) for s in samples) + b"\n"
    result = {
        "version": VERSION,
        "split": "test",
        "raw_sha256": sha256(raw),
        "split_sha256": fingerprint([s.to_json() for s in samples]),
        "members": [signatures(s) for s in samples],
        "context_audit": context,
        "counts": counts,
        "contains_original_inputs_or_gold": False,
    }
    return result, raw


def validate_commitment(commitment, development):
    if (
        not isinstance(commitment, dict)
        or commitment.get("version") != VERSION
        or commitment.get("split") != "test"
        or commitment.get("contains_original_inputs_or_gold") is not False
    ):
        raise ValueError("require opaque test commitments")
    if set(commitment) != {
        "version",
        "split",
        "raw_sha256",
        "split_sha256",
        "members",
        "context_audit",
        "counts",
        "contains_original_inputs_or_gold",
    }:
        raise ValueError("holdout commitment must contain only the known opaque fields")
    context, counts = commitment.get("context_audit"), commitment.get("counts")
    if (
        not isinstance(context, dict)
        or set(context) != {"max_tokens", "rendered_rows_sha256", "questions"}
        or any(
            type(context.get(key)) is not int or context[key] < 1
            for key in ("max_tokens", "questions")
        )
    ):
        raise ValueError("holdout context audit must retain positive exact counts")
    _sha(context["rendered_rows_sha256"], "rendered_rows_sha256")
    if (
        not isinstance(counts, dict)
        or set(counts) != {"samples", "questions", "types", "languages", "modalities", "families"}
        or any(
            type(counts.get(key)) is not int or counts[key] < 1 for key in ("samples", "questions")
        )
        or counts["questions"] != context["questions"]
    ):
        raise ValueError("holdout count audit is invalid")
    for key in ("types", "languages", "modalities", "families"):
        if (
            not isinstance(counts[key], dict)
            or not counts[key]
            or any(
                not isinstance(k, str) or not k or type(v) is not int or v < 1
                for k, v in counts[key].items()
            )
        ):
            raise ValueError("holdout category counts are invalid")
    if sum(counts["types"].values()) != counts["questions"] or set(counts["types"]) - {
        "noul",
        "choice",
        "score",
    }:
        raise ValueError("holdout type counts are invalid")
    for key in ("raw_sha256", "split_sha256"):
        _sha(commitment.get(key), key)
    members = commitment.get("members")
    if (
        not isinstance(members, list)
        or not members
        or commitment.get("counts", {}).get("samples") != len(members)
    ):
        raise ValueError("holdout commitment members/counts mismatch")
    hashes = {key: set() for key in ("state_sha256", "content_sha256")}
    lineage = {}
    for row in members:
        if set(row) != {"state_sha256", "content_sha256", "lineage_sha256"}:
            raise ValueError("holdout commitments must contain hashes only")
        for key in hashes:
            _sha(row[key], key)
            hashes[key].add(row[key])
        if not isinstance(row["lineage_sha256"], dict) or not row["lineage_sha256"].get(
            "source_lineage"
        ):
            raise ValueError("holdout commitments require source lineage")
        for key, value in row["lineage_sha256"].items():
            if key not in (
                *LINEAGE_KEYS,
                "source_example_id",
                "translation_of",
                "derived_from",
                "case_facts_sha256",
            ):
                raise ValueError("unknown holdout lineage field")
            _sha(value, key)
            lineage.setdefault(key, set()).add(value)
    for split, samples in development.items():
        if split not in DEVELOPMENT_SPLITS:
            raise ValueError("holdout isolation only accepts development splits")
        for sample in samples:
            row = signatures(sample)
            if any(row[key] in hashes[key] for key in hashes) or any(
                value in lineage.get(key, set()) for key, value in row["lineage_sha256"].items()
            ):
                raise ValueError("development/holdout evidence or lineage overlap")
    return commitment


def holdout_destination(bundle, explicit=None):
    root = Path(bundle).resolve()
    path = (
        Path(explicit).resolve() if explicit is not None else root.with_name(root.name + "-holdout")
    )
    if path == root or root in path.parents or path in root.parents:
        raise ValueError("holdout must be separate from the development bundle")
    if path.exists():
        raise ValueError("holdout output must be a new directory")
    return path


def write_holdout(path, raw, commitment, development_manifest_sha256):
    _sha(development_manifest_sha256, "development_manifest_sha256")
    validate_commitment(commitment, {})
    if sha256(raw) != commitment["raw_sha256"]:
        raise ValueError("holdout write must match committed original bytes")
    manifest = {
        "version": VERSION,
        "development_manifest_sha256": development_manifest_sha256,
        "commitment_sha256": fingerprint(commitment),
        "files": {"test.jsonl": sha256(raw)},
    }
    root = Path(path)
    root.mkdir(parents=True, exist_ok=False)
    (root / "test.jsonl").write_bytes(raw)
    (root / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    return manifest


def open_holdout(
    path,
    *,
    commitment,
    expected_manifest_sha256,
    frozen_selection,
    expected_selection_sha256,
):
    """Open original test only with an externally anchored, dev-only selection.

    The evaluator must independently verify selected checkpoint/policy bytes.
    A caller-declared frozen selection is not proof that test was never seen.
    """
    validate_commitment(commitment, {})
    for key, value in (
        ("holdout_manifest_sha256", expected_manifest_sha256),
        ("selection_sha256", expected_selection_sha256),
    ):
        _sha(value, key)
    if (
        not isinstance(frozen_selection, dict)
        or fingerprint(frozen_selection) != expected_selection_sha256
    ):
        raise ValueError("frozen dev selection differs from its external anchor")
    if (
        frozen_selection.get("version") != "ayaka-direct-selection-1"
        or frozen_selection.get("split") != "dev"
        or frozen_selection.get("complete") is not True
        or frozen_selection.get("test_opened") is not False
    ):
        raise ValueError("holdout requires complete dev-only selection frozen before test")
    for key in (
        "development_manifest_sha256",
        "candidate_model_sha256",
        "policy_sha256",
        "dev_report_sha256",
        "calibration_sha256",
    ):
        _sha(frozen_selection.get(key), key)
    root = Path(path)
    manifest_raw = (root / "manifest.json").read_bytes()
    if sha256(manifest_raw) != expected_manifest_sha256:
        raise ValueError("holdout manifest differs from its external anchor")
    manifest = json.loads(manifest_raw)
    if (
        manifest.get("version") != VERSION
        or manifest.get("commitment_sha256") != fingerprint(commitment)
        or manifest.get("development_manifest_sha256")
        != frozen_selection["development_manifest_sha256"]
        or set(manifest.get("files", {})) != {"test.jsonl"}
    ):
        raise ValueError("holdout is not bound to the selected development bundle")
    raw = (root / "test.jsonl").read_bytes()
    if sha256(raw) != commitment["raw_sha256"] or sha256(raw) != manifest["files"]["test.jsonl"]:
        raise ValueError("holdout original bytes differ from the preparation commitment")
    samples = [Sample.from_json(json.loads(line)) for line in raw.splitlines()]
    actual, _ = make_commitment(
        samples, context=commitment["context_audit"], counts=commitment["counts"]
    )
    if actual != commitment:
        raise ValueError("holdout content/lineage differs from the preparation commitment")
    return samples
