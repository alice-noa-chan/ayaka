import copy
import json

import pytest

from ayaka.training.prepare_v2 import (
    VERSION,
    audit_splits,
    build_splits,
    canonical,
    sha256,
    validate_bundle,
)


def test_bundle_split_audit_and_exact_target_roundtrip():
    splits = build_splits(per_type=4, image_cases=2, candidate_cases=4)
    counts = audit_splits(splits)
    assert all(count["samples"] == 104 for count in counts.values())
    assert all(set(count["languages"]) == {"en", "ko", "ja"} for count in counts.values())


@pytest.mark.parametrize(
    "key", ["source_lineage", "generator_template_id", "rule_combination", "document_voice"]
)
def test_derived_split_leakage_fails_closed(key):
    splits = build_splits(1, 1, 1)
    splits["dev"][0].metadata[key] = splits["train"][0].metadata[key]
    with pytest.raises(ValueError, match="cross-split leakage"):
        audit_splits(splits)


def test_identical_evidence_cannot_hide_behind_changed_target_or_lineage():
    splits = build_splits(1, 1, 1)
    source = copy.deepcopy(splits["train"][0])
    source.metadata = copy.deepcopy(splits["dev"][0].metadata)
    candidates = source.questions[0].candidates
    source.questions[0].target_distribution = {c.id: 1 / len(candidates) for c in candidates}
    splits["dev"][0] = source
    with pytest.raises(ValueError, match="leakage in content"):
        audit_splits(splits)


def test_verified_bundle_detects_tampering_and_manifest_path_injection(tmp_path):
    splits = build_splits(1, 1, 1)
    counts = audit_splits(splits)
    for split, samples in splits.items():
        (tmp_path / f"{split}.jsonl").write_bytes(
            b"\n".join(canonical(s.to_json()) for s in samples) + b"\n"
        )
    for name in ("training_config.json", "model_preflight.json"):
        (tmp_path / name).write_bytes(b"{}")
    files = {p.name: sha256(p.read_bytes()) for p in tmp_path.iterdir()}
    manifest = {"version": VERSION, "files": files, "counts": counts}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    validate_bundle(tmp_path)
    (tmp_path / "train.jsonl").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        validate_bundle(tmp_path)
    manifest["files"]["../../secret"] = "not-a-checksum"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        validate_bundle(tmp_path)
