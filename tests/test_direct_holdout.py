"""No holdout originals in development, and externally frozen opening checks."""

import copy
import json
from pathlib import Path

import pytest
from test_direct_bundle import dataset

from ayaka.config import tiny_config
from ayaka.eval.read_artifact import fingerprint
from ayaka.tokenization import ToyTokenizer
from ayaka.training.direct_bundle import audit_bundle, prepare_bundle
from ayaka.training.direct_holdout import (
    DEVELOPMENT_SPLITS,
    holdout_destination,
    open_holdout,
    validate_commitment,
)
from ayaka.training.prepare_v2 import sha256
from ayaka.training.run_direct import read_splits


@pytest.fixture
def prepared(tmp_path):
    splits = dataset()
    root = tmp_path / "dev"
    prepare_bundle(
        root,
        splits,
        ToyTokenizer(),
        tiny_config(readout="lm", max_seq_len=2048),
        {},
        steps=2,
        rows_per_step=16,
        allow_tiny=True,
    )
    commitment = json.loads((root / "test_commitment.json").read_bytes())
    holdout = tmp_path / "dev-holdout"
    frozen = {
        "version": "ayaka-direct-selection-1",
        "split": "dev",
        "complete": True,
        "test_opened": False,
        "development_manifest_sha256": sha256((root / "manifest.json").read_bytes()),
        "candidate_model_sha256": "1" * 64,
        "policy_sha256": "2" * 64,
        "dev_report_sha256": "3" * 64,
        "calibration_sha256": "4" * 64,
    }
    return root, holdout, splits, commitment, frozen


def opened(prepared, *, frozen=None, selection_sha=None, manifest_sha=None, commitment=None):
    _, holdout, _, original, selection = prepared
    selection = frozen if frozen is not None else selection
    return open_holdout(
        holdout,
        commitment=commitment or original,
        frozen_selection=selection,
        expected_selection_sha256=selection_sha or fingerprint(selection),
        expected_manifest_sha256=manifest_sha or sha256((holdout / "manifest.json").read_bytes()),
    )


def test_originals_live_only_in_separate_holdout_and_training_audit_never_opens_it(
    prepared, monkeypatch
):
    root, holdout, splits, commitment, _ = prepared
    assert not (root / "test.jsonl").exists()
    assert set(read_splits(root)) == set(DEVELOPMENT_SPLITS)
    originals = opened(prepared)
    assert [s.to_json() for s in originals] == [s.to_json() for s in splits["test"]]
    private = {p.resolve() for p in holdout.iterdir()}
    read = Path.read_bytes
    read_text = Path.read_text

    def protected_read(self, *args, **kwargs):
        assert self.resolve() not in private, "development audit must not open private holdout"
        return read(self, *args, **kwargs)

    def protected_text(self, *args, **kwargs):
        assert self.resolve() not in private, "development runner must not read private holdout"
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", protected_read)
    monkeypatch.setattr(Path, "read_text", protected_text)
    audit_bundle(root, allow_tiny=True)
    read_splits(root)
    assert commitment["contains_original_inputs_or_gold"] is False
    member = commitment["members"][0]
    assert set(member) == {"state_sha256", "content_sha256", "lineage_sha256"}
    text = (root / "test_commitment.json").read_text()
    assert splits["test"][0].metadata["source_example_id"] not in text
    assert str(splits["test"][0].state) not in text
    assert "target_distribution" not in text


@pytest.mark.parametrize("field", ["state_sha256", "source_lineage", "translation_of"])
def test_overlap_is_rejected_even_when_questions_or_gold_change(prepared, field):
    _, _, splits, commitment, _ = prepared
    development = copy.deepcopy({s: splits[s] for s in DEVELOPMENT_SPLITS})
    sample = development["train"][0]
    if field == "state_sha256":
        sample.state = splits["test"][0].state
        sample.questions[0].instruction = "Another instruction hiding the duplicated document?"
    elif field == "source_lineage":
        sample.metadata[field] = splits["test"][0].metadata[field]
    else:
        sample.metadata[field] = "shared-original-document"
        commitment = copy.deepcopy(commitment)
        commitment["members"][0]["lineage_sha256"][field] = fingerprint(sample.metadata[field])
    with pytest.raises(ValueError, match="overlap"):
        validate_commitment(commitment, development)


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "test"),
        ("test_opened", True),
        ("complete", False),
        ("candidate_model_sha256", "latest"),
    ],
)
def test_invalid_selection_is_rejected_before_opening_holdout(tmp_path, field, value):
    # An absent directory proves validation precedes any private file read.
    splits = dataset()
    from ayaka.training.direct_holdout import make_commitment
    from ayaka.training.prepare_v2 import audit_splits

    counts = audit_splits(splits)["test"]
    commitment, _ = make_commitment(
        splits["test"],
        context={
            "max_tokens": 500,
            "questions": counts["questions"],
            "rendered_rows_sha256": "a" * 64,
        },
        counts=counts,
    )
    selection = {
        "version": "ayaka-direct-selection-1",
        "split": "dev",
        "complete": True,
        "test_opened": False,
        **dict.fromkeys(
            (
                "development_manifest_sha256",
                "candidate_model_sha256",
                "policy_sha256",
                "dev_report_sha256",
                "calibration_sha256",
            ),
            "b" * 64,
        ),
    }
    selection[field] = value
    with pytest.raises(ValueError):
        open_holdout(
            tmp_path / "absent",
            commitment=commitment,
            expected_manifest_sha256="a" * 64,
            frozen_selection=selection,
            expected_selection_sha256=fingerprint(selection),
        )


def test_selection_and_holdout_external_anchors_reject_consistent_replacements(prepared):
    _, _, _, _, original = prepared
    changed = {**original, "candidate_model_sha256": "f" * 64}
    with pytest.raises(ValueError, match="external anchor"):
        opened(prepared, frozen=changed, selection_sha=fingerprint(original))
    with pytest.raises(ValueError, match="external anchor"):
        opened(prepared, manifest_sha="f" * 64)


def test_modified_holdout_bytes_fail_and_accidental_copy_in_bundle_is_rejected(prepared):
    root, holdout, _, _, _ = prepared
    path = holdout / "test.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="original bytes"):
        opened(prepared)
    (root / "test.jsonl").write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="must not contain original test"):
        audit_bundle(root, allow_tiny=True)


@pytest.mark.parametrize("suffix", ["", "nested/private", ".."])
def test_holdout_location_cannot_overlap_development_directory(tmp_path, suffix):
    root = tmp_path / "dev"
    with pytest.raises(ValueError, match="separate"):
        holdout_destination(root, root / suffix)


def test_private_path_conflict_leaves_both_existing_and_development_paths_untouched(tmp_path):
    holdout = tmp_path / "existing"
    holdout.mkdir()
    (holdout / "user-file.txt").write_text("preserve")
    root = tmp_path / "dev"
    with pytest.raises(ValueError, match="new directory"):
        prepare_bundle(
            root,
            dataset(),
            ToyTokenizer(),
            tiny_config(readout="lm"),
            {},
            steps=2,
            rows_per_step=16,
            allow_tiny=True,
            holdout_out=holdout,
        )
    assert not root.exists()
    assert (holdout / "user-file.txt").read_text() == "preserve"


@pytest.mark.parametrize("bad", ["raw_field", "context", "unknown_lineage", "count"])
def test_commitment_cannot_contain_raw_fields_or_invalid_counts(prepared, bad):
    commitment = copy.deepcopy(prepared[3])
    if bad == "raw_field":
        commitment["gold"] = {"yes": 1}
    elif bad == "context":
        commitment["context_audit"]["max_tokens"] = True
    elif bad == "unknown_lineage":
        commitment["members"][0]["lineage_sha256"]["unknown"] = "a" * 64
    else:
        commitment["counts"]["questions"] += 1
    with pytest.raises(ValueError):
        validate_commitment(commitment, {})
