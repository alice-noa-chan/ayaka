"""Metadata migration preserves private inputs and the later frozen eval gate."""

import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest
import torch
from test_direct_bundle import dataset

from ayaka.config import tiny_config
from ayaka.eval.read_artifact import fingerprint
from ayaka.tokenization import ToyTokenizer
from ayaka.training.direct_audit import create_audit_receipt
from ayaka.training.direct_bundle import prepare_bundle
from ayaka.training.direct_holdout import open_holdout
from ayaka.training.prepare_v2 import canonical, sha256
from scripts.direct_v2 import rebind_holdout as migration


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def inputs(tmp_path):
    new = tmp_path / "new"
    samples = dataset()
    prepare_bundle(
        new,
        samples,
        ToyTokenizer(),
        tiny_config(readout="lm", max_seq_len=2048),
        {},
        steps=2,
        rows_per_step=16,
        allow_tiny=True,
    )
    new_sha = sha256((new / "manifest.json").read_bytes())
    new_audit = tmp_path / "new-audit.json"
    create_audit_receipt(new, new_audit, expected_manifest_sha256=new_sha, allow_tiny=True)
    old = tmp_path / "old"
    shutil.copytree(new, old)
    old_manifest = json.loads((old / "manifest.json").read_bytes())
    # Represent a separately trusted historical source/receipt in this fixture.
    # Real production anchors are never inferred from these synthetic digests.
    old_manifest["source_sha256"]["training/native_snapshot.py"] = sha256(b"old fixture source")
    (old / "manifest.json").write_bytes(canonical(old_manifest) + b"\n")
    old_sha = sha256((old / "manifest.json").read_bytes())
    old_audit = tmp_path / "old-audit.json"
    receipt = json.loads(new_audit.read_bytes())
    receipt.update(bundle_manifest_sha256=old_sha, source_sha256=old_manifest["source_sha256"])
    old_audit.write_bytes(canonical(receipt) + b"\n")
    private = tmp_path / "new-holdout"
    header = json.loads((private / "manifest.json").read_bytes())
    header["development_manifest_sha256"] = old_sha
    (private / "manifest.json").write_bytes(canonical(header) + b"\n")
    args = {
        "old_bundle": old,
        "new_bundle": new,
        "old_audit": old_audit,
        "new_audit": new_audit,
        "holdout": private,
        "out": tmp_path / "migrated-holdout",
        "expected_old_bundle_sha256": old_sha,
        "expected_new_bundle_sha256": new_sha,
        "expected_old_audit_sha256": sha256(old_audit.read_bytes()),
        "expected_new_audit_sha256": sha256(new_audit.read_bytes()),
        "expected_holdout_manifest_sha256": sha256((private / "manifest.json").read_bytes()),
    }
    return args, samples


def prevent_private_reads(monkeypatch, private):
    info = (private / "test.jsonl").stat()
    protected = info.st_dev, info.st_ino
    real_fdopen, real_path_open = os.fdopen, Path.open

    class Stream:
        def __init__(self, raw):
            self.raw = raw

        def __enter__(self):
            self.raw.__enter__()
            return self

        def __exit__(self, *args):
            return self.raw.__exit__(*args)

        def fileno(self):
            return self.raw.fileno()

        def read(self, *args):
            value = os.fstat(self.fileno())
            assert (value.st_dev, value.st_ino) != protected, "original private bytes were read"
            return self.raw.read(*args)

    def fdopen(fd, *args, **kwargs):
        return Stream(real_fdopen(fd, *args, **kwargs))

    def path_open(self, mode="r", *args, **kwargs):
        if "r" in mode and self.exists():
            value = self.stat()
            assert (value.st_dev, value.st_ino) != protected, "original private input was opened"
        return real_path_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(migration.os, "fdopen", fdopen)
    monkeypatch.setattr(Path, "open", path_open)


def frozen_selection(args, *, old=False):
    return {
        "version": "ayaka-direct-selection-1",
        "split": "dev",
        "complete": True,
        "test_opened": False,
        "development_manifest_sha256": args[
            "expected_old_bundle_sha256" if old else "expected_new_bundle_sha256"
        ],
        "candidate_model_sha256": "1" * 64,
        "policy_sha256": "2" * 64,
        "dev_report_sha256": "3" * 64,
        "calibration_sha256": "4" * 64,
    }


def evaluate(args, result, *, old=False):
    selection = frozen_selection(args, old=old)
    commitment = json.loads((args["new_bundle"] / "test_commitment.json").read_bytes())
    return open_holdout(
        args["out"],
        commitment=commitment,
        expected_manifest_sha256=result["manifest_sha256"],
        frozen_selection=selection,
        expected_selection_sha256=fingerprint(selection),
    )


def test_migration_never_reads_originals_and_requires_a_new_frozen_selection(inputs, monkeypatch):
    args, samples = inputs
    previous = (args["holdout"] / "manifest.json").read_bytes()
    with monkeypatch.context() as guarded:
        prevent_private_reads(guarded, args["holdout"])
        result = migration.rebind_holdout(**args)
        with pytest.raises(ValueError, match="not bound"):
            evaluate(args, result, old=True)
    actual = evaluate(args, result)
    assert [s.to_json() for s in actual] == [s.to_json() for s in samples["test"]]
    assert (args["holdout"] / "manifest.json").read_bytes() == previous
    assert os.path.samefile(args["holdout"] / "test.jsonl", args["out"] / "test.jsonl")
    report = json.loads((args["out"] / "migration.json").read_bytes())
    assert report["deferred_content_validation"] and report["new_frozen_dev_selection_required"]
    assert report["same_private_file"] and not report["original_test_inputs_opened"]
    assert (
        not report["original_test_content_verified"]
        and not report["model_policy_selection_migrated"]
    )
    assert report["changed_sources"].keys() == {"training/native_snapshot.py"}
    assert {path.name for path in args["out"].iterdir()} == {
        "manifest.json",
        "migration.json",
        "test.jsonl",
    }
    assert sha256((args["out"] / "migration.json").read_bytes()) == result["migration_sha256"]
    assert not result["production_ready"] and not result["gpu_allocated"]


@pytest.mark.parametrize(
    "field",
    [
        "expected_old_bundle_sha256",
        "expected_new_bundle_sha256",
        "expected_old_audit_sha256",
        "expected_new_audit_sha256",
        "expected_holdout_manifest_sha256",
    ],
)
def test_wrong_external_anchors_fail_before_output_and_original_reads(inputs, monkeypatch, field):
    args, _ = inputs
    args[field] = "0" * 64
    prevent_private_reads(monkeypatch, args["holdout"])
    with pytest.raises(ValueError, match="external anchor"):
        migration.rebind_holdout(**args)
    assert not args["out"].exists()


@pytest.mark.parametrize("target", ["new_audit", "payload_alias", "metadata_alias"])
def test_private_receipt_or_hardlink_alias_is_rejected_before_any_read(inputs, monkeypatch, target):
    args, _ = inputs
    original = args["holdout"] / "test.jsonl"
    if target == "new_audit":
        args["new_audit"] = original
    else:
        name = "train.jsonl" if target == "payload_alias" else "manifest.json"
        path = args["new_bundle"] / name
        path.unlink()
        os.link(original, path)
    prevent_private_reads(monkeypatch, args["holdout"])
    with pytest.raises(ValueError, match="original private"):
        migration.rebind_holdout(**args)
    assert not args["out"].exists()


def test_replacement_between_preflight_and_open_cannot_read_original_bytes(inputs, monkeypatch):
    args, _ = inputs
    target = args["new_audit"]
    real_open = os.open
    swapped = False

    def replace_before_open(path, *flags, **kwargs):
        nonlocal swapped
        if Path(path) == target and not swapped:
            swapped = True
            target.unlink()
            os.link(args["holdout"] / "test.jsonl", target)
        return real_open(path, *flags, **kwargs)

    prevent_private_reads(monkeypatch, args["holdout"])
    monkeypatch.setattr(migration.os, "open", replace_before_open)
    with pytest.raises(ValueError, match="original private|safe input identity"):
        migration.rebind_holdout(**args)
    assert swapped and not args["out"].exists()


def test_resigned_development_payload_changes_cannot_be_rebound(inputs, monkeypatch):
    args, _ = inputs
    path = args["new_bundle"] / "train.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    manifest = json.loads((args["new_bundle"] / "manifest.json").read_bytes())
    manifest["files"][path.name] = sha256(path.read_bytes())
    (args["new_bundle"] / "manifest.json").write_bytes(canonical(manifest))
    args["expected_new_bundle_sha256"] = sha256((args["new_bundle"] / "manifest.json").read_bytes())
    prevent_private_reads(monkeypatch, args["holdout"])
    with pytest.raises(ValueError, match="identical payloads"):
        migration.rebind_holdout(**args)
    assert not args["out"].exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("regeneration_complete", False),
        ("model_weights_loaded", True),
        ("optimizer_steps", False),
        ("rows", 0),
        ("inventory_sha256", "a" * 64),
    ],
)
def test_rechecksummed_incomplete_or_changed_audit_is_rejected(inputs, monkeypatch, field, value):
    args, _ = inputs
    audit = json.loads(args["new_audit"].read_bytes())
    audit[field] = value
    args["new_audit"].write_bytes(canonical(audit))
    args["expected_new_audit_sha256"] = sha256(args["new_audit"].read_bytes())
    prevent_private_reads(monkeypatch, args["holdout"])
    with pytest.raises(ValueError, match="complete CPU audit|complete regeneration"):
        migration.rebind_holdout(**args)
    assert not args["out"].exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("development_manifest_sha256", "a" * 64),
        ("commitment_sha256", "b" * 64),
        ("files", {"../test.jsonl": "c" * 64}),
        ("version", "ayaka-direct-holdout-1"),
        ("unbound", True),
    ],
)
def test_wrong_resigned_holdout_metadata_is_rejected(inputs, monkeypatch, field, value):
    args, _ = inputs
    path = args["holdout"] / "manifest.json"
    value_dict = json.loads(path.read_bytes())
    value_dict[field] = value
    path.write_bytes(canonical(value_dict))
    args["expected_holdout_manifest_sha256"] = sha256(path.read_bytes())
    prevent_private_reads(monkeypatch, args["holdout"])
    with pytest.raises(ValueError, match="holdout metadata"):
        migration.rebind_holdout(**args)
    assert not args["out"].exists()


def test_fresh_output_is_required_and_preexisting_records_are_preserved(inputs):
    args, _ = inputs
    args["out"].mkdir()
    record = args["out"] / "keep.txt"
    record.write_bytes(b"existing user record")
    with pytest.raises(ValueError, match="fresh separate"):
        migration.rebind_holdout(**args)
    assert record.read_bytes() == b"existing user record"


@pytest.mark.parametrize("root", ["old_bundle", "new_bundle", "holdout"])
def test_nested_output_is_rejected(inputs, root):
    args, _ = inputs
    args["out"] = args[root] / "nested-output"
    with pytest.raises(ValueError, match="fresh separate"):
        migration.rebind_holdout(**args)
    assert not args["out"].exists()


def test_link_failure_does_not_publish_successful_metadata(inputs, monkeypatch):
    args, _ = inputs

    def fail_link(*unused, **kwargs):
        raise OSError("hard links unavailable")

    monkeypatch.setattr(migration.os, "link", fail_link)
    prevent_private_reads(monkeypatch, args["holdout"])
    with pytest.raises(OSError, match="unavailable"):
        migration.rebind_holdout(**args)
    assert not (args["out"] / "manifest.json").exists()
    assert not (args["out"] / "migration.json").exists()


def test_payload_changed_during_link_cannot_publish_a_manifest(inputs, monkeypatch):
    args, _ = inputs
    real_link = os.link

    def link_and_mutate(*values, **kwargs):
        real_link(*values, **kwargs)
        path = args["new_bundle"] / "train.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")

    prevent_private_reads(monkeypatch, args["holdout"])
    monkeypatch.setattr(migration.os, "link", link_and_mutate)
    with pytest.raises(ValueError, match="changed before publication"):
        migration.rebind_holdout(**args)
    assert not (args["out"] / "manifest.json").exists()


def test_original_replaced_at_last_payload_check_cannot_claim_same_file(inputs, monkeypatch):
    args, _ = inputs
    target = args["new_bundle"] / "train_items.jsonl"
    original_digest = migration._digest
    checks = 0

    def digest_and_replace(path, guard):
        nonlocal checks
        result = original_digest(path, guard)
        if path == target:
            checks += 1
            if checks == 3:
                replacement = args["holdout"] / "replacement-fixture"
                replacement.write_bytes(b"unrelated synthetic private input")
                os.replace(replacement, args["holdout"] / "test.jsonl")
        return result

    prevent_private_reads(monkeypatch, args["holdout"])
    monkeypatch.setattr(migration, "_digest", digest_and_replace)
    with pytest.raises(ValueError, match="identity changed before publication"):
        migration.rebind_holdout(**args)
    assert not (args["out"] / "manifest.json").exists()


def test_later_evaluator_checks_corrupted_shared_bytes_after_rebinding(inputs):
    args, _ = inputs
    result = migration.rebind_holdout(**args)
    (args["out"] / "test.jsonl").write_bytes(b"corrupted synthetic holdout")
    with pytest.raises(ValueError, match="original bytes differ"):
        evaluate(args, result)


@pytest.mark.parametrize("target", ["migration.json", "manifest.json", "private_alias"])
def test_corrupted_or_replaced_output_cannot_return_successful_anchors(inputs, monkeypatch, target):
    args, _ = inputs
    real_write = migration._write_record

    def write_and_mutate(path, raw):
        real_write(path, raw)
        if path.name == "manifest.json":
            if target == "private_alias":
                record = args["out"] / "migration.json"
                record.unlink()
                os.link(args["holdout"] / "test.jsonl", record)
            else:
                (args["out"] / target).write_bytes(b"damaged metadata after closing")

    prevent_private_reads(monkeypatch, args["holdout"])
    monkeypatch.setattr(migration, "_write_record", write_and_mutate)
    with pytest.raises(ValueError, match="stored metadata|original private"):
        migration.rebind_holdout(**args)
    assert args["out"].is_dir()  # Preserve the partial operation for diagnosis.


def test_metadata_sync_failure_preserves_existing_records_without_publishing_manifest(
    inputs, monkeypatch
):
    args, _ = inputs
    previous = (args["holdout"] / "manifest.json").read_bytes()
    original_identity = migration._test_identity(args["holdout"] / "test.jsonl")

    def fail_sync(*unused):
        raise OSError("metadata sync failed")

    prevent_private_reads(monkeypatch, args["holdout"])
    monkeypatch.setattr(migration.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="sync failed"):
        migration.rebind_holdout(**args)
    assert (args["holdout"] / "manifest.json").read_bytes() == previous
    assert migration._test_identity(args["holdout"] / "test.jsonl") == original_identity
    assert not (args["out"] / "manifest.json").exists()


@pytest.mark.parametrize("operation", ["add", "remove"])
def test_source_inventory_additions_and_removals_keep_new_selection_required(
    inputs, monkeypatch, operation
):
    args, _ = inputs
    path = args["new_bundle"] / "manifest.json"
    manifest = json.loads(path.read_bytes())
    if operation == "add":
        name = "training/new_fixture.py"
        expected = {"old": None, "new": sha256(b"new fixture source")}
        manifest["source_sha256"][name] = expected["new"]
    else:
        name = "training/native_snapshot.py"
        old = json.loads((args["old_bundle"] / "manifest.json").read_bytes())
        expected = {"old": old["source_sha256"][name], "new": None}
        del manifest["source_sha256"][name]
    path.write_bytes(canonical(manifest) + b"\n")
    args["expected_new_bundle_sha256"] = sha256(path.read_bytes())
    audit = json.loads(args["new_audit"].read_bytes())
    audit.update(
        source_sha256=manifest["source_sha256"],
        bundle_manifest_sha256=args["expected_new_bundle_sha256"],
    )
    args["new_audit"].write_bytes(canonical(audit) + b"\n")
    args["expected_new_audit_sha256"] = sha256(args["new_audit"].read_bytes())
    with monkeypatch.context() as guarded:
        prevent_private_reads(guarded, args["holdout"])
        result = migration.rebind_holdout(**args)
    report = json.loads((args["out"] / "migration.json").read_bytes())
    assert report["changed_sources"][name] == expected
    assert not report["model_policy_selection_migrated"]
    assert report["new_frozen_dev_selection_required"]
    assert not result["production_ready"]


def test_original_replaced_during_output_readback_cannot_return_success(inputs, monkeypatch):
    args, _ = inputs
    real_digest = migration._digest

    def digest_and_replace(path, guard):
        result = real_digest(path, guard)
        if path == args["out"] / "manifest.json":
            replacement = args["holdout"] / "output-readback-replacement"
            replacement.write_bytes(b"another synthetic input")
            os.replace(replacement, args["holdout"] / "test.jsonl")
        return result

    prevent_private_reads(monkeypatch, args["holdout"])
    monkeypatch.setattr(migration, "_digest", digest_and_replace)
    with pytest.raises(ValueError, match="identity changed after publication"):
        migration.rebind_holdout(**args)


def test_reparse_original_is_rejected_before_any_content_read(inputs, monkeypatch):
    args, _ = inputs
    original = args["holdout"] / "test.jsonl"
    real_lstat = Path.lstat

    class ReparseStat:
        st_file_attributes = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            return getattr(self.original, name)

    def lstat(self):
        value = real_lstat(self)
        return ReparseStat(value) if self == original else value

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="reparse"):
        migration.rebind_holdout(**args)
    assert not args["out"].exists()


def test_cross_filesystem_has_no_copy_or_symlink_fallback(inputs, monkeypatch):
    args, _ = inputs
    real_stat = Path.stat
    parent = args["out"].parent

    def other_device(self, *values, **kwargs):
        result = real_stat(self, *values, **kwargs)
        if self == parent:
            fields = list(result)
            fields[2] += 1
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(Path, "stat", other_device)
    prevent_private_reads(monkeypatch, args["holdout"])
    with pytest.raises(ValueError, match="same filesystem"):
        migration.rebind_holdout(**args)
    assert not args["out"].exists()


def test_cli_imports_only_standard_library():
    import subprocess

    command = [
        sys.executable,
        "-c",
        "import sys; import scripts.direct_v2.rebind_holdout; assert 'torch' not in sys.modules",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
