"""Rebind unchanged holdout metadata after a separately audited source change.

Original test bytes are never read. A fresh hard link keeps the private inputs
in a separate directory; open_holdout must verify those bytes after a new,
independently frozen dev-only model/policy selection. No selection is copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

VERSION = "ayaka-direct-holdout-rebinding-1"
FILES = {
    "train.jsonl",
    "router_train.jsonl",
    "dev.jsonl",
    "calibration.jsonl",
    "test_commitment.json",
    "teacher_reads.json",
    "recipe.json",
    "preparation.json",
    "train_items.jsonl",
}
BLOCK = 4 * 1024 * 1024
MAX_METADATA = 32 * 1024 * 1024


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def _sha(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("rebinding requires externally pinned SHA256 digests")
    return value


def _regular(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or getattr(value, "st_file_attributes", 0) & getattr(
        stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
    ):
        raise ValueError(
            "rebinding inputs must be regular files, not symbolic links/reparse points"
        )
    return value


@contextmanager
def _reader(path, read_guard):
    read_guard(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        # A path can be replaced after preflight. Check the actual opened
        # descriptor before reading any bytes, including hard-link aliases.
        read_guard(path, os.fstat(stream.fileno()))
        yield stream


def _digest(path, read_guard):
    digest = hashlib.sha256()
    with _reader(path, read_guard) as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_record(path, raw):
    with path.open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("rebinding metadata write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("rebinding JSON contains duplicate keys")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"rebinding JSON contains a nonfinite value: {value}")


def _json(raw):
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)


def _pinned(path, expected, read_guard):
    _sha(expected)
    if read_guard(path).st_size > MAX_METADATA:
        raise ValueError("rebinding metadata exceeds the size limit")
    with _reader(path, read_guard) as stream:
        raw = stream.read(MAX_METADATA + 1)
    if len(raw) > MAX_METADATA or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("rebinding metadata differs from its external anchor")
    value = _json(raw)
    if not isinstance(value, dict):
        raise ValueError("rebinding metadata must be a JSON object")
    return value


def _sources(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("rebinding requires a nonempty source inventory")
    for name, digest in value.items():
        path = PurePosixPath(name)
        if (
            not isinstance(name, str)
            or not name.endswith(".py")
            or "\\" in name
            or ":" in name
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or str(path) != name
        ):
            raise ValueError("rebinding source paths must be safe relative Python paths")
        _sha(digest)


def _bundle(root, expected, read_guard):
    manifest = _pinned(root / "manifest.json", expected, read_guard)
    if (
        manifest.get("version") != "ayaka-direct-bundle-6"
        or manifest.get("status") != "cpu_prepared_no_model_or_optimizer_execution"
        or manifest.get("promotable") is not False
        or manifest.get("execution_attested") is not False
        or type(manifest.get("optimizer_steps_executed")) is not int
        or manifest["optimizer_steps_executed"] != 0
        or manifest.get("test_storage") != "separate_holdout_directory_with_opaque_commitments"
        or not isinstance(manifest.get("files"), dict)
        or set(manifest["files"]) != FILES
        or (root / "test.jsonl").exists()
        or (root / "test.jsonl").is_symlink()
    ):
        raise ValueError("rebinding requires an unpromoted development-only bundle")
    _sources(manifest.get("source_sha256"))
    for name, digest in manifest["files"].items():
        if _digest(root / name, read_guard) != _sha(digest):
            raise ValueError("rebinding development payload differs from its manifest")
    return manifest


def _audit(path, expected, manifest, bundle_sha, read_guard):
    audit = _pinned(path, expected, read_guard)
    if (
        audit.get("version") != "ayaka-direct-cpu-audit-1"
        or audit.get("bundle_manifest_sha256") != bundle_sha
        or audit.get("source_sha256") != manifest["source_sha256"]
        or audit.get("regeneration_complete") is not True
        or audit.get("model_weights_loaded") is not False
        or audit.get("promotable") is not False
        or audit.get("execution_attested") is not False
        or type(audit.get("optimizer_steps")) is not int
        or audit["optimizer_steps"] != 0
        or type(audit.get("rows")) is not int
        or audit["rows"] < 1
    ):
        raise ValueError("rebinding requires the corresponding complete CPU audit")
    return audit


def _without(value, names):
    return {key: item for key, item in value.items() if key not in names}


def _separate(left, right):
    return left != right and left not in right.parents and right not in left.parents


def _test_identity(path):
    value = _regular(path)
    if not value.st_ino:
        raise ValueError("rebinding requires a filesystem with stable hard-link identities")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def rebind_holdout(
    *,
    old_bundle,
    new_bundle,
    old_audit,
    new_audit,
    holdout,
    out,
    expected_old_bundle_sha256,
    expected_new_bundle_sha256,
    expected_old_audit_sha256,
    expected_new_audit_sha256,
    expected_holdout_manifest_sha256,
):
    old_root, new_root, private, destination = (
        Path(value).resolve() for value in (old_bundle, new_bundle, holdout, out)
    )
    if (
        destination.exists()
        or Path(out).is_symlink()
        or old_root == new_root
        or not destination.parent.is_dir()
        or any(not _separate(private, root) for root in (old_root, new_root))
        or any(not _separate(destination, root) for root in (old_root, new_root, private))
    ):
        raise ValueError("rebinding requires a fresh separate holdout under an existing parent")
    original = private / "test.jsonl"
    identity = _test_identity(original)
    if identity[0] != destination.parent.stat().st_dev:
        raise ValueError(
            "rebinding requires the same filesystem; original bytes must not be copied"
        )
    pinned_inputs = [
        (old_root / "manifest.json", expected_old_bundle_sha256),
        (new_root / "manifest.json", expected_new_bundle_sha256),
        (Path(old_audit), expected_old_audit_sha256),
        (Path(new_audit), expected_new_audit_sha256),
        (private / "manifest.json", expected_holdout_manifest_sha256),
    ]
    read_targets = [path for path, _ in pinned_inputs] + [
        root / name for root in (old_root, new_root) for name in sorted(FILES)
    ]
    allowed_reads = {path.resolve() for path in read_targets}

    def read_guard(path, opened=None):
        value = _regular(path)
        resolved = path.resolve()
        if (
            resolved not in allowed_reads
            or (value.st_dev, value.st_ino) == identity[:2]
            or resolved.is_relative_to(private)
            and resolved != private / "manifest.json"
        ):
            raise ValueError("rebinding must not open original private test inputs or aliases")
        if opened is not None and (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) == identity[:2]
            or (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            != (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        ):
            raise ValueError("rebinding opened descriptor differs from its safe input identity")
        return value

    # Wrong receipt paths and payload hard-link aliases must fail before the
    # first read, including when their digest would ultimately be rejected.
    for path in read_targets:
        read_guard(path)
    old = _bundle(old_root, expected_old_bundle_sha256, read_guard)
    new = _bundle(new_root, expected_new_bundle_sha256, read_guard)
    if expected_old_bundle_sha256 == expected_new_bundle_sha256 or _without(
        old, {"source_sha256"}
    ) != _without(new, {"source_sha256"}):
        raise ValueError("rebinding only permits identical payloads and source-inventory changes")
    audits = [
        _audit(
            Path(old_audit), expected_old_audit_sha256, old, expected_old_bundle_sha256, read_guard
        ),
        _audit(
            Path(new_audit), expected_new_audit_sha256, new, expected_new_bundle_sha256, read_guard
        ),
    ]
    ignored = {"source_sha256", "bundle_manifest_sha256"}
    if _without(audits[0], ignored) != _without(audits[1], ignored):
        raise ValueError("rebinding requires identical complete regeneration results")
    commitment = _pinned(
        new_root / "test_commitment.json", new["files"]["test_commitment.json"], read_guard
    )
    previous = _pinned(private / "manifest.json", expected_holdout_manifest_sha256, read_guard)
    if (
        set(previous) != {"version", "development_manifest_sha256", "commitment_sha256", "files"}
        or previous.get("version") != "ayaka-direct-holdout-2"
        or previous.get("development_manifest_sha256") != expected_old_bundle_sha256
        or previous.get("commitment_sha256") != hashlib.sha256(_canonical(commitment)).hexdigest()
        or commitment.get("version") != "ayaka-direct-holdout-2"
        or commitment.get("split") != "test"
        or commitment.get("contains_original_inputs_or_gold") is not False
        or previous.get("files") != {"test.jsonl": _sha(commitment.get("raw_sha256"))}
    ):
        raise ValueError("rebinding holdout metadata differs from the unchanged opaque commitment")
    manifest = {**previous, "development_manifest_sha256": expected_new_bundle_sha256}
    manifest_raw = _canonical(manifest) + b"\n"
    # Read development payloads and metadata again before publishing. Original
    # test content is intentionally deferred to the independently frozen eval.
    for path, expected in pinned_inputs:
        if _digest(path, read_guard) != expected:
            raise ValueError("rebinding metadata changed during validation")
    for root in (old_root, new_root):
        for name, expected in new["files"].items():
            if _digest(root / name, read_guard) != expected:
                raise ValueError("rebinding development payload changed during validation")
    if _test_identity(original) != identity:
        raise ValueError("rebinding original file identity changed during validation")
    destination.mkdir(exist_ok=False)
    os.link(original, destination / "test.jsonl", follow_symlinks=False)
    if (
        _test_identity(original) != identity
        or _test_identity(destination / "test.jsonl") != identity
    ):
        raise ValueError("rebinding hard link differs from the original private file")
    for path, expected in pinned_inputs:
        if _digest(path, read_guard) != expected:
            raise ValueError("rebinding metadata changed before publication")
    for root in (old_root, new_root):
        for name, expected in new["files"].items():
            if _digest(root / name, read_guard) != expected:
                raise ValueError("rebinding development payload changed before publication")
    if (
        _test_identity(original) != identity
        or _test_identity(destination / "test.jsonl") != identity
    ):
        raise ValueError("rebinding private file identity changed before publication")
    report = {
        "version": VERSION,
        "old_development_manifest_sha256": expected_old_bundle_sha256,
        "new_development_manifest_sha256": expected_new_bundle_sha256,
        "old_cpu_audit_sha256": expected_old_audit_sha256,
        "new_cpu_audit_sha256": expected_new_audit_sha256,
        "old_holdout_manifest_sha256": expected_holdout_manifest_sha256,
        "new_holdout_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "commitment_file_sha256": new["files"]["test_commitment.json"],
        "development_payloads_sha256": new["files"],
        "changed_sources": {
            name: {"old": old["source_sha256"].get(name), "new": new["source_sha256"].get(name)}
            for name in sorted(old["source_sha256"].keys() | new["source_sha256"].keys())
            if old["source_sha256"].get(name) != new["source_sha256"].get(name)
        },
        "private_file_identity": dict(
            zip(("device", "inode", "bytes", "mtime_ns"), identity, strict=True)
        ),
        "same_private_file": True,
        "original_test_inputs_opened": False,
        "original_test_content_verified": False,
        "deferred_content_validation": True,
        "model_policy_selection_migrated": False,
        "new_frozen_dev_selection_required": True,
        "execution_attested": False,
        "promotable": False,
        "gpu_allocated": False,
        "optimizer_steps": 0,
        "scope": "metadata/payload identity rebinding; test bytes and model/policy selection verified by later evaluator",
    }
    report_raw = _canonical(report) + b"\n"
    report_path = destination / "migration.json"
    manifest_path = destination / "manifest.json"
    _write_record(report_path, report_raw)
    # The normal holdout manifest is published last. A failed operation may
    # leave a diagnostic partial directory; it returns no successful anchor.
    _write_record(manifest_path, manifest_raw)
    # Expected bytes alone cannot attest to the files actually stored. The
    # guarded descriptor also rejects an output replaced by a private alias.
    allowed_reads.update({report_path.resolve(), manifest_path.resolve()})
    report_sha = hashlib.sha256(report_raw).hexdigest()
    for path, expected in (
        (report_path, report_sha),
        (manifest_path, report["new_holdout_manifest_sha256"]),
    ):
        if _digest(path, read_guard) != expected:
            raise ValueError("rebinding stored metadata differs from its expected bytes")
    if (
        _test_identity(original) != identity
        or _test_identity(destination / "test.jsonl") != identity
    ):
        raise ValueError("rebinding private file identity changed after publication")
    return {
        "holdout": str(destination),
        "manifest_sha256": report["new_holdout_manifest_sha256"],
        "migration_sha256": report_sha,
        "original_test_inputs_opened": False,
        "production_ready": False,
        "gpu_allocated": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-bundle", "new-bundle", "old-audit", "new-audit", "holdout", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("old-bundle", "new-bundle", "old-audit", "new-audit", "holdout-manifest"):
        parser.add_argument("--expected-" + name + "-sha256", required=True)
    print(json.dumps(rebind_holdout(**vars(parser.parse_args(argv))), sort_keys=True))


if __name__ == "__main__":
    main()
