"""Upload payloads preserve audited inputs and never initiate paid work."""

import hashlib
import importlib.util
import io
import json
import marshal
import os
import shutil
import struct
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import torch
import zstandard
from test_direct_native_upload import prepared

from ayaka.training.prepare_v2 import canonical, sha256
from scripts.direct_v2 import package

SOURCE = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(tmp_path):
    bundle, native, _, receipt, anchors, snapshot = prepared(tmp_path)
    record = tmp_path / "snapshot.json"
    record.write_bytes(canonical(snapshot) + b"\n")
    return {
        "source_root": SOURCE,
        "bundle": bundle,
        "audit_receipt": receipt,
        "snapshot_record": record,
        "snapshot_path": native,
        "out": tmp_path / "upload.tar.zst",
        "allow_tiny": True,
        "expected_bundle_sha256": anchors["expected_manifest_sha256"],
        "expected_audit_receipt_sha256": anchors["expected_receipt_sha256"],
        "expected_snapshot_record_sha256": sha256(record.read_bytes()),
    }


def unpack(path, destination):
    with (
        path.open("rb") as raw,
        zstandard.ZstdDecompressor().stream_reader(raw) as stream,
        tarfile.open(fileobj=stream, mode="r|") as archive,
    ):
        for member in archive:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())


def test_upload_relocates_audited_bytes_and_runs_cpu_audit_in_empty_cache(tmp_path):
    args = inputs(tmp_path)
    hidden = args["snapshot_path"] / ".cache/huggingface/private.json"
    hidden.parent.mkdir(parents=True)
    hidden.write_text("cache sentinel")
    holdout = args["bundle"].with_name("bundle-holdout")
    holdout.mkdir(exist_ok=True)
    (holdout / "unseen-original.jsonl").write_text("holdout sentinel")
    (args["bundle"] / "unbound-file.txt").write_text("not audited")
    result = package.build_archive(**args)
    assert not result["production_ready"] and not result["gpu_allocated"]
    assert result["compression_level"] == 3
    manifest = package.verify_archive(args["out"], result["sha256"])
    assert manifest["audit_command"][3] == "audit"
    assert not any(
        name.startswith(("native/.cache/", "bundle-holdout/")) or name == "bundle/unbound-file.txt"
        for name in manifest["files"]
    )
    assert not manifest["original_test_inputs_included"]
    relocated = tmp_path / "relocated"
    unpack(args["out"], relocated)
    root = relocated / "ayaka-direct"
    for name, expected in manifest["files"].items():
        assert package._digest(root / name) == expected["sha256"]
    command = [sys.executable, *manifest["audit_command"][1:]]
    environment = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        HF_HUB_CACHE=str(tmp_path / "empty-cache"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        command, cwd=root, env=environment, capture_output=True, text=True, timeout=90
    )
    assert completed.returncode == 0, completed.stderr
    audited = json.loads(completed.stdout)
    assert audited["status"] == "audited_cpu_only"
    assert not audited["model_weights_loaded"] and audited["optimizer_steps"] == 0
    assert audited["native_weight_bytes_verified"]


@pytest.mark.parametrize("anchor", ["bundle", "audit_receipt", "snapshot_record"])
def test_wrong_external_anchor_fails_before_archive_publication(tmp_path, anchor):
    args = inputs(tmp_path)
    args["expected_" + anchor + "_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256|external|anchor|digest|unbound"):
        package.build_archive(**args)
    assert not args["out"].exists()


def test_changed_weight_during_streaming_leaves_no_partial_archive(tmp_path, monkeypatch):
    args = inputs(tmp_path)
    original = package.CheckedReader.__init__
    changed = False

    def mutate(self, stream):
        nonlocal changed
        original(self, stream)
        if not changed and str(stream.name).endswith(".safetensors"):
            changed = True
            with Path(stream.name).open("r+b") as file:
                file.seek(-1, 2)
                byte = file.read(1)[0]
                file.seek(-1, 2)
                file.write(bytes([byte ^ 1]))

    monkeypatch.setattr(package.CheckedReader, "__init__", mutate)
    with pytest.raises(ValueError, match="changed while streaming"):
        package.build_archive(**args)
    assert changed and not args["out"].exists()


def test_production_packaging_rejects_already_loaded_ayaka(tmp_path):
    args = inputs(tmp_path)
    receipt = package._pinned_receipt(args["audit_receipt"], args["expected_audit_receipt_sha256"])
    with pytest.raises(ValueError, match="preloaded Ayaka"):
        package._source_imports(SOURCE, receipt["source_sha256"])


def test_changed_frozen_source_is_rejected_before_executing_imports(tmp_path):
    args = inputs(tmp_path)
    frozen = tmp_path / "frozen"
    shutil.copytree(
        SOURCE / "ayaka", frozen / "ayaka", ignore=shutil.ignore_patterns("__pycache__")
    )
    module = frozen / "ayaka/training/direct_audit.py"
    module.write_bytes(b"raise RuntimeError('UNTRUSTED CODE EXECUTED')\n" + module.read_bytes())
    command = [sys.executable, "-m", "scripts.direct_v2.package", "build", "--allow-tiny"]
    for key, value in {**args, "source_root": frozen}.items():
        if key == "allow_tiny":
            continue
        command.extend(["--" + key.replace("_", "-"), str(value)])
    completed = subprocess.run(command, cwd=SOURCE, capture_output=True, text=True, timeout=45)
    assert completed.returncode != 0
    assert "before import" in completed.stderr and "UNTRUSTED CODE EXECUTED" not in completed.stderr
    assert not args["out"].exists()


def test_native_recipe_cannot_use_the_tiny_preloaded_exception(tmp_path):
    args = inputs(tmp_path)
    root = args["bundle"]
    recipe = json.loads((root / "recipe.json").read_bytes())
    recipe["model"]["backbone"] = "publisher/native"
    (root / "recipe.json").write_bytes(canonical(recipe) + b"\n")
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest["files"]["recipe.json"] = sha256((root / "recipe.json").read_bytes())
    (root / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    args["expected_bundle_sha256"] = sha256((root / "manifest.json").read_bytes())
    receipt = json.loads(args["audit_receipt"].read_bytes())
    receipt["bundle_manifest_sha256"] = args["expected_bundle_sha256"]
    args["audit_receipt"].write_bytes(canonical(receipt) + b"\n")
    args["expected_audit_receipt_sha256"] = sha256(args["audit_receipt"].read_bytes())
    with pytest.raises(ValueError, match="actual anchored tiny recipe"):
        package.build_archive(**args)
    assert not args["out"].exists()


def test_existing_valid_timestamp_bytecode_cannot_replace_anchored_source(tmp_path):
    args = inputs(tmp_path)
    frozen = tmp_path / "frozen"
    shutil.copytree(
        SOURCE / "ayaka", frozen / "ayaka", ignore=shutil.ignore_patterns("__pycache__")
    )
    for name in ("LICENSE", "pyproject.toml"):
        shutil.copyfile(SOURCE / name, frozen / name)
    module = frozen / "ayaka/training/direct_audit.py"
    stat = module.stat()
    compiled = compile("raise RuntimeError('BYTECODE SENTINEL EXECUTED')", str(module), "exec")
    data = (
        importlib.util.MAGIC_NUMBER
        + struct.pack("<III", 0, int(stat.st_mtime), stat.st_size)
        + marshal.dumps(compiled)
    )
    bytecode = Path(importlib.util.cache_from_source(str(module)))
    bytecode.parent.mkdir(parents=True, exist_ok=True)
    bytecode.write_bytes(data)
    command = [sys.executable, "-m", "scripts.direct_v2.package", "build", "--allow-tiny"]
    for key, value in {**args, "source_root": frozen}.items():
        if key != "allow_tiny":
            command.extend(["--" + key.replace("_", "-"), str(value)])
    completed = subprocess.run(command, cwd=SOURCE, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["optimizer_steps"] == 0 and not result["gpu_allocated"]
    assert package.verify_archive(args["out"], result["sha256"])["mechanics_only"] is True


def fake_archive(path, damage):
    content = b"small payload"
    anchors = {
        "bundle_manifest_sha256": sha256(content),
        "cpu_audit_sha256": sha256(content),
        "native_snapshot_record_sha256": sha256(content),
    }
    names = ["file.txt", "bundle/manifest.json", "cpu-audit.json", "native-snapshot.json"]
    manifest = {
        "version": package.VERSION,
        "files": {name: {"bytes": len(content), "sha256": sha256(content)} for name in names},
        "production_ready": False,
        "linux_cuda_runtime_included": False,
        "original_test_inputs_included": False,
        "gpu_allocated": False,
        "default_action": "audit",
        "anchors": anchors,
        "mechanics_only": False,
        "audit_command": package._audit_command(anchors, False),
    }
    if damage == "inventory":
        manifest["files"]["file.txt"]["sha256"] = "0" * 64
    if damage == "promoted":
        manifest["production_ready"] = True
    if damage == "command":
        manifest["audit_command"] = ["python", "-c", "raise RuntimeError('execute')"]
    if damage == "execute":
        manifest["audit_command"].append("--execute")
    if damage == "anchor":
        manifest["anchors"]["cpu_audit_sha256"] = "0" * 64
    with (
        path.open("wb") as raw,
        zstandard.ZstdCompressor(write_checksum=True).stream_writer(raw) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        member = tarfile.TarInfo(
            package.PREFIX + ("../outside.txt" if damage == "unsafe" else "file.txt")
        )
        member.size = len(content)
        if damage == "symlink":
            member.type, member.linkname, member.size = tarfile.SYMTYPE, "../../outside", 0
        archive.addfile(member, io.BytesIO(content) if member.isfile() else None)
        for name in names[1:]:
            extra = tarfile.TarInfo(package.PREFIX + name)
            extra.size = len(content)
            archive.addfile(extra, io.BytesIO(content))
        if damage == "duplicate":
            archive.addfile(member, io.BytesIO(content))
        payload = canonical(manifest)
        info = tarfile.TarInfo(package.MANIFEST)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        if damage == "trailing":
            archive.addfile(member, io.BytesIO(content))


@pytest.mark.parametrize(
    "damage",
    [
        "inventory",
        "promoted",
        "unsafe",
        "symlink",
        "duplicate",
        "trailing",
        "command",
        "execute",
        "anchor",
    ],
)
def test_archive_validator_rejects_unbound_or_unsafe_members(tmp_path, damage):
    path = tmp_path / "untrusted.tar.zst"
    fake_archive(path, damage)
    with pytest.raises(ValueError, match="inventory|CPU-only|relative|regular|duplicate|unbound"):
        package.verify_archive(path, package._digest(path))


def test_archive_verifier_checks_external_digest_and_zstd_checksum(tmp_path):
    path = tmp_path / "untrusted.tar.zst"
    fake_archive(path, "none")
    with pytest.raises(ValueError, match="external SHA256"):
        package.verify_archive(path, "0" * 64)
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(zstandard.ZstdError):
        package.verify_archive(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_verification_hashes_the_consumed_stream_without_a_separate_path_read(
    tmp_path, monkeypatch
):
    path = tmp_path / "upload.tar.zst"
    fake_archive(path, "none")
    expected = sha256(path.read_bytes())
    monkeypatch.setattr(
        package, "_digest", lambda *a: pytest.fail("hash the consumed compressed stream")
    )
    manifest = package.verify_archive(path, expected)
    assert manifest["audit_command"][3] == "audit"


@pytest.mark.parametrize("same_frame", [True, False])
def test_tar_eof_cannot_hide_concatenated_unbound_members(tmp_path, same_frame):
    path = tmp_path / "appended.tar.zst"
    fake_archive(path, "none")
    first = zstandard.ZstdDecompressor().decompress(path.read_bytes(), max_output_size=1024 * 1024)
    appended = io.BytesIO()
    with tarfile.open(fileobj=appended, mode="w") as archive:
        member = tarfile.TarInfo(package.PREFIX + "unbound.txt")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    compressor = zstandard.ZstdCompressor(write_checksum=True)
    path.write_bytes(
        compressor.compress(first + appended.getvalue())
        if same_frame
        else path.read_bytes() + compressor.compress(appended.getvalue())
    )
    with pytest.raises(ValueError, match="trailing unbound data"):
        package.verify_archive(path, package._digest(path))
