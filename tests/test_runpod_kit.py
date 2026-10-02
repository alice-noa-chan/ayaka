import hashlib
import json
from pathlib import Path

import pytest

from scripts.runpod_v2.launch import VERSION, check_records, environment, run_path, train_arguments
from scripts.runpod_v2.package import archive_kit, fingerprint_tree, validate_weights


def test_portable_manifest_rejects_mutated_missing_and_escaping_files(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_bytes(b"dataset\n")
    manifest = {"version": VERSION, "files": fingerprint_tree(tmp_path)}
    check_records(tmp_path, manifest)
    data.write_bytes(b"badset!\n")
    with pytest.raises(ValueError, match="checksum"):
        check_records(tmp_path, manifest)
    data.unlink()
    with pytest.raises(ValueError, match="checksum"):
        check_records(tmp_path, manifest)
    for name in ("../secret", "/secret", "a\\secret"):
        with pytest.raises(ValueError, match="unsafe"):
            check_records(tmp_path, {"version": VERSION, "files": {name: {}}})


def test_archive_roundtrip_has_hash_and_keeps_only_explicit_kit(tmp_path):
    import tarfile

    root = tmp_path / "kit"
    root.mkdir()
    (root / "launch.json").write_text(json.dumps({"steps": 1200}))
    (tmp_path / "do-not-package.key").write_text("private")
    archive = tmp_path / "kit.tar"
    result = archive_kit(root, archive)
    assert result["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive.with_name("kit.tar.sha256").read_text().endswith("  kit.tar\n")
    with tarfile.open(archive) as packed:
        assert packed.getnames() == ["ayaka-v2", "ayaka-v2/launch.json"]
        assert all(row.uid == row.gid == 0 for row in packed)
    with pytest.raises(ValueError, match="new file"):
        archive_kit(root, archive)


def test_pinned_weights_require_complete_native_tokenizer_and_correct_digest(tmp_path):
    from scripts.runpod_v2.package import SNAPSHOT_FILES

    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"native")
    audit = {"shards": {weight.name: {"bytes": 6, "sha256": hashlib.sha256(b"native").hexdigest()}}}
    for name in SNAPSHOT_FILES:
        (tmp_path / name).write_text("metadata")
    validate_weights(tmp_path, audit)
    (tmp_path / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="tokenizer"):
        validate_weights(tmp_path, audit)
    weight.write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        validate_weights(tmp_path, audit)


def test_launch_preserves_full_plan_and_never_requests_downloads():
    config = {"steps": 1200, "max_train_seconds": 14400, "checkpoint_every": 100}
    args = train_arguments(Path("/workspace/ayaka-v2"), config, "v2-main-1200")
    assert args[args.index("--steps") + 1] == "1200"
    assert "--execute" in args and "--allow-weight-downloads" not in args
    env = environment("/workspace/ayaka-v2", 4)
    assert env["HF_HUB_OFFLINE"] == env["HF_DATASETS_OFFLINE"] == "1"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "4"
    for name in ("../other", "", "/other", "a/b", "a\\b"):
        with pytest.raises(ValueError, match="run name"):
            run_path("/workspace/ayaka-v2", name)


def test_restart_and_missing_evaluation_budget_fail_before_audit_or_gpu(tmp_path, monkeypatch):
    from scripts.runpod_v2 import launch

    monkeypatch.setattr(launch, "audit", lambda _: pytest.fail("should reject before runtime work"))
    used = tmp_path / "outputs/v2-main-1200"
    used.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        launch.main(["train", "--root", str(tmp_path)])
    with pytest.raises(ValueError, match="explicit split and time"):
        launch.main(["evaluate", "--root", str(tmp_path), "--split", "dev"])


def test_gpu_check_rejects_wrong_or_multiple_devices_before_training(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from scripts.runpod_v2.launch import require_gpu

    config = {
        "gpu_name_contains": "RTX PRO 6000",
        "minimum_gpu_gib": 90,
        "minimum_free_disk_gib": 0,
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="one CUDA"):
        require_gpu(tmp_path, config)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(name="A100", total_memory=80 * 1024**3),
    )
    with pytest.raises(ValueError, match="allocated hardware"):
        require_gpu(tmp_path, config)


def test_copy_digest_must_match_original_pinned_weight_audit():
    from scripts.runpod_v2.launch import check_pinned_weights

    model = {"backbone": "namespace/mini", "backbone_revision": "abc"}
    audit = {
        "repo": "namespace/mini",
        "revision": "abc",
        "optimizer_steps": 0,
        "shards": {"model.safetensors": {"bytes": 6, "sha256": "original"}},
    }
    path = "hf-cache/hub/models--namespace--mini/snapshots/abc/model.safetensors"
    records = {path: {"bytes": 6, "sha256": "original"}}
    check_pinned_weights(model, audit, records)
    # A self-consistent kit manifest cannot legitimize a corrupted model copy.
    records[path]["sha256"] = "corrupted"
    with pytest.raises(ValueError, match="original pinned audit"):
        check_pinned_weights(model, audit, records)
