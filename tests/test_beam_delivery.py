import json

import pytest

from scripts.beam_v2.verify import verify_delivery
from scripts.beam_v2.worker import package_results, write_json


def test_verified_delivery_requires_complete_fixed_training_and_preserves_failed_reports(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    write_json(root / "operational-outcome.json", {"status": "failed", "error": "preflight"})
    archive = tmp_path / "failed.tar.zst"
    receipt = {**package_results(root, archive), "status": "failed"}
    verified = verify_delivery(archive, receipt, tmp_path / "received")
    assert verified["completion"] is None and verified["outcome"]["error"] == "preflight"
    archive.write_bytes(archive.read_bytes()[:-1] + b"!")
    with pytest.raises(ValueError, match="durable receipt"):
        verify_delivery(archive, receipt, tmp_path / "corrupt")


def test_archive_checksums_do_not_turn_partial_training_into_success(tmp_path):
    root = tmp_path / "outputs"
    (root / "recovery").mkdir(parents=True)
    write_json(root / "operational-outcome.json", {"status": "complete"})
    write_json(root / "recovery/complete.json", {"complete": True, "training_steps": 199})
    archive = tmp_path / "partial.tar.zst"
    receipt = {**package_results(root, archive), "status": "complete"}
    with pytest.raises(ValueError, match="200-step"):
        verify_delivery(archive, receipt, tmp_path / "received")
    # All member bytes were valid; the semantic completion gate is independent.
    assert (
        json.loads((tmp_path / "received/result/recovery/complete.json").read_text())[
            "training_steps"
        ]
        == 199
    )
