import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.beam_v2 import worker
from scripts.beam_v2.transfer import remote_path


def test_training_allowance_changes_only_time_and_rebinds_recipe(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    recipe = {"completion_target_seconds": 3600, "training": {"lr": 0.00002, "seed": 20261002}}
    worker.write_json(bundle / "training_config.json", recipe)
    previous = worker.digest(bundle / "training_config.json")
    monkeypatch.setattr(worker, "RECIPE_SHA", previous)
    manifest = {
        "files": {"training_config.json": previous, "train.jsonl": "data-digest"},
        "source_sha256": "unchanged-code",
    }
    worker.write_json(bundle / "manifest.json", manifest)
    result = worker.apply_training_allowance(tmp_path)
    new_recipe = json.loads((bundle / "training_config.json").read_text())
    assert new_recipe == {**recipe, "completion_target_seconds": 5400}
    new_manifest = json.loads((bundle / "manifest.json").read_text())
    assert new_manifest["source_sha256"] == "unchanged-code"
    assert new_manifest["files"]["train.jsonl"] == "data-digest"
    assert new_manifest["files"]["training_config.json"] == result["effective_recipe_sha256"]
    with pytest.raises(ValueError, match="different prepared"):
        worker.apply_training_allowance(tmp_path)


def test_result_receipt_detects_corruption_and_excludes_symlinks(tmp_path):
    directory = tmp_path / "result"
    directory.mkdir()
    (directory / "complete.json").write_text('{"complete":true}')
    archive = tmp_path / "receipt.tar.zst"
    receipt = worker.package_results(directory, archive)
    assert receipt["sha256"] == worker.digest(archive)
    manifest = json.loads((directory / "receipt-manifest.json").read_text())
    (directory / "complete.json").write_text('{"complete":false}')
    assert manifest["files"]["complete.json"]["sha256"] != worker.digest(
        directory / "complete.json"
    )


def test_remote_volume_paths_are_posix_on_windows():
    assert remote_path(SimpleNamespace(volume_name="volume", volume_path="a\\b.json")) == (
        "volume/a/b.json"
    )


def test_streamed_subprocess_preserves_diagnostics_and_failure(tmp_path):
    log = tmp_path / "execution.log"
    worker.run_logged([sys.executable, "-c", "print('step 200')"], log, 5)
    assert "step 200" in log.read_text()
    with pytest.raises(subprocess.CalledProcessError):
        worker.run_logged([sys.executable, "-c", "raise RuntimeError('diagnostic')"], log, 5)
    assert "diagnostic" in log.read_text()


def test_streamed_subprocess_enforces_timeout(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        worker.run_logged(
            [sys.executable, "-c", "import time; time.sleep(10)"], tmp_path / "log", 0.05
        )
