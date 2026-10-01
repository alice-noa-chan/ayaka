import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ayaka.experiments.v2 import TOTAL_SECONDS, bounded_run, candidate_config, load_manifest


def test_pinned_manifest_enforces_license_total_size_and_lora_targets():
    manifest = load_manifest(str(Path(__file__).parents[1] / "docs/experiments/v2_candidates.json"))
    assert len(manifest["candidates"]) == 12
    by_name = {r["name"]: r for r in manifest["candidates"]}
    assert by_name["gemma4-e4b"]["total_parameters"] > 7e9
    assert "in_proj_qkv" in candidate_config(by_name["qwen35-9b"]).lora_targets
    assert "qkv_proj" in candidate_config(by_name["phi4-mini"]).lora_targets
    assert candidate_config(by_name["granite42-8b"]).version == 2


def test_deadline_budget_is_reserved_before_subprocess_and_never_reset(tmp_path, monkeypatch):
    import ayaka.experiments.v2 as experiment

    monkeypatch.setattr(experiment, "gpu_info", lambda: {})

    def run(command, timeout, check):
        ledger = json.loads((tmp_path / "budget.json").read_text())
        assert ledger["elapsed_s"] >= timeout
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(experiment.subprocess, "run", run)
    first = bounded_run("manifest", str(tmp_path), stages=("screen",), scale=0.01)
    second = bounded_run("manifest", str(tmp_path), stages=("heads",), scale=0.01)
    assert second["elapsed_s"] > first["elapsed_s"]
    assert len(second["stages"]) == 2 and second["elapsed_s"] <= TOTAL_SECONDS
    with pytest.raises(ValueError, match="scale"):
        bounded_run("manifest", str(tmp_path), scale=2)
