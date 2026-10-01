import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ayaka.experiments.v2 import (
    TOTAL_SECONDS,
    archive_stale_curriculum,
    bounded_run,
    candidate_config,
    load_manifest,
)


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


def test_recovery_keeps_interrupted_reservation_and_caps_new_stage(tmp_path, monkeypatch):
    import ayaka.experiments.v2 as experiment

    monkeypatch.setattr(experiment, "gpu_info", lambda: {})
    prior = {
        "elapsed_s": 7200,
        "stages": [{"stage": "screen", "allocation_s": 7200, "status": "running"}],
    }
    (tmp_path / "budget.json").write_text(json.dumps(prior))
    monkeypatch.setattr(
        experiment.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )
    recovered = bounded_run(
        "manifest", str(tmp_path), stages=("screen",), stage_limits={"screen": 3600}
    )
    assert recovered["elapsed_s"] == 10800
    assert recovered["stages"][0]["status"] == "interrupted"
    assert recovered["stages"][1]["allocation_s"] == 3600
    with pytest.raises(ValueError, match="stage limits"):
        bounded_run("manifest", str(tmp_path), stage_limits={"screen": 7201})


def test_curriculum_refresh_preserves_old_reports_and_budget(tmp_path):
    (tmp_path / "preparation.json").write_text('{"curriculum_version": 1}')
    (tmp_path / "budget.json").write_text('{"elapsed_s": 10800}')
    (tmp_path / "screen").mkdir()
    (tmp_path / "screen/model.json").write_text('{"n":70}')
    archive_stale_curriculum(tmp_path)
    assert (tmp_path / "interrupted-curriculum-v1/screen/model.json").read_text() == '{"n":70}'
    assert (tmp_path / "budget.json").read_text() == '{"elapsed_s": 10800}'
    assert (tmp_path / "interrupted-curriculum-v1/budget.json").read_bytes() == (
        tmp_path / "budget.json"
    ).read_bytes()


def test_failed_first_sft_candidate_does_not_skip_second(tmp_path, monkeypatch):
    import hashlib

    import torch

    import ayaka.experiments.v2 as experiment
    from ayaka.config import tiny_config
    from ayaka.data.reasoning_v2 import CURRICULUM_VERSION

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    (tmp_path / "preparation.json").write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "curriculum_version": CURRICULUM_VERSION,
            }
        )
    )
    (tmp_path / "selection.json").write_text('{"ranked": ["broken", "working"]}')
    monkeypatch.setattr(experiment, "gpu_info", lambda: {})
    monkeypatch.setattr(
        experiment,
        "load_manifest",
        lambda *args: {"candidates": [{"name": "broken"}, {"name": "working"}]},
    )
    calls, saved = [], []

    def load(candidate):
        calls.append(candidate["name"])
        if candidate["name"] == "broken":
            raise RuntimeError("unsupported training kernel")
        return SimpleNamespace(cfg=tiny_config(readout="lm"), backbone=torch.nn.Linear(1, 1)), None

    class FakeTrainer:
        stopped_early = False

        def __init__(self, *args):
            pass

        def train(self, *args):
            return [{"step": 500}]

    monkeypatch.setattr(experiment, "load_candidate", load)
    monkeypatch.setattr(experiment, "curriculum", lambda *args: [])
    monkeypatch.setattr(experiment, "apply_lora", lambda *args: None)
    monkeypatch.setattr(experiment, "Trainer", FakeTrainer)
    monkeypatch.setattr(experiment, "save_checkpoint", lambda *args: saved.append(args[1]))
    experiment.worker(str(manifest), str(tmp_path), "sft", 60)
    assert calls == ["broken", "working"] and len(saved) == 1
    status = json.loads((tmp_path / "sft_status.json").read_text())
    assert status["status"] == "incomplete" and status["failed_candidates"] == ["broken"]
    assert "unsupported training kernel" in (tmp_path / "sft-broken-failure.json").read_text()
