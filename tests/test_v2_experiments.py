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
    shipping_regressions,
)


def test_simple_multilingual_regressions_really_generate_at_forced_high():
    from test_reasoning_pipeline import Generator, Original

    from ayaka.eval.reasoning_v2 import evaluate_efforts
    from ayaka.primitives import DecisionResult
    from ayaka.reasoning_pipeline import ControlledDecision

    original, generator = Original(), Generator()
    original.decide = lambda state, questions, device=None: [
        DecisionResult(q.type, [0.01, 0.99], dict(zip(q.candidates, [0.01, 0.99], strict=True)))
        for q in questions
    ]
    samples = shipping_regressions()
    assert all(not any(c.isdigit() for c in s.state) for s in samples)
    report = evaluate_efforts(ControlledDecision(original, generator), samples, efforts=("high",))
    assert generator.budgets == [1024, 1024, 1024]
    assert all(r["budget"] == 1024 and r["route"] == "reasoned" for r in report["rows"]["high"])
    assert {r["language"] for r in report["rows"]["high"]} == {"en", "ko", "ja"}
    assert all(r["reasoning_tokens"] == 0 for r in report["rows"]["off"])


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
    assert recovered["elapsed_s"] == 7200
    assert recovered["stages"][0]["status"] == "interrupted"
    assert len(recovered["stages"]) == 1  # unverified screen time cannot be released
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


def test_recovery_reuses_partial_pairs_skips_complete_and_ignores_stale_files(
    tmp_path, monkeypatch
):
    import hashlib

    import ayaka.experiments.v2 as experiment
    from ayaka.data.reasoning_v2 import CURRICULUM_VERSION, curriculum
    from ayaka.eval.reasoning_v2 import dataset_signature

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
    full = {
        "name": "complete",
        "status": "complete",
        "n": 96,
        "by_type": {kind: {"n": 32} for kind in ("choice", "noul", "score")},
        "cc_equal_types": 80,
        "proper_loss": 0.1,
        "p95_s": 1,
    }
    (tmp_path / "screen_summary.json").write_text(
        json.dumps([full, {"name": "partial", "status": "incomplete"}])
    )
    (tmp_path / "screen").mkdir()
    cached = {"off": [{"id": "cached"}], "low": []}
    signature = dataset_signature([s for s, _ in curriculum("dev")])
    (tmp_path / "screen/partial.json").write_text(
        json.dumps({"rows": cached, "dataset_signature": signature})
    )
    (tmp_path / "screen/new.json").write_text(
        json.dumps({"rows": {"off": [{"id": "stale"}]}, "dataset_signature": "old-curriculum"})
    )
    candidates = [
        {"name": name, "support": "builtin", "total_parameters": 1}
        for name in ("complete", "partial", "new")
    ]
    monkeypatch.setattr(experiment, "gpu_info", lambda: {})
    monkeypatch.setattr(experiment, "load_manifest", lambda *args: {"candidates": candidates})
    monkeypatch.setattr(experiment, "load_candidate", lambda c: (c["name"], None))
    monkeypatch.setattr(experiment, "controlled_decision", lambda m, t: m)
    seen = []

    def evaluate(name, samples, **kwargs):
        seen.append((name, kwargs["resume_rows"]))
        return {
            "status": "complete",
            "reports": {"low": full},
            "rows": {},
            "dataset_signature": signature,
        }

    monkeypatch.setattr(experiment, "run_efforts", evaluate)
    experiment.worker(str(manifest), str(tmp_path), "recover_screen", 60)
    assert seen == [("partial", cached), ("new", None)]
    assert set(json.loads((tmp_path / "selection.json").read_text())["ranked"]) == {
        "complete",
        "partial",
        "new",
    }


def test_screen_and_recovery_share_caps_and_recovery_cannot_spend_reserve_twice(
    tmp_path, monkeypatch
):
    import ayaka.experiments.v2 as experiment

    monkeypatch.setattr(experiment, "gpu_info", lambda: {})
    monkeypatch.setattr(
        experiment.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )
    ledger = {
        "elapsed_s": 3242,
        "stages": [
            {"stage": "screen", "allocation_s": 7200, "charged_s": 3242, "status": "interrupted"}
        ],
    }
    (tmp_path / "budget.json").write_text(json.dumps(ledger))
    result = bounded_run(
        "manifest", str(tmp_path), stages=("recover_screen", "reserve", "heads", "heads")
    )
    assert result["elapsed_s"] == 3242 + 3600 + 3600
    assert [row["stage"] for row in result["stages"]] == ["screen", "recover_screen", "heads"]
    assert [row["stage"] for row in result["skipped_stages"]] == ["reserve", "heads"]
