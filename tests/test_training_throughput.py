from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_multimodal import build, media

from ayaka.checkpoint import apply_lora
from ayaka.data.schema import Question, Sample
from ayaka.training.multimodal import image_items
from ayaka.training.throughput import (
    completion_plan,
    profile_backward,
    profile_overheads,
    profile_production,
)
from ayaka.training.trainer import TrainConfig, Trainer


def setup():
    _, model, tok, backend = build("gemma4")
    model.cfg = replace(model.cfg, max_seq_len=4096)
    model.backbone.requires_grad_(False)
    apply_lora(model)
    sample = Sample(
        "Document",
        [Question.noul("q", "Visible?", 1)],
        {
            "media": media(),
            "task_family": "document",
            "language": "en",
            "source_lineage": "test-document",
        },
    )
    trainer = Trainer(model, tok, TrainConfig(bf16=False), "cpu", image_backend=backend)
    return trainer, image_items(sample, backend, {"q": "The document is visible."}), sample, backend


def test_production_backward_profile_preserves_weights_optimizer_and_rng():
    import itertools

    trainer, items, _, _ = setup()
    rng = torch.get_rng_state().clone()
    report = profile_backward(trainer, itertools.repeat(items), warmup=1, repeats=2)
    assert report["weights_unchanged"] and report["optimizer_steps"] == 0
    assert report["measured_batches"] == 2 and report["median_seconds"] > 0
    assert all(batch["rows"] == 2 and batch["image_rows"] == 2 for batch in report["batches"])
    assert torch.equal(rng, torch.get_rng_state()) and not trainer.opt.state
    assert all(parameter.grad is None for parameter in trainer.model.parameters())


def test_completion_plan_accounts_for_every_step_and_rejects_insufficient_time():
    plan = completion_plan({"max_seconds": 8}, 1200, 14400)
    assert plan["fits"] and plan["planned_steps"] == 1200
    assert plan["estimated_remaining_seconds"] == 13620
    assert not completion_plan({"max_seconds": 15}, 1200, 14400)["fits"]
    with pytest.raises(ValueError, match="finite"):
        completion_plan({"max_seconds": float("inf")}, 1200, 14400)


def test_backward_profile_uses_oom_fallback_without_optimizer_updates(monkeypatch):
    import itertools

    trainer, items, _, _ = setup()
    original = trainer._backward
    calls = []

    def first_oom(rows):
        calls.append(1)
        if len(calls) == 1:
            trainer.model.gate.grad = torch.ones_like(trainer.model.gate)
            raise torch.cuda.OutOfMemoryError("simulated forward OOM")
        assert trainer.model.gate.grad is None
        return original(rows)

    monkeypatch.setattr(trainer, "_backward", first_oom)
    result = profile_backward(trainer, itertools.repeat(items), warmup=0, repeats=1)
    assert result["optimizer_steps"] == 0 and trainer.ckpt_threshold == 1024
    assert len(calls) == 2 and not trainer.opt.state


@pytest.mark.parametrize("profile_only", [False, True])
def test_runner_profiles_or_refuses_an_unfinishable_schedule_before_optimizer(
    tmp_path, monkeypatch, profile_only
):
    from ayaka.training import run_v2

    trainer, _, sample, backend = setup()
    monkeypatch.setattr(
        run_v2, "fresh_image_backend", lambda *args, **kwargs: (trainer.model, trainer.tok, backend)
    )
    monkeypatch.setattr(
        run_v2.Trainer, "train", lambda *args, **kwargs: pytest.fail("started optimization")
    )
    args = SimpleNamespace(
        device="cpu",
        allow_weight_downloads=False,
        steps=1,
        max_train_seconds=1,
        out=str(tmp_path / "run"),
        backward_only=False,
        profile_only=profile_only,
    )
    recipe = {
        "training": {"seed": 7, "bf16": False, "questions_per_step": 2},
        "language_sampling": {"en": 1},
    }
    if profile_only:
        result = run_v2.execute_training(args, trainer.model.cfg, recipe, {"train": [sample]})
        assert result["optimizer_steps"] == 0 and result["status"] == "throughput_profile_only"
        assert result["completion_plan"]["available_remaining_seconds"] > args.max_train_seconds
    else:
        with pytest.raises(ValueError, match="no optimizer steps"):
            run_v2.execute_training(args, trainer.model.cfg, recipe, {"train": [sample]})
    assert (tmp_path / "run" / "throughput.json").is_file()
    assert not (tmp_path / "run" / "checkpoint").exists()


def test_admitted_schedule_completes_every_step_and_marks_final_checkpoint(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from ayaka.training import run_v2

    trainer, _, sample, backend = setup()
    monkeypatch.setattr(
        run_v2, "fresh_image_backend", lambda *args, **kwargs: (trainer.model, trainer.tok, backend)
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")

    def save(model, path, meta):
        Path(path).mkdir(parents=True)

    monkeypatch.setattr(run_v2, "save_checkpoint", save)
    original = run_v2.Trainer.train

    def train(self, *args, **kwargs):
        assert self.cfg.max_train_seconds == 0  # completion is the fixed schedule
        return original(self, *args, **kwargs)

    monkeypatch.setattr(run_v2.Trainer, "train", train)
    args = SimpleNamespace(
        device="cpu",
        allow_weight_downloads=False,
        steps=2,
        max_train_seconds=600,
        out=str(tmp_path / "run"),
        backward_only=False,
        profile_only=False,
        checkpoint_every=100,
        bundle=str(bundle),
    )
    recipe = {
        "training": {"seed": 7, "bf16": False, "questions_per_step": 2},
        "language_sampling": {"en": 1},
        "initialization": "fresh_lora_from_pinned_base",
    }
    result = run_v2.execute_training(args, trainer.model.cfg, recipe, {"train": [sample]})
    assert result["complete"] and result["steps"] == 2 and not result["stopped_early"]
    marker = json.loads((tmp_path / "run" / "checkpoint" / "complete.json").read_text())
    assert marker == {"steps": 2, "complete": True}


def test_measured_optimizer_and_io_profiles_never_touch_live_weights_state_or_rng(tmp_path):
    from ayaka.training.run_v2 import trainable_digest

    trainer, _, _, _ = setup()
    before, rng = trainable_digest(trainer.model), torch.get_rng_state().clone()
    report = profile_overheads(trainer, tmp_path / "probe", repeats=2)
    assert report["checkpoint_bytes"] > 0 and report["checkpoint_seconds"] > 0
    assert report["max_optimizer_seconds"] > 0 and report["disposable_optimizer_steps"] == 3
    assert report["model_optimizer_steps"] == 0 and not report["probe_is_trained_checkpoint"]
    assert (
        trainable_digest(trainer.model) == before and not trainer.opt.state and trainer.step_i == 0
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert all(parameter.grad is None for parameter in trainer.model.parameters())
    assert not (tmp_path / "probe" / "untrained_io_probe" / "complete.json").exists()


def test_measured_completion_counts_all_writes_and_rejects_disk_shortfall():
    overheads = {
        "max_optimizer_seconds": 0.5,
        "checkpoint_seconds": 10,
        "checkpoint_bytes": 1000,
        "disk_free_bytes": 1_000_000,
    }
    plan = completion_plan(
        {"max_seconds": 8}, 1200, 14400, overheads=overheads, checkpoint_every=100
    )
    assert plan["checkpoint_writes"] == 14 and plan["save_seconds_reserved"] == 140
    assert plan["estimated_remaining_seconds"] == (8.5 * 1200 + 140) * 1.25
    assert plan["overheads_measured"] and plan["fits"]
    daily = completion_plan({"max_seconds": 1}, 2, 1000, overheads=overheads, checkpoint_every=1)
    assert daily["checkpoint_writes"] == 3
    no_disk = completion_plan(
        {"max_seconds": 1}, 2, 1000, overheads={**overheads, "disk_free_bytes": 1}
    )
    assert not no_disk["fits"] and not no_disk["disk_fits"]


def test_schedule_forecast_retains_stress_bound_and_refuses_different_step_count():
    profile = {"max_seconds": 18, "schedule": {"planned_steps": 1200, "max_seconds": 6}}
    result = completion_plan(profile, 1200, 14400)
    assert result["fits"] and result["forecast_backward_seconds_per_step"] == 6
    assert result["all_stress_seconds_estimate"] > 14400
    assert result["observed_max_backward_seconds"] == 18
    with pytest.raises(ValueError, match="match"):
        completion_plan(profile, 1199, 14400)
    with pytest.raises(ValueError, match="finite"):
        completion_plan(
            {**profile, "schedule": {"planned_steps": 1200, "max_seconds": float("nan")}},
            1200,
            14400,
        )


def test_completion_charges_optimizer_allocation_once_and_every_warm_step():
    overheads = {
        "optimizer_seconds": [0.1, 0.005, 0.006, 0.004],
        "max_optimizer_seconds": 0.1,
        "checkpoint_seconds": 1,
        "checkpoint_bytes": 1000,
        "disk_free_bytes": 1_000_000,
    }
    plan = completion_plan({"max_seconds": 6}, 1200, 14400, overheads=overheads)
    total = 0.1 + 1199 * 0.006
    assert plan["optimizer_cold_start_seconds"] == 0.1
    assert plan["optimizer_seconds_allowance_per_step"] == 0.006
    assert plan["optimizer_total_seconds_reserved"] == total
    assert plan["estimated_remaining_seconds"] == (6 * 1200 + total + 14) * 1.25
    single = completion_plan({"max_seconds": 6}, 1, 100, overheads=overheads)
    assert single["optimizer_total_seconds_reserved"] == 0.1
    with pytest.raises(ValueError, match="finite"):
        completion_plan(
            {"max_seconds": 6},
            1200,
            14400,
            overheads={**overheads, "optimizer_seconds": [0.1, float("nan")]},
        )


def test_production_profile_covers_cold_image_and_trace_strata_without_gradients():
    import itertools

    from ayaka.training.workload import describe_rows

    trainer, rows, sample, _ = setup()
    trainer.cfg.questions_per_step = 2
    report = profile_production(
        trainer,
        itertools.repeat(rows),
        [sample],
        [describe_rows(sample, rows)],
        lambda _: rows,
        steps=2,
        seed=7,
    )
    assert report["measured_batches"] == 6
    assert len(report["stress"]) == 2
    assert all(row["cold_image_features"] and row["weights_unchanged"] for row in report["stress"])
    assert report["max_seconds"] >= max(row["max_seconds"] for row in report["stress"])
    assert trainer.step_i == 0 and not trainer.opt.state
    assert [row["step_index"] for row in report["schedule"]["batches"]] == [0, 1]
    assert report["schedule"]["planned_steps"] == 2
    assert all(row["cold_image_features"] for row in report["schedule"]["batches"])


def test_profile_rejects_finite_losses_with_nonfinite_gradients_before_updates(monkeypatch):
    import itertools

    trainer, rows, _, _ = setup()

    def bad_gradients(_):
        trainer.model.gate.grad = torch.full_like(trainer.model.gate, float("nan"))
        return {"total": torch.tensor(1.0)}

    monkeypatch.setattr(trainer, "backward_step", bad_gradients)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="nonfinite gradients"):
        profile_backward(trainer, itertools.repeat(rows), warmup=0, repeats=1)
    assert trainer.step_i == 0 and not trainer.opt.state
    assert torch.equal(rng, torch.get_rng_state())
    assert all(parameter.grad is None for parameter in trainer.model.parameters())
