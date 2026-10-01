from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_multimodal import build, media

from ayaka.checkpoint import apply_lora
from ayaka.data.schema import Question, Sample
from ayaka.training.multimodal import image_items
from ayaka.training.throughput import completion_plan, profile_backward
from ayaka.training.trainer import TrainConfig, Trainer


def setup():
    _, model, tok, backend = build("gemma4")
    model.cfg = replace(model.cfg, max_seq_len=4096)
    model.backbone.requires_grad_(False)
    apply_lora(model)
    sample = Sample(
        "Document",
        [Question.noul("q", "Visible?", 1)],
        {"media": media(), "task_family": "document", "language": "en"},
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
