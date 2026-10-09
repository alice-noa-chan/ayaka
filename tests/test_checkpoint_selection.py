"""Validation-based checkpoint selection and early stopping for direct training."""

import json

import pytest
import torch
from test_run_direct import bundle

from ayaka.training.checkpoint_selection import (
    BEST,
    PROGRESS,
    CheckpointSelector,
    selection_loss,
)
from ayaka.training.run_direct import run_pipeline


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2))
        self.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)


class _Trainer:
    def __init__(self):
        self.model = _Model()


def summary(nll):
    return {"summary": {"by_type": {"noul": {"nll": nll}}, "cc_equal_types": 50.0}}


def run(selector, trainer, losses, steps):
    """Feed one loss per due read; set the weight to the step so restores are visible."""
    losses, stopped = iter(losses), None
    selector.evaluate = lambda _: summary(next(losses))
    for step in range(1, steps + 1):
        with torch.no_grad():
            trainer.model.w.fill_(step)
        if selector(trainer, step, final=step == steps):
            stopped = step
            break
    return stopped


def test_selection_loss_is_the_equal_type_mean_raw_nll():
    by_type = {"choice": {"nll": 0.4}, "noul": {"nll": 0.2}, "score": {"nll": 1.2}}
    assert selection_loss({"by_type": by_type}) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        selection_loss({"by_type": {}})
    with pytest.raises(ValueError):
        selection_loss({"by_type": {"noul": {"nll": float("nan")}}})


def test_best_weights_are_restored_and_patience_stops_the_schedule(tmp_path):
    trainer = _Trainer()
    selector = CheckpointSelector(None, split="router_train", every=2, patience=2, root=tmp_path)
    stopped = run(selector, trainer, [1.0, 0.5, 0.7, 0.6, 0.4], steps=12)
    # Reads at steps 2, 4, 6, 8: best 0.5 at step 4, then two reads without improvement.
    assert stopped == 8
    assert selector.restore(trainer) == {"step": 4, "loss": 0.5}
    assert torch.equal(trainer.model.w, torch.full((2,), 4.0))
    report = selector.report(8)
    assert report["stopped_early"] is True
    assert [h["step"] for h in report["history"]] == [2, 4, 6, 8]
    assert (tmp_path / BEST.format(step=4)).exists()
    assert json.loads((tmp_path / PROGRESS).read_text())["best"]["step"] == 4


def test_final_step_is_always_read_and_never_reported_as_an_early_stop():
    trainer = _Trainer()
    selector = CheckpointSelector(None, split="router_train", every=4, patience=1)
    assert run(selector, trainer, [0.9, 0.8, 0.7], steps=10) is None
    assert [h["step"] for h in selector.history] == [4, 8, 10]
    assert selector.restore(trainer)["step"] == 10


def test_min_delta_requires_a_real_improvement():
    trainer = _Trainer()
    selector = CheckpointSelector(None, split="dev", every=1, patience=1, min_delta=0.05)
    assert run(selector, trainer, [1.0, 0.98], steps=5) == 2
    assert selector.best == {"step": 1, "loss": 1.0}


def test_resume_restores_history_and_the_last_best_before_the_resumed_step(tmp_path):
    trainer = _Trainer()
    first = CheckpointSelector(None, split="router_train", every=2, patience=3, root=tmp_path)
    run(first, trainer, [1.0, 0.5, 0.7, 0.3], steps=8)
    resumed = CheckpointSelector(None, split="router_train", every=2, patience=3)
    resumed.load(tmp_path, resumed_step=6)
    assert [h["step"] for h in resumed.history] == [2, 4, 6]
    assert resumed.best == {"step": 4, "loss": 0.5} and resumed.stale == 1
    fresh = _Trainer()
    resumed.restore(fresh)
    assert torch.equal(fresh.model.w, torch.full((2,), 4.0))
    other = CheckpointSelector(None, split="router_train", every=4, patience=3)
    with pytest.raises(ValueError):
        other.load(tmp_path, resumed_step=6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"every": 0, "patience": 1},
        {"every": 1, "patience": 0},
        {"every": 1, "patience": 1, "min_delta": -1},
    ],
)
def test_invalid_selection_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CheckpointSelector(None, split="dev", **kwargs)


def test_restore_without_any_read_is_an_error():
    with pytest.raises(ValueError):
        CheckpointSelector(None, split="dev", every=1, patience=1).restore(_Trainer())


def test_direct_run_reads_selection_split_and_exports_the_selected_weights(tmp_path):
    root = bundle(tmp_path, steps=4)
    training = {"bf16": False, "log_every": 0, "micro_batch_tokens": 8192}
    completed = run_pipeline(
        root,
        tmp_path / "full",
        action="train",
        mechanics_only=True,
        training=training,
        checkpoint_every=2,
        select_every=2,
        select_patience=1,
    )
    selection = completed["checkpoint_selection"]
    assert selection["split"] == "router_train"
    assert [h["step"] for h in selection["history"]][0] == 2
    assert completed["selected_step"] == selection["best"]["step"]
    assert completed["reload_probability_parity"]
    report = json.loads((tmp_path / "full/checkpoint_selection_report.json").read_bytes())
    assert report == selection
    with pytest.raises(ValueError, match="multiple of the selection interval"):
        run_pipeline(
            root,
            tmp_path / "bad",
            action="train",
            mechanics_only=True,
            training=training,
            checkpoint_every=3,
            select_every=2,
            select_patience=1,
        )
    with pytest.raises(ValueError, match="both an interval and a patience"):
        run_pipeline(root, tmp_path / "bad2", action="train", mechanics_only=True, select_every=2)
