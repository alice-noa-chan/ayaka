import itertools
import json

import pytest
from test_training_throughput import setup

from ayaka.training.workload import describe_rows
from benchmark_gpu_v2 import matched_profile, prepare_batch


def test_selected_batch_preserves_order_duplicates_and_prepares_each_source_once():
    calls = []

    def prepare(sample):
        calls.append(sample)
        return [sample + "0", sample + "1"]

    assert prepare_batch(["a", "b"], [(0, 1), (1, 0), (0, 0), (0, 1)], prepare) == [
        "a1",
        "b0",
        "a0",
        "a1",
    ]
    assert calls == ["a", "b"]


def test_matched_profile_is_cold_deterministic_and_leaves_live_training_untouched(tmp_path):
    trainer, rows, sample, _ = setup()
    trainer.cfg.questions_per_step = 2
    result = matched_profile(
        trainer,
        [sample],
        [describe_rows(sample, rows)],
        lambda _: rows,
        tmp_path / "profile",
        steps=2,
        seed=7,
    )
    assert result["status"] == "complete" and result["weights_unchanged"]
    assert result["selected_steps"] == [0, 1] and result["optimizer_steps"] == 0
    assert all(
        row["warmup_batches"] == 1 and row["measured_batches"] == 2 and row["cold_image_features"]
        for row in result["batches"]
    )
    assert trainer.step_i == 0 and not trainer.opt.state
    assert all(parameter.grad is None for parameter in trainer.model.parameters())
    assert json.loads((tmp_path / "profile/comparison.json").read_text())["status"] == "complete"


def test_failed_profile_preserves_completed_batches_and_does_not_claim_completion(
    tmp_path, monkeypatch
):
    import benchmark_gpu_v2

    trainer, rows, sample, _ = setup()
    trainer.cfg.questions_per_step = 2
    original = benchmark_gpu_v2.profile_backward
    iterations = itertools.count()

    def fail_second(*args, **kwargs):
        if next(iterations) == 1:
            raise ValueError("simulated incompatible kernel")
        return original(*args, **kwargs)

    monkeypatch.setattr(benchmark_gpu_v2, "profile_backward", fail_second)
    with pytest.raises(ValueError, match="incompatible"):
        matched_profile(
            trainer,
            [sample],
            [describe_rows(sample, rows)],
            lambda _: rows,
            tmp_path / "profile",
            steps=2,
            seed=7,
        )
    saved = json.loads((tmp_path / "profile/comparison.json").read_text())
    assert saved["status"] == "running" and len(saved["batches"]) == 1
    assert saved["optimizer_steps"] == 0 and not trainer.opt.state
