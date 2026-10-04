import copy
import json

import pytest
import torch

from ayaka.config import tiny_config
from ayaka.data.reasoning_v2 import SPLITS, curriculum
from ayaka.tokenization import ToyTokenizer
from ayaka.training.direct_bundle import prepare_bundle
from ayaka.training.run_direct import evaluate_direct, mechanics_budget, run_pipeline


def bundle(tmp_path, *, rows=10, steps=2):
    torch.set_num_threads(1)
    splits = {}
    for split in SPLITS:
        splits[split] = [sample for sample, _ in curriculum(split, rows)]
        for sample in splits[split]:
            sample.metadata["source_lineage"] = sample.metadata["case_facts_sha256"]
    cfg = tiny_config(readout="lm", max_seq_len=2048, lora_dropout=0.05)
    root = tmp_path / "bundle"
    prepare_bundle(
        root,
        splits,
        ToyTokenizer(),
        cfg,
        {},
        steps=steps,
        rows_per_step=3,
        seed=19,
        allow_tiny=True,
    )
    return root


def test_audit_allocates_no_weights_or_execution_output(tmp_path, monkeypatch):
    from ayaka.training import run_direct

    root = bundle(tmp_path)

    def no_load(*args, **kwargs):
        raise AssertionError("audit must not load pretrained or random model weights")

    monkeypatch.setattr(run_direct.ElectraDecisionModel, "from_config", no_load)
    result = run_pipeline(root, tmp_path / "absent", action="audit", mechanics_only=True)
    assert result["optimizer_steps"] == 0
    assert not (tmp_path / "absent").exists()


def test_full_credit_rejection_precedes_any_weight_load_or_output(tmp_path, monkeypatch):
    from ayaka.training import run_direct

    root = bundle(tmp_path)
    cfg = mechanics_budget()
    cfg.update(hourly_usd=2, prepaid_usd=0)
    cfg["stages"]["evaluation"]["seconds"] = 60

    def no_load(*args, **kwargs):
        raise AssertionError("insufficient credit must reject before loading model weights")

    monkeypatch.setattr(run_direct.ElectraDecisionModel, "from_config", no_load)
    with pytest.raises(ValueError, match="exceeds prepaid credit"):
        run_pipeline(root, tmp_path / "absent", mechanics_only=True, budget=cfg)
    assert not (tmp_path / "absent").exists()


def test_zero_update_profile_then_complete_train_calibrate_test_export_and_resume(tmp_path):
    root = bundle(tmp_path)
    training = {"bf16": False, "log_every": 0, "micro_batch_tokens": 8192}
    profile = run_pipeline(
        root, tmp_path / "profile", action="profile", mechanics_only=True, training=training
    )
    assert profile["status"] == "profiled_zero_updates" and profile["optimizer_steps"] == 0
    assert not list((tmp_path / "profile").glob("state-*"))
    completed = run_pipeline(
        root,
        tmp_path / "full",
        action="train",
        mechanics_only=True,
        training=training,
        checkpoint_every=1,
    )
    assert completed["optimizer_steps"] == 2 and completed["reload_probability_parity"]
    assert completed["promotable"] is False and completed["test_used_for_selection"] is False
    for split in ("calibration_raw", "dev"):
        result = json.loads((tmp_path / "full" / f"{split}.json").read_bytes())
        assert result["summary"]["reasoning_tokens"] == 0
        assert result["summary"]["official_composite"] is None
        assert all(row["route"] == "direct" for row in result["rows"])
    assert not (tmp_path / "full" / "test.json").exists()
    assert completed["independent_test_required"] is True
    resumed = run_pipeline(
        root,
        tmp_path / "resumed",
        action="train",
        mechanics_only=True,
        training=training,
        resume=tmp_path / "full/state-00000001",
        checkpoint_every=1,
    )
    assert resumed["optimizer_steps"] == 2 and resumed["reload_probability_parity"]
    from safetensors.torch import load_file

    a = load_file(tmp_path / "full/state-00000002/trainable.safetensors")
    b = load_file(tmp_path / "resumed/state-00000002/trainable.safetensors")
    for name in a:
        torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)


def test_calibration_and_test_inputs_cannot_inject_trace_or_teacher_metadata(tmp_path):
    from ayaka.model.electra import ElectraDecisionModel
    from ayaka.training.trainer import Trainer

    sample, _ = curriculum("test", 1)[0]
    sample.metadata["source_lineage"] = sample.metadata["case_facts_sha256"]
    sample.metadata["teacher_probs"] = {"q": [0, 0, 1]}
    sample.metadata["verified_traces"] = {"q": {"text": "Gold answer injected into rationale"}}
    sample.metadata["proposal_supervision"] = {"text": "injected proposal"}
    cfg = tiny_config(readout="lm", max_seq_len=2048)
    model = ElectraDecisionModel.from_config(cfg, dtype=torch.float32)
    from ayaka.training.run_direct import training_config

    recipe = {
        "schedule": {"steps": 1, "rows_per_step": 1, "seed": 0},
    }
    # Derive the actual public loss schema rather than duplicating its fields.
    from dataclasses import asdict

    from ayaka.losses import LossWeights

    recipe["loss_weights"] = asdict(LossWeights(pointer_aux=0, gold_nll_with_teacher=True))
    trainer = Trainer(model, ToyTokenizer(), training_config(recipe), "cpu")
    captured = []
    original = trainer.predict

    def inspect(items):
        captured.extend(copy.deepcopy(items))
        return original(items)

    trainer.predict = inspect
    rows = evaluate_direct(trainer, [sample], "test")["rows"]
    assert rows and all(
        it.reasoning_labels is None and it.proposal_labels is None and it.teacher is None
        for it in captured
    )
    with pytest.raises(ValueError, match="reserved split"):
        evaluate_direct(trainer, [sample], "dev")


def test_unapproved_native_execution_and_tiny_cuda_are_rejected_before_weights(tmp_path):
    root = bundle(tmp_path)
    with pytest.raises(ValueError, match="explicit mechanics-only mode"):
        run_pipeline(root, tmp_path / "absent", mechanics_only=False)
    with pytest.raises(ValueError, match="random tiny model on CPU"):
        run_pipeline(root, tmp_path / "absent", mechanics_only=True, device="cuda")


def test_native_execute_flag_and_prior_paid_time_are_required_before_weight_load(
    tmp_path, monkeypatch
):
    from ayaka.training import run_direct
    from ayaka.training.direct_audit import audit_snapshot

    root = bundle(tmp_path)
    audited = audit_snapshot(root, allow_tiny=True)
    audited.recipe["model"]["backbone"] = "publisher/native"
    audited.recipe["model"]["backbone_revision"] = "a" * 40
    monkeypatch.setattr(run_direct, "audit_snapshot", lambda *args, **kwargs: audited)
    with pytest.raises(ValueError, match="explicit --execute and CUDA"):
        run_pipeline(root, tmp_path / "absent", device="cuda")
    with pytest.raises(ValueError, match="already billed"):
        run_pipeline(root, tmp_path / "absent", device="cuda", execute=True)
    with pytest.raises(ValueError, match="externally pinned"):
        run_pipeline(root, tmp_path / "absent", device="cuda", execute=True, paid_elapsed_seconds=0)
    assert not (tmp_path / "absent").exists()


def test_measured_whole_workflow_rejection_precedes_first_optimizer_step(tmp_path, monkeypatch):
    from ayaka.training import run_direct

    root = bundle(tmp_path)
    cfg = mechanics_budget()
    cfg.update(hourly_usd=1, prepaid_usd=0.01)
    monkeypatch.setattr(
        run_direct,
        "completion_plan",
        lambda *args, **kwargs: {"estimated_remaining_seconds": 10000, "disk_fits": True},
    )

    def no_update(*args, **kwargs):
        raise AssertionError("measured over-budget workflow must not start optimizer updates")

    monkeypatch.setattr(run_direct.Trainer, "train_step", no_update)
    with pytest.raises(ValueError, match="measured complete workflow does not fit"):
        run_pipeline(root, tmp_path / "rejected", action="train", mechanics_only=True, budget=cfg)
    assert not list((tmp_path / "rejected").glob("state-*"))
    assert not (tmp_path / "rejected/checkpoint").exists()
