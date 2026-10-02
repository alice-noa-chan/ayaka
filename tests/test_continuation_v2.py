from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_multimodal import build

from ayaka.checkpoint import apply_lora, save_checkpoint
from ayaka.multimodal import ImageDecision
from ayaka.reasoning_pipeline import controlled_decision
from ayaka.training.run_v2 import continuation_image_backend
from ayaka.training.scoped_calibration import checkpoint_fingerprint


def test_continuation_preserves_trained_parameters_and_freezes_native_vision(tmp_path, monkeypatch):
    from ayaka import multimodal

    _, model, tok, backend = build("gemma4")
    model.backbone.requires_grad_(False)
    apply_lora(model)
    native = backend.native
    native.language_model = model.backbone
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.01)
        model.gate.fill_(0.7)
    root = tmp_path / "v1"
    save_checkpoint(model, str(root))
    fingerprint = checkpoint_fingerprint(root)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    decision = ImageDecision(controlled_decision(model, tok), backend)
    monkeypatch.setattr(multimodal, "load_image_decision", lambda *a, **kw: decision)
    cfg = replace(model.cfg, version=2, name="continued")
    resumed, _, loaded_backend = continuation_image_backend(root, cfg, "cpu", fingerprint)
    assert resumed.cfg == cfg and loaded_backend.processing_device == "cpu"
    assert all(
        torch.equal(before[name], parameter) for name, parameter in resumed.named_parameters()
    )
    assert any(
        parameter.requires_grad
        for name, parameter in resumed.named_parameters()
        if "lora_B" in name
    )
    model_ids = {id(p) for p in resumed.parameters()}
    assert all(not p.requires_grad for p in native.parameters() if id(p) not in model_ids)
    with pytest.raises(ValueError, match="architecture"):
        continuation_image_backend(root, replace(cfg, lora_r=cfg.lora_r + 1), "cpu", fingerprint)
    with pytest.raises(ValueError, match="identity"):
        continuation_image_backend(root, cfg, "cpu", "f" * 64)


def test_continuation_execution_uses_parent_without_fresh_initialization(tmp_path, monkeypatch):
    from ayaka.data.schema import Question, Sample
    from ayaka.training import run_v2

    _, model, tok, backend = build("gemma4")
    model.backbone.requires_grad_(False)
    apply_lora(model)
    sample = Sample(
        "A blue flag", [Question.noul("q", "Blue?", 1)], {"task_family": "flag", "language": "en"}
    )
    monkeypatch.setattr(run_v2, "continuation_image_backend", lambda *a: (model, tok, backend))
    monkeypatch.setattr(
        run_v2, "fresh_image_backend", lambda *a, **kw: pytest.fail("reinitialized parent")
    )
    args = SimpleNamespace(
        device="cpu",
        init_checkpoint="parent",
        steps=1,
        max_train_seconds=30,
        out=str(tmp_path / "preflight"),
        backward_only=True,
    )
    result = run_v2.execute_training(
        args,
        model.cfg,
        {
            "initialization": "checkpoint_continuation",
            "initial_checkpoint_sha256": "a" * 64,
            "training": {"seed": 7, "bf16": False},
        },
        {"train": [sample]},
    )
    assert result["optimizer_steps"] == 0 and result["weights_unchanged"]
