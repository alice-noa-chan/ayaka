from dataclasses import replace

import pytest
import torch
from test_multimodal import build, media

from ayaka.checkpoint import apply_lora
from ayaka.data.schema import Candidate, Question, Sample
from ayaka.losses import decision_loss
from ayaka.training.multimodal import image_items
from ayaka.training.trainer import TrainConfig, Trainer


@pytest.mark.parametrize("family", ["gemma4", "gemma4_unified"])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_native_joint_loss_backpropagates_to_adapter_only_and_preserves_weights(
    family, checkpointed
):
    _, model, tok, backend = build(family)
    model.cfg = replace(model.cfg, max_seq_len=4096)
    model.backbone.requires_grad_(False)
    apply_lora(model)
    sample = Sample(
        "Read the receipt.",
        [Question("q", "choice", "Total?", [Candidate("a", "10"), Candidate("b", "20")], {"a": 1})],
        {"media": media()},
    )
    items = image_items(sample, backend, {"q": "The printed total is 10."})
    trainer = Trainer(
        model,
        tok,
        TrainConfig(bf16=False, grad_checkpointing=checkpointed),
        "cpu",
        image_backend=backend,
    )
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.train()
    assert [kind for kind, _, _ in trainer._plan(items)] == ["image"]
    for item in items:
        trainer._set_checkpointing(checkpointed)
        out, tensors = trainer._forward("image", [item])
        loss = decision_loss(out, tensors.targets)["total"] + tensors.reasoning_ce
        assert torch.isfinite(loss)
        loss.backward()
    adapters = [p.grad for n, p in model.named_parameters() if "lora_B" in n]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in adapters)
    text_ids = {id(p) for p in model.backbone.parameters()}
    assert all(
        p.grad is None and not p.requires_grad
        for p in backend.native.parameters()
        if id(p) not in text_ids
    )
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())  # no optimizer step
    assert all(sum(row) == pytest.approx(1, abs=1e-5) for row in trainer.predict(items))


def test_image_training_rejects_truncation_and_missing_backend():
    _, model, tok, backend = build("gemma4")
    sample = Sample("Document", [Question.noul("q", "Visible?", 1)], {"media": media()})
    model.cfg = replace(model.cfg, max_seq_len=4096)
    items = image_items(sample, backend)
    trainer = Trainer(model, tok, TrainConfig(bf16=False), "cpu")
    with pytest.raises(ValueError, match="image backend"):
        trainer._forward("image", items)
    model.cfg = replace(model.cfg, max_seq_len=32)
    with pytest.raises(ValueError, match="do not truncate"):
        image_items(sample, backend)
