import base64
import copy
import io
from dataclasses import replace

import pytest
import torch
from PIL import Image
from test_multimodal import build, media

from ayaka.checkpoint import apply_lora
from ayaka.data.schema import Candidate, Question, Sample
from ayaka.losses import decision_loss
from ayaka.training.multimodal import image_items
from ayaka.training.trainer import TrainConfig, Trainer


def samples():
    buffer = io.BytesIO()
    Image.new("RGB", (24, 18), color=(20, 60, 90)).save(buffer, format="PNG")
    second = {
        "type": "image",
        "mime_type": "image/png",
        "data": base64.b64encode(buffer.getvalue()).decode(),
    }
    return [
        Sample("First document", [Question.noul("q", "Visible?", 1)], {"media": media()}),
        Sample(
            "Two different documents with more text",
            [
                Question(
                    "q",
                    "choice",
                    "Type?",
                    [Candidate("a", "receipt"), Candidate("b", "letter")],
                    {"a": 1},
                )
            ],
            {"media": media() + [second]},
        ),
        Sample(
            "Third scored document",
            [
                Question(
                    "q",
                    "score",
                    "Level?",
                    [Candidate("a", "level zero", 0), Candidate("b", "level one", 1)],
                    {"b": 1},
                )
            ],
            {"media": media(), "evidence_state": "deleted"},
        ),
    ]


@pytest.mark.parametrize("family", ["gemma4", "gemma4_unified"])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_image_batch_cache_preserves_masked_logits_losses_and_adapter_gradients(
    family, checkpointed, cached
):
    _, model, tok, backend = build(family)
    model.cfg = replace(model.cfg, max_seq_len=4096)
    model.backbone.requires_grad_(False)
    apply_lora(model)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    reference_model, reference_backend = copy.deepcopy((model, backend))
    reference_backend.model = reference_model
    options = {"bf16": False, "grad_checkpointing": checkpointed, "micro_batch_tokens": 8192}
    optimized = Trainer(
        model,
        tok,
        TrainConfig(**options, image_feature_cache_bytes=128 * 1024 * 1024 if cached else 0),
        "cpu",
        image_backend=backend,
    )
    reference = Trainer(
        reference_model,
        tok,
        TrainConfig(**options, image_batch_rows=1, image_feature_cache_bytes=0),
        "cpu",
        image_backend=reference_backend,
    )
    rows = [
        item
        for sample in samples()
        for item in image_items(sample, backend, {"q": "Visible evidence determines the answer."})
    ]
    expected_logits = reference.predict(rows, return_logits=True)[1]
    actual_logits = optimized.predict(rows, return_logits=True)[1]
    for a, b in zip(actual_logits, expected_logits, strict=True):
        torch.testing.assert_close(torch.tensor(a), torch.tensor(b), atol=2e-6, rtol=2e-5)
    assert len(reference._plan(rows)) == 6 and len(optimized._plan(rows)) == 3
    if cached:
        assert optimized.image_features.misses == 2 and optimized.image_features.hits == 4

    def backward(trainer):
        trainer.model.train()
        total = 0
        for kind, group, ckpt in trainer._plan(rows):
            trainer._set_checkpointing(ckpt)
            output, tensors = trainer._forward(kind, group)
            loss = decision_loss(
                output, tensors.targets, ordinals=tensors.ordinals, missing_mask=tensors.flagged
            )["total"]
            loss = loss + trainer.cfg.reasoning_ce_weight * tensors.reasoning_ce
            loss = loss * len(group) / len(rows)
            loss.backward()
            total += loss.detach()
        return total

    expected, actual = backward(reference), backward(optimized)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for a, b in zip(model.parameters(), reference_model.parameters(), strict=True):
        if b.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=3e-4)
            assert (a.grad - b.grad).norm() <= 3e-5 * (1 + b.grad.norm())
    if cached:
        assert optimized.image_features.misses == 2  # LoRA gradients never invalidate frozen vision
        assert optimized.image_features.bytes <= optimized.image_features.max_bytes
        assert all(
            not tensor.requires_grad
            for entry in optimized.image_features.entries.values()
            for tensor in entry
        )


def test_feature_cache_is_bounded_invalidates_frozen_weight_changes_and_restores_method(
    monkeypatch,
):
    _, model, tok, backend = build("gemma4")
    model.cfg = replace(model.cfg, max_seq_len=4096)
    trainer = Trainer(model, tok, TrainConfig(bf16=False), "cpu", image_backend=backend)
    rows = image_items(samples()[0], backend)
    trainer.predict(rows)
    frozen = [
        p
        for p in backend.native.parameters()
        if id(p) not in {id(x) for x in model.backbone.parameters()}
    ]
    with torch.no_grad():
        frozen[0].add_(0)  # version changes, even when the numerical output does not
    trainer.predict(rows)
    assert trainer.image_features.misses == 2
    trainer.image_features.max_bytes = 1
    trainer.image_features.entries.clear()
    trainer.image_features.bytes = 0
    trainer.predict(rows)
    assert trainer.image_features.bytes == 0 and not trainer.image_features.entries

    original = backend.native.get_image_features
    monkeypatch.setattr(
        backend.native,
        "forward",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed native forward")),
    )
    with pytest.raises(RuntimeError, match="failed native"):
        trainer._forward("image", rows)
    assert backend.native.get_image_features == original
    frozen[0].requires_grad_(True)
    with pytest.raises(ValueError, match="frozen vision"):
        trainer._forward("image", rows)
