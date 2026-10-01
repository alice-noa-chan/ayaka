from dataclasses import replace

import pytest
from test_multimodal import build, media

from ayaka.checkpoint import apply_lora
from ayaka.data.candidate_v2 import candidate_curriculum
from ayaka.data.schema import Question, Sample
from ayaka.training.candidates import proposal_items
from ayaka.training.multimodal import image_items
from ayaka.training.run_v2 import backward_preflight, sample_stream
from ayaka.training.trainer import TrainConfig, Trainer


def test_mixed_backward_preflight_does_not_update_any_optimizer_state():
    _, model, tok, backend = build("gemma4")
    model.cfg = replace(model.cfg, max_seq_len=4096)
    model.backbone.requires_grad_(False)
    apply_lora(model)
    image = Sample("Document", [Question.noul("q", "Visible?", 1)], {"media": media()})
    items = image_items(image, backend, {"q": "The document is visible."})
    items += proposal_items(candidate_curriculum("train", 1)[0], tok, model.cfg)
    trainer = Trainer(model, tok, TrainConfig(bf16=False), "cpu", image_backend=backend)
    result = backward_preflight(trainer, items)
    assert result["optimizer_steps"] == 0 and result["weights_unchanged"]
    assert trainer.step_i == 0 and not trainer.opt.state
    assert all(p.grad is None for p in model.parameters())


def test_lazy_stream_preserves_pending_questions_and_is_deterministic():
    _, model, tok, backend = build("gemma4")
    model.cfg = replace(model.cfg, max_seq_len=4096)
    samples = candidate_curriculum("train", 4)
    a = sample_stream(samples, tok, model.cfg, backend, 3, 42)
    b = sample_stream(samples, tok, model.cfg, backend, 3, 42)
    for _ in range(4):
        aa, bb = next(a), next(b)
        assert len(aa) == 3 and [i.sample_id for i in aa] == [i.sample_id for i in bb]


def test_failed_preflight_clears_gradients_without_update():
    _, model, tok, _ = build("gemma4")
    trainer = Trainer(model, tok, TrainConfig(bf16=False), "cpu")
    with pytest.raises(ValueError, match="missing"):
        backward_preflight(trainer, [])
    assert trainer.step_i == 0 and all(p.grad is None for p in model.parameters())


def test_language_sampler_preserves_english_priority_without_dropping_other_languages():
    import copy
    from collections import Counter

    _, model, tok, backend = build("gemma4")
    model.cfg = replace(model.cfg, max_seq_len=4096)
    samples = []
    for language in ("en", "ko", "ja"):
        sample = copy.deepcopy(candidate_curriculum("train", 1)[0])
        sample.metadata.update(language=language, source_example_id=language)
        samples.append(sample)
    stream = sample_stream(
        samples, tok, model.cfg, backend, 1, 42, {"en": 0.6, "ko": 0.2, "ja": 0.2}
    )
    counts = Counter(next(stream)[0].sample_id for _ in range(120))
    assert counts["en"] > counts["ko"] > 0 and counts["en"] > counts["ja"] > 0
