"""Native CPU model/logit/loss/gradient parity for the opt-in Swift student."""

import copy
from dataclasses import replace
from decimal import Decimal

import pytest
import torch
from test_evidence_swift_bridge import native_model, tokenizer

from ayaka.backbone import detach_text_backbone
from ayaka.config import tiny_config
from ayaka.data.schema import Candidate, Question, Sample
from ayaka.losses import LossWeights, decision_loss
from ayaka.model.decision import AyakaDecisionModel
from ayaka.prompt import render_prefix
from ayaka.swift.prompt import PROMPT_VARIANTS, render_question
from ayaka.swift.readers import HFReader
from ayaka.tokenization import HFTokenizer, ToyTokenizer
from ayaka.training.batching import collate_items, sample_to_items
from ayaka.training.swift_direct import swift_question, swift_sample_to_items
from ayaka.training.trainer import TrainConfig, Trainer


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sample():
    return Sample(
        {"document": "Approved\n승인はい. Apply rule 3, then exception 2."},
        [
            Question.noul("judge", "Approved?", 0.5, "No approval", "Approval granted"),
            Question(
                "choose",
                "choice",
                "Which rule?",
                [Candidate("third", "Rule 3"), Candidate("second", "Rule 2")],
                {"third": 0.7, "second": 0.3},
            ),
            Question(
                "rate",
                "score",
                "Which level?",
                [
                    Candidate("high", "High", 10),
                    Candidate("low", "Low", -2),
                    Candidate("mid", "Mid", 3),
                ],
                {"high": 0.2, "low": 0.3, "mid": 0.5},
            ),
        ],
        {
            "source_example_id": "test-fixture",
            "verified_traces": {"judge": "SECRET TRACE"},
            "teacher_probs": {"judge": [0.1, 0.9]},
            "split": "train",
        },
    )


def setup(family, *, variant="min", state_format="pretty", shared=False):
    tok = HFTokenizer(tokenizer(), "offline-random-fixture")
    lm, text = native_model(family)
    cfg = tiny_config(readout="lm", max_seq_len=2048)
    model = AyakaDecisionModel(cfg, text, text.config)
    trainer = Trainer(
        model,
        tok,
        TrainConfig(
            steps=1,
            bf16=False,
            questions_per_step=3,
            loss_weights=LossWeights(gold_nll_with_teacher=True, pointer_aux=0),
        ),
        "cpu",
    )
    trainer.can_share = shared
    src = sample()
    if shared:
        src.state["pad"] = "Evidence only. " * 25
    items = swift_sample_to_items(src, tok, cfg, prompt_variant=variant, state_format=state_format)
    return tok, lm, trainer, src, items


@pytest.mark.parametrize("family", ["gemma", "granite"])
@pytest.mark.parametrize("variant", PROMPT_VARIANTS)
@pytest.mark.parametrize("state_format", ["pretty", "compact"])
@pytest.mark.parametrize("shared", [False, True])
def test_actual_trainer_matches_swift_reader_and_original_gold_order(
    family, variant, state_format, shared
):
    tok, lm, trainer, src, items = setup(
        family, variant=variant, state_format=state_format, shared=shared
    )
    before = copy.deepcopy(src)
    probabilities, logits = trainer.predict(items, apply_temperature=False, return_logits=True)
    reader = HFReader("offline-random-fixture", device="cpu", dtype="float32")
    reader.model, reader.tokenizer = lm, tok.hf
    for q, item, probs, raw_logits in zip(src.questions, items, probabilities, logits, strict=True):
        _, wire, semantic = swift_question(q)
        messages, mapping = render_question(
            src.state, wire, prompt_variant=variant, state_format=state_format
        )
        actual = reader.read(messages, list(mapping))
        assert item.enc.prefix_ids + item.enc.rendered.suffix_ids == actual.input_token_ids
        displayed = [semantic[label] for label in mapping.values()]
        order = [displayed.index(c.id) for c in q.candidates]
        expected = torch.tensor(list(actual.letter_probs.values()))[order]
        assert torch.allclose(torch.tensor(probs), expected, atol=2e-6, rtol=2e-6)
        expected_logits = torch.tensor(list(actual.letter_log_masses.values()))[order]
        assert torch.allclose(torch.tensor(raw_logits), expected_logits, atol=2e-5, rtol=2e-5)
        assert item.target == [q.target_distribution[c.id] for c in q.candidates]
        assert item.teacher is item.reasoning_labels is item.reasoning_positions is None
    assert items[2].enc.rendered.display_order == [1, 2, 0]
    assert items[0].enc.prefix_ids is items[2].enc.prefix_ids
    assert src == before
    assert {kind for kind, _, _ in trainer._plan(items)} == ({"shared"} if shared else {"rows"})


@pytest.mark.parametrize("family", ["gemma", "granite"])
@pytest.mark.parametrize("shared", [False, True])
def test_trainer_loss_and_all_backbone_gradients_match_native_full_lm(family, shared):
    _, lm, trainer, _, items = setup(family, shared=shared)
    reference = copy.deepcopy(lm)
    trainer.can_share = shared
    parts = trainer.backward_step(items)
    tensors = collate_items(items, trainer.tok.pad_id)
    reference.train()
    full_logits = reference(
        input_ids=tensors.batch.input_ids, attention_mask=tensors.batch.attention_mask
    ).logits
    rows = tensors.batch.cand_question
    positions = tensors.batch.answer_pos[rows]
    label_logits = full_logits[rows, positions, tensors.batch.label_ids]
    out = trainer._forward("rows", items, auxiliary=False)[0]
    native_out = replace(out, logits=label_logits, label_logits=label_logits)
    expected = decision_loss(
        native_out,
        tensors.targets,
        ordinals=tensors.ordinals,
        missing_mask=tensors.flagged,
        weights=trainer.cfg.loss_weights,
    )
    expected["total"].backward()
    assert parts.keys() == expected.keys()
    for key in parts:
        assert torch.allclose(parts[key], expected[key].detach(), atol=3e-6, rtol=3e-6), key
    native_text = detach_text_backbone(reference)
    expected_parameters = dict(native_text.named_parameters())
    for name, parameter in trainer.model.backbone.named_parameters():
        grad = expected_parameters[name].grad
        assert (parameter.grad is None) == (grad is None), name
        if grad is not None:
            assert torch.allclose(parameter.grad, grad, atol=2e-5, rtol=2e-4), name
    assert trainer.step_i == 0 and not trainer.opt.state


@pytest.mark.parametrize(
    "field", ["tokens", "labels", "spans", "gold", "ordinals", "recipe", "candidate_ids", "trace"]
)
def test_changed_rows_rejected_before_any_forward(field, monkeypatch):
    _, _, trainer, _, items = setup("gemma")
    item = items[2]
    if field == "tokens":
        item.enc.rendered.suffix_ids[-1] += 1
    elif field == "labels":
        item.enc.rendered.label_ids.reverse()
    elif field == "spans":
        item.enc.rendered.option_spans.reverse()
    elif field == "gold":
        item.target.reverse()
    elif field == "ordinals":
        item.ordinals.reverse()
    elif field == "recipe":
        item.direct_input_binding["recipe"]["prompt_variant"] = "rules"
    elif field == "candidate_ids":
        item.direct_input_binding["candidate_ids"].reverse()
    else:
        item.reasoning_labels = [1]
    monkeypatch.setattr(trainer.model, "forward", lambda *a, **kw: pytest.fail("no forward"))
    with pytest.raises(ValueError, match="binding changed|must not contain"):
        trainer.predict(items)
    with pytest.raises(ValueError, match="binding changed|must not contain"):
        trainer.backward_step(items)


def test_mixed_encoders_and_recipes_rejected_before_chunks(monkeypatch):
    tok, _, trainer, src, items = setup("gemma")
    legacy = sample_to_items(
        Sample(src.state, [src.questions[1]], src.metadata), tok, trainer.model.cfg
    )
    other = swift_sample_to_items(src, tok, trainer.model.cfg, prompt_variant="rules")
    monkeypatch.setattr(trainer.model, "forward", lambda *a, **kw: pytest.fail("no forward"))
    for combined, message in (
        ([items[0], legacy[0]], "legacy"),
        ([items[0], other[0]], "one exact"),
    ):
        with pytest.raises(ValueError, match=message):
            trainer.predict(combined)
        with pytest.raises(ValueError, match=message):
            trainer.backward_step(combined)


def test_context_no_truncation_and_unsupported_fractional_levels():
    tok = HFTokenizer(tokenizer(), "offline-random-fixture")
    cfg = tiny_config(readout="lm", max_seq_len=64)
    with pytest.raises(ValueError, match="context"):
        swift_sample_to_items(sample(), tok, cfg)
    src = sample()
    src.questions[2].candidates[0].ordinal = Decimal("10.25")
    with pytest.raises(ValueError, match="integers"):
        swift_sample_to_items(src, tok, replace(cfg, max_seq_len=2048))
    with pytest.raises(ValueError, match="LM-only"):
        swift_sample_to_items(sample(), tok, replace(cfg, readout="hybrid"))
    with pytest.raises(ValueError, match="fast offset"):
        swift_sample_to_items(sample(), ToyTokenizer(), replace(cfg, max_seq_len=2048))


def test_default_ayaka_encoder_still_uses_its_existing_prefix_and_no_binding():
    src, tok, cfg = sample(), ToyTokenizer(), tiny_config(max_seq_len=2048)
    items = sample_to_items(src, tok, cfg)
    assert all(item.direct_input_binding is None for item in items)
    assert items[0].enc.prefix_ids == render_prefix(src.state, tok)
