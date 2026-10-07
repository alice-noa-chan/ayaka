"""Microbatch planning must preserve the full optimizer-batch objective."""

from dataclasses import replace
from types import MethodType

import pytest
import torch

from ayaka.checkpoint import apply_lora
from ayaka.config import tiny_config
from ayaka.data.schema import Candidate, Question, Sample
from ayaka.losses import LossWeights, decision_loss
from ayaka.model.decision import AyakaDecisionModel, DecisionOutput
from ayaka.tokenization import ToyTokenizer
from ayaka.training.batching import collate_items, sample_to_items
from ayaka.training.trainer import TrainConfig, Trainer


def _items():
    def score(qid, width=3):
        return Question(
            qid,
            "score",
            "Rate the evidence",
            [Candidate(f"s{i}", f"level {i}", ordinal=i) for i in range(width)],
            {f"s{i}": float(i == width - 1) for i in range(width)},
        )

    samples = [
        Sample(
            "A long contract with shared conditions and exceptions. " * 30,
            [score("a"), score("b"), score("one")],
        ),
        Sample("Incomplete evidence", [Question.noul("c", "Does it hold?", 1.0)]),
        Sample("Different evidence", [Question.noul("d", "Does it hold?", 0.0)]),
        Sample("Other rating evidence", [score("e")]),
    ]
    tok = ToyTokenizer()
    items = [
        item
        for sample in samples
        for item in sample_to_items(sample, tok, tiny_config(max_seq_len=4096))
    ]
    for index, item in enumerate(items):
        item.flagged = index in (1, 3)
        if index in (0, 3, 5):
            item.teacher = [1 / len(item.target)] * len(item.target)
        if index in (1, 4):
            item.base_probs = [1 / len(item.target)] * len(item.target)
    return tok, items


class LogitModel(torch.nn.Module):
    """Independent learnable logits isolate reductions from transformer noise."""

    def __init__(self, items):
        super().__init__()
        self.backbone = torch.nn.Identity()
        self.offsets = {}
        start = 0
        for item in items:
            self.offsets[id(item)] = list(range(start, start + len(item.target)))
            start += len(item.target)
        self.logits = torch.nn.Parameter(torch.linspace(-2, 3, start, dtype=torch.float64))
        self.pointer = torch.nn.Parameter(torch.linspace(1, -1, start, dtype=torch.float64))


def _trainer(tok, items, weights):
    trainer = Trainer.__new__(Trainer)
    trainer.model = LogitModel(items)
    trainer.cfg = TrainConfig(loss_weights=weights, missing_tau=0.3)
    trainer.tok = tok
    trainer.can_share = True
    trainer.micro_tokens = 100_000
    trainer.micro_ckpt_tokens = 100_000
    trainer.ckpt_threshold = None
    trainer._ckpt_active = False

    def forward(self, kind, chunk):
        tensors = collate_items(chunk, tok.pad_id)
        positions = [position for item in chunk for position in self.model.offsets[id(item)]]
        logits = self.model.logits[positions]
        batch = tensors.batch
        output = DecisionOutput(
            logits,
            logits,
            self.model.pointer[positions],
            batch.cand_cu,
            batch.cand_question,
            batch.primitive,
        )
        return output, tensors

    trainer._forward = MethodType(forward, trainer)
    return trainer


def _full_loss(trainer, items):
    out, tensors = trainer._forward("rows", items)
    return decision_loss(
        out,
        tensors.targets,
        ordinals=tensors.ordinals,
        missing_mask=tensors.flagged,
        teacher_probs=tensors.teacher,
        teacher_mask=tensors.teacher_mask,
        base_probs=tensors.base_probs,
        base_mask=tensors.base_mask,
        weights=trainer.cfg.loss_weights,
        missing_tau=trainer.cfg.missing_tau,
    )


@pytest.mark.parametrize("planning", ["shared", "small_budget", "checkpointed"])
@pytest.mark.parametrize("objective", ["rps", "missing", "legacy_teacher", "anchored_replay"])
def test_full_batch_loss_gradients_and_metrics_do_not_depend_on_chunks(planning, objective):
    tok, items = _items()
    weights = LossWeights(
        nll=1.0,
        brier=0.5,
        pointer_aux=0.3,
        rps=0.35 if objective != "missing" else 0.0,
        missing=0.25 if objective != "rps" else 0.0,
        distill=0.6,
        gold_nll_with_teacher=objective == "anchored_replay",
        base_replay=0.4 if objective == "anchored_replay" else 0.0,
    )
    for item in items:
        item.direct_distillation = weights.gold_nll_with_teacher
        if objective != "anchored_replay":
            item.base_probs = None
        if objective in ("rps", "missing"):
            item.teacher = None
    trainer = _trainer(tok, items, weights)
    if planning == "small_budget":
        trainer.can_share = False
        trainer.micro_tokens = 2048
    elif planning == "checkpointed":
        trainer.ckpt_threshold = 256
        trainer.micro_ckpt_tokens = 2048
    plan = trainer._plan(items)
    assert len(plan) > 1
    if planning == "shared":
        assert any(kind == "shared" for kind, _, _ in plan)

    reference = _full_loss(trainer, items)
    reference["total"].backward()
    expected = [parameter.grad.clone() for parameter in trainer.model.parameters()]
    trainer.model.zero_grad()

    actual = trainer._backward(items)
    for key in reference:
        torch.testing.assert_close(actual[key], reference[key].detach(), atol=1e-10, rtol=1e-7)
    for parameter, gradient in zip(trainer.model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, atol=1e-10, rtol=1e-7)


def test_single_candidate_score_has_zero_rps_without_zero_denominator():
    tok, items = _items()
    items = [items[2], items[3]]
    # The public encoder requires >=2 candidates; exercise loss-level zero
    # eligibility with an explicit handcrafted single-candidate buffer.
    item = items[0]
    item.enc = replace(
        item.enc,
        rendered=replace(
            item.enc.rendered,
            option_spans=item.enc.rendered.option_spans[:1],
            label_ids=item.enc.rendered.label_ids[:1],
            display_order=[0],
        ),
    )
    item.target = [1.0]
    item.ordinals = [0]
    for item in items:
        item.teacher = None
        item.base_probs = None
        item.flagged = False
    trainer = _trainer(tok, items, LossWeights())
    reference = _full_loss(trainer, items)
    actual = trainer._backward(items)
    assert float(actual["rps"]) == 0
    torch.testing.assert_close(actual["total"], reference["total"].detach())
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainer.model.parameters())


def test_actual_tiny_lora_shared_and_checkpointed_gradients_match_full_batch():
    tok, items = _items()
    for item in items:
        item.direct_distillation = True
    torch.manual_seed(17)
    model = AyakaDecisionModel.from_config(
        tiny_config(max_seq_len=4096, lora_dropout=0), dtype=torch.float64
    )
    model.backbone.requires_grad_(False)
    apply_lora(model)
    model.head.double()
    model.gate.data = model.gate.data.double().fill_(0.5)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.01)
    trainer = Trainer(
        model,
        tok,
        TrainConfig(
            steps=1,
            bf16=False,
            micro_batch_tokens=100_000,
            missing_tau=0.3,
            loss_weights=LossWeights(gold_nll_with_teacher=True, base_replay=0.4),
        ),
        "cpu",
    )
    model.train()
    reference = _full_loss(trainer, items)
    reference["total"].backward()
    expected = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert any("lora_B" in name and gradient.abs().sum() > 0 for name, gradient in expected.items())
    heavy_threshold = (
        max(item.length for item in items[3:]) + min(item.length for item in items[:3])
    ) // 2
    assert (
        max(item.length for item in items[3:])
        < heavy_threshold
        < min(item.length for item in items[:3])
    )
    for threshold in (None, heavy_threshold):
        model.zero_grad()
        trainer.ckpt_threshold = threshold
        plan = trainer._plan(items)
        assert len(plan) > 1
        assert any(checkpointed for _, _, checkpointed in plan) == (threshold is not None)
        actual = trainer._backward(items)
        torch.testing.assert_close(
            actual["total"], reference["total"].detach(), rtol=1e-5, atol=1e-7
        )
        for name, parameter in model.named_parameters():
            if name in expected:
                torch.testing.assert_close(parameter.grad, expected[name], rtol=1e-4, atol=2e-7)
