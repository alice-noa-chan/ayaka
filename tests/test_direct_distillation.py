import copy
import math
from dataclasses import replace

import pytest
import torch

from ayaka.checkpoint import apply_lora
from ayaka.config import tiny_config
from ayaka.data.reasoning_v2 import SPLITS
from ayaka.data.schema import Candidate, Question, Sample
from ayaka.eval.read_artifact import fingerprint
from ayaka.losses import LossWeights
from ayaka.model.electra import ElectraDecisionModel
from ayaka.tokenization import ToyTokenizer
from ayaka.training.direct_distillation import make_teacher_read, prepare_direct_distillation
from ayaka.training.trainer import TrainConfig, Trainer


def dataset():
    return {
        split: [
            Sample(
                {"answer": "x", "count": 2, "insufficient": True, "document": split},
                [
                    Question(
                        "pick",
                        "choice",
                        "Select the answer",
                        [Candidate("x", "x"), Candidate("y", "y")],
                        {"x": 1.0, "y": 0.0},
                    ),
                    Question.noul("known", "Is the unknown fact true?", 0.5),
                    Question(
                        "rate",
                        "score",
                        "Select the count",
                        [
                            Candidate("one", "1", 1),
                            Candidate("two", "2", 2),
                            Candidate("three", "3", 3),
                        ],
                        {"one": 0.0, "two": 1.0, "three": 0.0},
                    ),
                ],
                {
                    "split": split,
                    "source_example_id": split,
                    "source_lineage": f"lineage/{split}",
                    "generator_template_id": f"template/{split}",
                    "rule_combination": f"rules/{split}",
                    "document_voice": f"voice/{split}",
                    "license": "MIT",
                    "verified_traces": {"pick": "SECRET_TEACHER_TRACE"},
                },
            )
        ]
        for split in SPLITS
    }


def verify(sample, q):
    # Derive labels from evidence instead of reading target_distribution.
    if q.type == "noul":
        assert sample.state["insufficient"]
        return {c.id: 0.5 for c in q.candidates}
    if q.type == "choice":
        return {c.id: float(c.id == sample.state["answer"]) for c in q.candidates}
    return {c.id: float(c.ordinal == sample.state["count"]) for c in q.candidates}


def reads(splits):
    sample = splits["train"][0]
    probabilities = [
        ([0.9, 0.1], [0.1, 0.9]),
        ([0.5, 0.5], [0.3, 0.7]),
        ([0.1, 0.8, 0.1], [0.6, 0.3, 0.1]),
    ]
    return {
        f"train/{q.id}": make_teacher_read(
            sample,
            q,
            probs=p,
            direct_probs=d,
            model_sha256="a" * 64,
            recipe_sha256="b" * 64,
            trace_sha256=fingerprint("completed trace"),
            generated_tokens=8,
        )
        for q, (p, d) in zip(sample.questions, probabilities, strict=True)
    }


def test_direct_items_use_original_input_preserve_soft_gold_and_do_not_mutate():
    splits = dataset()
    before = copy.deepcopy(splits)
    teacher = reads(splits)
    teacher_before = copy.deepcopy(teacher)

    class LoggedTokenizer(ToyTokenizer):
        texts = []

        def encode(self, text):
            self.texts.append(text)
            return super().encode(text)

    tok = LoggedTokenizer()
    items, report = prepare_direct_distillation(
        splits, tok, tiny_config(max_seq_len=2048), teacher, verify
    )
    assert len(items) == report["accepted_teacher_questions"] == 3
    assert items[1].target == items[1].teacher == [0.5, 0.5]
    assert all(it.reasoning_positions is None and it.reasoning_labels is None for it in items)
    assert not any("SECRET_TEACHER_TRACE" in text for text in tok.texts)
    assert splits == before and teacher == teacher_before
    assert report["reasoning_training_tokens"] == 0
    assert report["direct_training_tokens"] == sum(it.length for it in items)
    assert set(report["split_sha256"]) == set(SPLITS)
    assert report["promotable"] is report["execution_attested"] is False


def test_rejected_teachers_leave_gold_replay_and_score_uses_continuous_metrics():
    splits = dataset()
    teacher = reads(splits)
    teacher["train/pick"]["probs"] = [0.4, 0.6]  # better NLL, still wrong
    teacher["train/known"]["direct_probs"] = [0.1, 0.9]
    teacher["train/known"]["probs"] = [0.3, 0.7]  # better NLL, now abstains
    teacher["train/rate"]["direct_probs"] = [0.45, 0.1, 0.45]
    teacher["train/rate"]["probs"] = [0.2, 0.5, 0.3]  # better NLL, worse expected position
    items, report = prepare_direct_distillation(
        splits, ToyTokenizer(), tiny_config(max_seq_len=2048), teacher, verify
    )
    assert all(it.teacher is None for it in items)
    assert report["gold_replay_questions"] == 3
    assert "wrong_or_abstaining_hard_gold" in report["items"][0]["reasons"]
    assert "thresholded_credit_regression" in report["items"][1]["reasons"]
    assert "score_nmae_regression" in report["items"][2]["reasons"]


@pytest.mark.parametrize("change", ["state", "question", "candidate", "order", "gold", "lineage"])
def test_teacher_binding_rejects_semantic_and_candidate_changes(change):
    splits = dataset()
    teacher = reads(splits)
    sample = splits["train"][0]
    q = sample.questions[0]
    if change == "state":
        sample.state["document"] = "different evidence"
    elif change == "question":
        q.instruction += " changed"
    elif change == "candidate":
        q.candidates[0].description += " changed"
    elif change == "order":
        q.candidates.reverse()
    elif change == "gold":
        sample.state["answer"] = "y"
        q.target_distribution = {"x": 0, "y": 1}
    else:
        sample.metadata["source_lineage"] += " changed"
    with pytest.raises(ValueError, match="binding mismatch"):
        prepare_direct_distillation(
            splits, ToyTokenizer(), tiny_config(max_seq_len=2048), teacher, verify
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("generated_tokens", 0),
        ("generated_tokens", 1025),
        ("finish_reason", "length"),
        ("route", "fallback"),
        ("model_sha256", "latest"),
        ("execution_attested", True),
        ("probs", [float("nan"), 0]),
    ],
)
def test_rejects_incomplete_invalid_or_falsely_attested_reads(field, value):
    splits = dataset()
    teacher = reads(splits)
    teacher["train/pick"][field] = value
    with pytest.raises(ValueError):
        prepare_direct_distillation(
            splits, ToyTokenizer(), tiny_config(max_seq_len=2048), teacher, verify
        )


def test_rejects_holdout_teacher_and_cross_split_lineage_before_tokenization():
    splits = dataset()
    teacher = reads(splits)
    teacher["test/pick"] = teacher["train/pick"]
    with pytest.raises(ValueError, match="train cohort"):
        prepare_direct_distillation(splits, None, tiny_config(), teacher, verify)
    teacher.pop("test/pick")
    splits["test"][0].metadata["source_lineage"] = splits["train"][0].metadata["source_lineage"]
    with pytest.raises(ValueError, match="cross-split"):
        prepare_direct_distillation(splits, None, tiny_config(), teacher, verify)
    test = dataset()["test"][0]
    with pytest.raises(ValueError, match="train"):
        make_teacher_read(
            test,
            test.questions[0],
            probs=[0.9, 0.1],
            direct_probs=[0.1, 0.9],
            model_sha256="a" * 64,
            recipe_sha256="b" * 64,
            trace_sha256="c" * 64,
            generated_tokens=8,
        )


def test_requires_independent_gold_and_positive_gold_loss_and_rejects_overflow():
    splits = dataset()
    cfg = tiny_config(max_seq_len=2048)
    with pytest.raises(ValueError, match="gold disagrees"):
        prepare_direct_distillation(
            splits,
            ToyTokenizer(),
            cfg,
            {},
            lambda s, q: {c.id: 1 / len(q.candidates) for c in q.candidates},
        )
    with pytest.raises(ValueError, match="positive gold"):
        prepare_direct_distillation(splits, None, cfg, {}, verify, weights=LossWeights())
    with pytest.raises(ValueError, match="refuse truncation"):
        prepare_direct_distillation(splits, ToyTokenizer(), replace(cfg, max_seq_len=8), {}, verify)


def test_noul_reordered_teacher_is_canonicalized_before_direct_training():
    splits = dataset()
    sample = splits["train"][0]
    q = sample.questions[1]
    q.candidates.reverse()
    q.target_distribution = {"false": 0, "true": 1}
    sample.state["insufficient"] = False
    teacher = {
        "train/known": make_teacher_read(
            sample,
            q,
            probs=[0.9, 0.1],
            direct_probs=[0.2, 0.8],
            model_sha256="a" * 64,
            recipe_sha256="b" * 64,
            trace_sha256="c" * 64,
            generated_tokens=8,
        )
    }

    def known(sample, question):
        return {"false": 0, "true": 1} if question.type == "noul" else verify(sample, question)

    items, _ = prepare_direct_distillation(
        splits, ToyTokenizer(), tiny_config(max_seq_len=2048), teacher, known
    )
    assert teacher["train/known"]["candidate_ids"] == ["false", "true"]
    assert items[1].target == [0, 1] and items[1].teacher == [0.1, 0.9]


def test_prepared_items_reach_real_tiny_lora_backward_without_trace_ce():
    torch.manual_seed(7)
    splits = dataset()
    cfg = tiny_config(max_seq_len=2048)
    tok = ToyTokenizer()
    w = LossWeights(gold_nll_with_teacher=True, distill=0.2)
    items, _ = prepare_direct_distillation(splits, tok, cfg, reads(splits), verify, weights=w)
    model = ElectraDecisionModel.from_config(cfg, dtype=torch.float32)
    model.backbone.requires_grad_(False)
    apply_lora(model)
    trainer = Trainer(
        model,
        tok,
        TrainConfig(steps=1, micro_batch_tokens=8192, loss_weights=w, log_every=0),
        "cpu",
    )
    parts = trainer.train_step(items)
    assert "kl" in parts and "reasoning_ce" not in parts
    assert all(math.isfinite(parts[name]) for name in ("total", "nll", "kl", "rps"))


@pytest.mark.parametrize("operation", ["train_step", "backward_step"])
@pytest.mark.parametrize("weights", [LossWeights(), LossWeights(gold_nll_with_teacher=True, nll=0)])
def test_trainer_rejects_direct_items_with_legacy_or_disabled_gold_before_forward(
    monkeypatch, operation, weights
):
    splits = dataset()
    cfg = tiny_config(max_seq_len=2048)
    tok = ToyTokenizer()
    items, _ = prepare_direct_distillation(splits, tok, cfg, reads(splits), verify)
    # Exercise the real guard without constructing or calling a model/optimizer.
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = TrainConfig(loss_weights=weights)
    monkeypatch.setattr(trainer, "_plan", lambda _: pytest.fail("forward planning must not happen"))
    with pytest.raises(ValueError, match="gold-anchored"):
        getattr(trainer, operation)(items)


@pytest.mark.parametrize(
    "field",
    [
        "reasoning_positions",
        "reasoning_labels",
        "proposal_input_ids",
        "proposal_positions",
        "proposal_labels",
        "native_inputs",
    ],
)
def test_trainer_rejects_auxiliary_inputs_even_when_empty_before_forward(field):
    splits = dataset()
    tok = ToyTokenizer()
    cfg = tiny_config(max_seq_len=2048)
    items, _ = prepare_direct_distillation(splits, tok, cfg, {}, verify)
    items[0] = replace(items[0], **{field: [] if field != "native_inputs" else {}})
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = TrainConfig(loss_weights=LossWeights(gold_nll_with_teacher=True))
    with pytest.raises(ValueError, match="must not contain"):
        trainer.backward_step(items)
