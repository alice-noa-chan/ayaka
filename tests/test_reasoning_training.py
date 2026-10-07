from dataclasses import replace

import pytest
import torch

from ayaka.checkpoint import apply_lora
from ayaka.config import tiny_config
from ayaka.data.reasoning_v2 import curriculum
from ayaka.model.decision import AyakaDecisionModel
from ayaka.tokenization import ToyTokenizer
from ayaka.training.reasoning import reasoning_items
from ayaka.training.trainer import TrainConfig, Trainer


@pytest.mark.parametrize("primitive", ["choice", "noul", "score"])
def test_joint_step_trains_trace_and_decision_without_prompt_ce(primitive):
    torch.set_num_threads(1)
    tok, cfg = ToyTokenizer(), tiny_config(version=2, max_seq_len=2048)
    sample, traces = [
        row for row in curriculum("train", 3) if row[0].questions[0].type == primitive
    ][2]
    items = reasoning_items(sample, tok, cfg, traces)
    assert len(items) == 2 and items[0].reasoning_labels is None
    assert len(items[1].reasoning_labels) == len(items[1].reasoning_positions)
    assert items[1].reasoning_positions[0] > 0
    model = AyakaDecisionModel.from_config(cfg, dtype=torch.float32)
    model.backbone.requires_grad_(False)
    apply_lora(model)
    trainer = Trainer(model, tok, TrainConfig(steps=1, bf16=False, micro_batch_tokens=4096), "cpu")
    before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}
    record = trainer.train_step(items)
    assert record["reasoning_ce"] > 0 and record["nll"] > 0
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n in before)


@pytest.mark.parametrize(
    "levels", [("0.02", "0.01"), ("1152921504606846977", "1152921504606846976")]
)
def test_score_collation_preserves_exact_fractional_and_large_order(levels):
    from decimal import Decimal

    from ayaka.data.schema import Candidate, Question, Sample
    from ayaka.training.batching import collate_items, sample_to_items

    ordinals = list(map(Decimal, levels))
    sample = Sample(
        "Levels",
        [
            Question(
                "q",
                "score",
                "Rate?",
                [Candidate("a", "higher", ordinals[0]), Candidate("b", "lower", ordinals[1])],
                {"a": 1},
            )
        ],
    )
    items = sample_to_items(sample, ToyTokenizer(), tiny_config())
    assert collate_items(items, 0).ordinals.tolist() == [1, 0]
    assert items[0].ordinals == ordinals


def test_complete_trace_rejected_instead_of_truncated():
    sample, traces = curriculum("train", 1)[0]
    with pytest.raises(ValueError, match="do not truncate"):
        reasoning_items(sample, ToyTokenizer(), tiny_config(max_seq_len=64), traces)
    with pytest.raises(ValueError, match="non-empty"):
        reasoning_items(sample, ToyTokenizer(), tiny_config(), {"q": ""})


def test_prepared_decimal_curriculum_roundtrips_into_joint_training():
    import json

    from ayaka.data.schema import Sample
    from ayaka.eval.reasoning_v2 import dataset_signature
    from ayaka.training.batching import collate_items

    sample, traces = next(
        (s, t)
        for s, t in curriculum("train")
        if s.questions[0].type == "score" and s.metadata["task_family"] == "business"
    )
    restored = Sample.from_json(json.loads(json.dumps(sample.to_json())))
    assert dataset_signature([restored]) == dataset_signature([sample])
    assert [c.ordinal for c in restored.questions[0].candidates] == [
        c.ordinal for c in sample.questions[0].candidates
    ]
    items = reasoning_items(
        restored, ToyTokenizer(), tiny_config(version=2, max_seq_len=2048), traces
    )
    assert collate_items(items, 0).ordinals.tolist() == [1, 2, 3, 0, 1, 2, 3, 0]


@pytest.mark.parametrize("invalid", ["not-a-number", "NaN", "Infinity"])
def test_nonfinite_serialized_score_ordinal_is_rejected(invalid):
    from ayaka.data.schema import Sample

    sample, _ = next((s, t) for s, t in curriculum("train", 1) if s.questions[0].type == "score")
    serialized = sample.to_json()
    serialized["questions"][0]["candidates"][0]["ordinal"] = invalid
    with pytest.raises(ValueError, match="finite numeric"):
        Sample.from_json(serialized)


@pytest.mark.parametrize("readout", ["lm", "pointer", "hybrid"])
def test_head_ablation_runs_and_returns_valid_distributions(readout):
    torch.set_num_threads(1)
    from ayaka.primitives import Decision, QuestionSpec

    cfg = replace(tiny_config(), readout=readout, set_mixer_layers=0)
    model = AyakaDecisionModel.from_config(cfg, dtype=torch.float32).eval()
    result = Decision(model, ToyTokenizer()).decide(
        "Hello", [QuestionSpec("choice", "Intent?", ["hi", "refund"])]
    )[0]
    assert sum(result.probs) == pytest.approx(1, abs=1e-5)


def test_offline_boundary_oracles_and_balanced_curriculum():
    from ayaka.data.reasoning_v2 import _case

    assert _case("month_end", 0, "train")[1] == 29
    assert _case("month_end", 0, "router_train")[1] == 28
    assert _case("leap", 0, "train")[1] == 1  # 2000
    assert _case("leap", 1, "train")[1] == 0  # 2100
    assert _case("timezone", 0, "train")[1] == 28
    assert _case("rounding", 0, "train")[1] == 2050
    records = curriculum("dev")
    assert len(records) == 96
    assert all(sum(s.questions[0].target_distribution.values()) == 1 for s, _ in records)


def test_curriculum_has_no_family_polarity_or_fixed_rendered_answer_shortcut():
    from collections import defaultdict

    from ayaka.prompt import canonical_order

    for split in ("train", "router_train", "dev", "calibration", "test"):
        polarities, positions = defaultdict(set), defaultdict(set)
        for sample, _ in curriculum(split):
            q = sample.questions[0]
            if q.type == "noul":
                polarities[sample.metadata["task_family"]].add(q.target_distribution["true"])
            else:
                gold = next(i for i, c in enumerate(q.candidates) if q.target_distribution[c.id])
                order = (
                    canonical_order([c.description for c in q.candidates])
                    if q.type == "choice"
                    else sorted(range(len(q.candidates)), key=lambda i: q.candidates[i].ordinal)
                )
                positions[q.type].add(order.index(gold))
        assert all(labels == {0, 1} for labels in polarities.values())
        assert positions["choice"] == positions["score"] == {0, 1, 2, 3}
        from ayaka.data.reasoning_v2 import _case

        assert {_case("leap", i, split)[1] for i in (1, 11, 21, 31)} == {0, 1}


def test_curriculum_prefix_is_stable_and_polarity_depends_on_proposition():
    short = {
        (s.questions[0].type, s.metadata["source_example_id"]): s.to_json()
        for s, _ in curriculum("train", 8)
    }
    full = {
        (s.questions[0].type, s.metadata["source_example_id"]): s.to_json()
        for s, _ in curriculum("train", 128)
    }
    assert all(full[key] == sample for key, sample in short.items())
