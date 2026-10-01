from dataclasses import replace

import pytest
import torch

from ayaka.checkpoint import apply_lora
from ayaka.config import tiny_config
from ayaka.data.reasoning_v2 import curriculum
from ayaka.model.electra import ElectraDecisionModel
from ayaka.tokenization import ToyTokenizer
from ayaka.training.reasoning import reasoning_items
from ayaka.training.trainer import TrainConfig, Trainer


def test_joint_step_trains_trace_and_decision_without_prompt_ce():
    torch.set_num_threads(1)
    tok, cfg = ToyTokenizer(), tiny_config(version=2, max_seq_len=2048)
    sample, traces = curriculum("train", 1)[0]
    items = reasoning_items(sample, tok, cfg, traces)
    assert len(items) == 2 and items[0].reasoning_labels is None
    assert len(items[1].reasoning_labels) == len(items[1].reasoning_positions)
    assert items[1].reasoning_positions[0] > 0
    model = ElectraDecisionModel.from_config(cfg, dtype=torch.float32)
    model.backbone.requires_grad_(False)
    apply_lora(model)
    trainer = Trainer(model, tok, TrainConfig(steps=1, bf16=False, micro_batch_tokens=4096), "cpu")
    before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}
    record = trainer.train_step(items)
    assert record["reasoning_ce"] > 0 and record["nll"] > 0
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n in before)


def test_complete_trace_rejected_instead_of_truncated():
    sample, traces = curriculum("train", 1)[0]
    with pytest.raises(ValueError, match="do not truncate"):
        reasoning_items(sample, ToyTokenizer(), tiny_config(max_seq_len=64), traces)
    with pytest.raises(ValueError, match="non-empty"):
        reasoning_items(sample, ToyTokenizer(), tiny_config(), {"q": ""})


@pytest.mark.parametrize("readout", ["lm", "pointer", "hybrid"])
def test_head_ablation_runs_and_returns_valid_distributions(readout):
    torch.set_num_threads(1)
    from ayaka.primitives import Decision, QuestionSpec

    cfg = replace(tiny_config(), readout=readout, set_mixer_layers=0)
    model = ElectraDecisionModel.from_config(cfg, dtype=torch.float32).eval()
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
