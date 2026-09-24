"""Training / calibration / checkpoint / distillation on the tiny backbone."""

import json
from pathlib import Path

import pytest
import torch

from ayaka.checkpoint import apply_lora, load_checkpoint, save_checkpoint
from ayaka.config import tiny_config
from ayaka.model.electra import ElectraDecisionModel
from ayaka.primitives import Decision, QuestionSpec
from ayaka.tokenization import ToyTokenizer
from ayaka.training.batching import budget_batches, collate_items, sample_to_items
from ayaka.training.calibrate import fit_temperature
from ayaka.training.run import RunConfig, run_training, synthetic_pools
from ayaka.training.trainer import TrainConfig, Trainer

TOK = ToyTokenizer()


def _model():
    m = ElectraDecisionModel.from_config(tiny_config(), dtype=torch.float32)
    for p in m.backbone.parameters():
        p.requires_grad_(False)
    return apply_lora(m)


def test_items_and_budget_batches():
    pools = synthetic_pools(4)
    items = [
        it for cell in pools.values() for s in cell for it in sample_to_items(s, TOK, tiny_config())
    ]
    assert len(items) == 16
    batches = budget_batches(items, max_tokens=2000, shuffle_seed=None)
    assert sum(len(b) for b in batches) == 16
    for b in batches:
        assert max(it.length for it in b) * len(b) <= 2000 or len(b) == 1
    t = collate_items(batches[0], TOK.pad_id)
    assert t.targets.numel() == t.batch.cand_cu[-1]
    assert t.teacher is None


def test_lora_only_trains_adapters_and_head():
    m = _model()
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable
    assert all("lora_" in n or n.startswith(("head.", "gate")) for n in trainable)


def test_training_reduces_loss_on_fixed_batch():
    m = _model()
    tr = Trainer(
        m,
        TOK,
        TrainConfig(
            steps=40,
            questions_per_step=8,
            lr=3e-3,
            head_lr=3e-3,
            grad_checkpointing=False,
            log_every=0,
        ),
        "cpu",
    )
    pools = synthetic_pools(2)
    items = [it for cell in pools.values() for s in cell for it in sample_to_items(s, TOK, m.cfg)]
    first = tr.train_step(items)["total"]
    for _ in range(25):
        last = tr.train_step(items)["total"]
    assert last < first * 0.7


def test_fit_temperature_recovers_scale():
    torch.manual_seed(0)
    logits, targets = [], []
    for _ in range(400):
        z = torch.randn(3) * 2
        p = torch.softmax(z, 0)
        y = int(torch.multinomial(p, 1))
        logits.append((z * 3).tolist())  # model is 3x overconfident
        targets.append([float(i == y) for i in range(3)])
    t = fit_temperature(logits, targets)
    assert 2.0 < t < 4.5


def test_checkpoint_roundtrip(tmp_path):
    m = _model()
    with torch.no_grad():
        m.gate.fill_(0.3)
        m.temperature[1] = 1.7
        for n, p in m.backbone.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)  # make the adapter non-trivial
    state = {"light": "red"}
    q = QuestionSpec("choice", "Which color?", ["red", "green", "blue"])
    before = Decision(m.eval(), TOK).decide(state, [q])[0].probs
    save_checkpoint(m, str(tmp_path / "ck"), {"step": 1})
    loaded = load_checkpoint(str(tmp_path / "ck"), dtype=torch.float32)
    after = Decision(loaded, TOK).decide(state, [q])[0].probs
    assert after == pytest.approx(before, abs=1e-4)
    assert float(loaded.temperature[1]) == pytest.approx(1.7)


def test_run_training_synthetic_end_to_end(tmp_path):
    cfg = RunConfig(
        model_size="tiny",
        steps=3,
        questions_per_step=8,
        micro_batch_tokens=4096,
        eval_questions=8,
        eval_every=0,
        fidelity_questions=0,
        jevbench=False,
        decontaminate=False,
        artifacts_dir=str(tmp_path),
        run_name="smoke",
        log_every=0,
        bf16=False,
    )
    res = run_training(cfg, pools=synthetic_pools(8), verbose=False)
    assert res["steps"] == 3
    ck = Path(res["checkpoint"])
    assert (ck / "adapter").is_dir() and (ck / "head.pt").exists()
    assert "heldout" in res and 0.0 <= res["heldout"]["accuracy"] <= 1.0
    assert len(json.loads((tmp_path / "smoke" / "history.json").read_text())) == 3


def test_distillation_roundtrip(tmp_path):
    """Teacher labels -> student run trains with KL on teacher questions."""
    from ayaka.training.distill import label_with_teacher

    teacher = _model()
    save_checkpoint(teacher, str(tmp_path / "teacher"))
    res = label_with_teacher(
        str(tmp_path / "teacher"),
        str(tmp_path / "t.jsonl"),
        specs=[],
        n_samples=12,
        pools=synthetic_pools(4),
        verbose=False,
    )
    assert res["samples"] == 12
    rows = [
        json.loads(line) for line in (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    tp = rows[0]["metadata"]["teacher_probs"]
    assert sum(next(iter(tp.values()))) == pytest.approx(1.0, abs=1e-4)

    cfg = RunConfig(
        model_size="tiny",
        steps=2,
        questions_per_step=8,
        micro_batch_tokens=4096,
        eval_questions=0,
        calibrate=False,
        fidelity_questions=0,
        jevbench=False,
        decontaminate=False,
        teacher_labels=str(tmp_path / "t.jsonl"),
        teacher_quota=0.9,
        artifacts_dir=str(tmp_path),
        run_name="student",
        log_every=0,
        bf16=False,
    )
    out = run_training(cfg, pools=synthetic_pools(4), verbose=False)
    hist = json.loads((tmp_path / "student" / "history.json").read_text())
    assert out["steps"] == 2
    assert any("kl" in h for h in hist)
