import pytest
import torch

from ayaka.config import tiny_config
from ayaka.data.schema import Candidate, Question, Sample
from ayaka.model.model import ElectraDecisionModel
from ayaka.tokenizer import HashTokenizer
from ayaka.training.batch import build_train_batch
from ayaka.training.calibrate import apply_temperatures, fit_temperatures
from ayaka.training.pretrain import (
    RTDHead,
    SpanRelationHead,
    multilingual_alignment_loss,
    rtd_corrupt,
    rtd_loss,
    span_relation_loss,
    structured_corrupt,
)
from ayaka.training.schedule import build_optimizer, cosine_warmup_schedule
from ayaka.training.trainer import TrainConfig, Trainer


def _samples(n=4, seed=0):
    out = []
    for i in range(n):
        cands = [Candidate(f"c{j}", f"opt{j}") for j in range(3)]
        q = Question(f"q{i}", "choice", "pick?", cands, {"c0": 1.0, "c1": 0.0, "c2": 0.0})
        out.append(Sample(state=f"state {i} text", questions=[q], metadata={"language": "en"}))
    return out


def test_build_train_batch_layout():
    tok = HashTokenizer(512)
    cfg = tiny_config()
    samples = _samples(2)
    batch = build_train_batch(samples, tok, cfg.block_size)
    assert batch.targets.shape == (6,)  # 2 questions x 3 candidates
    assert batch.inputs["state_cu"].tolist() == [
        0,
        batch.inputs["state_cu"][1],
        batch.inputs["state_cu"][2],
    ]
    assert batch.score_question_mask.tolist() == [False, False]
    assert batch.cand_ordinals.tolist() == [-1] * 6
    assert not batch.missing_mask.any()
    # targets sum to 1 per question
    assert batch.targets[:3].sum() == pytest.approx(1.0)


def test_build_train_batch_score_and_missing():
    tok = HashTokenizer(512)
    cfg = tiny_config()
    cands = [Candidate(f"l{i}", f"lv{i}", ordinal=i) for i in range(3)]
    q = Question("q", "score", "sev?", cands, {"l0": 0.0, "l1": 0.6, "l2": 0.4})
    s = Sample(state="s", questions=[q], metadata={"evidence_state": "deleted"})
    batch = build_train_batch([s], tok, cfg.block_size)
    assert batch.score_question_mask.tolist() == [True]
    assert batch.missing_mask.tolist() == [True]
    assert batch.cand_ordinals.tolist() == [0, 1, 2]


def test_build_train_batch_router_supervision():
    tok = HashTokenizer(512)
    cfg = tiny_config()
    s = Sample(
        state="x " * 200,  # ~200 tokens -> 4 blocks of 64
        questions=[_choice_q()],
        metadata={"evidence_blocks": [0, 2]},
    )
    batch = build_train_batch([s], tok, cfg.block_size)
    assert batch.block_target is not None and batch.block_mask is not None
    n_c = batch.targets.shape[0]
    assert batch.block_target.shape[0] == n_c
    assert batch.block_target[0, 0] == 1.0 and batch.block_target[0, 2] == 1.0
    assert batch.block_mask[0].sum() > 0


def _choice_q():
    cands = [Candidate(f"c{j}", f"o{j}") for j in range(2)]
    return Question("q", "choice", "p?", cands, {"c0": 1.0, "c1": 0.0})


def test_optimizer_param_groups():
    model = ElectraDecisionModel(tiny_config())
    opt = build_optimizer(model, lr=1e-3, fused=False)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 0.1
    assert opt.param_groups[1]["weight_decay"] == 0.0


def test_lr_schedule_warmup_floor():
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0)
    sched = cosine_warmup_schedule(opt, total_steps=100, warmup_frac=0.1, min_lr_frac=0.1)
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(99):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[9] == pytest.approx(1.0, abs=0.15)  # end of warmup
    assert lrs[-1] == pytest.approx(0.1, abs=1e-3)  # floor
    assert max(lrs) <= 1.0 + 1e-6


def test_rtd_corruption_and_loss():
    torch.manual_seed(0)
    cfg = tiny_config()
    ids = torch.randint(30, 500, (40,))
    cu = torch.tensor([0, 40])
    corrupted, labels = rtd_corrupt(
        ids, cu, 30, 500, rate=0.5, generator=torch.Generator().manual_seed(0)
    )
    assert (corrupted != ids).float().mean() > 0.2
    assert labels.sum() > 0
    assert (labels[ids < 30] == 0).all()  # special tokens never corrupted
    model = ElectraDecisionModel(cfg)
    head = RTDHead(cfg.hidden)
    loss = rtd_loss(model, head, ids, corrupted, labels, cu)
    assert loss > 0


def test_structured_corrupt_swaps_markers():
    from ayaka.special_tokens import SPECIAL_TOKEN_IDS

    torch.manual_seed(0)
    num_id = SPECIAL_TOKEN_IDS["<num>"]
    ids = torch.tensor([num_id, 100, num_id, 200])
    corrupted, labels = structured_corrupt(
        ids, rate=1.0, generator=torch.Generator().manual_seed(0)
    )
    assert labels[0] == 1 and labels[2] == 1
    assert corrupted[0] != num_id and corrupted[2] != num_id
    assert corrupted[1] == 100 and corrupted[3] == 200


def test_span_and_alignment_losses():
    head = SpanRelationHead(16)
    a, b = torch.randn(4, 16), torch.randn(4, 16)
    labels = torch.tensor([0, 1, 2, 0])
    assert span_relation_loss(head, a, b, labels) > 0
    x = torch.randn(6, 16)
    same = multilingual_alignment_loss(x, x.clone())
    diff = multilingual_alignment_loss(x, torch.randn(6, 16))
    assert same < diff


def test_trainer_step_and_fit_temperature():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = ElectraDecisionModel(cfg)
    tok = HashTokenizer(512)
    trainer = Trainer(model, TrainConfig(steps=3, lr=1e-3, bf16=False, device="cpu", fused=False))
    samples = _samples(4)
    batches = [
        build_train_batch(samples[:2], tok, cfg.block_size),
        build_train_batch(samples[2:], tok, cfg.block_size),
    ]
    losses = []
    for _ in range(3):
        rec = trainer.step(batches[0])
        losses.append(rec["loss"])
    assert all(x > 0 for x in losses)
    assert trainer.step_i == 3
    # temperature fitting on held-out
    with torch.no_grad():
        out = model(**{**batches[0].inputs, "apply_temperature": False})
    temps = fit_temperatures(
        {1: out.logits},
        {1: batches[0].targets},
        {1: out.cand_cu},
        iters=10,
    )
    assert temps[1] > 0
    apply_temperatures(model, temps)
    assert model.temperature[1].item() == pytest.approx(temps[1])
