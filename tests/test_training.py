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


def test_heldout_groups_same_state_and_lineage_across_cells():
    from ayaka.data.schema import Question, Sample
    from ayaka.training.run import split_eval

    pools = {}
    for family in ("a", "b"):
        pools[(family, "en")] = [
            Sample(
                state=f"case {i}",
                questions=[Question.noul("q", "test?", 1.0)],
                metadata={"source_family": family, "source_example_id": str(i)},
            )
            for i in range(20)
        ]
    # A translated/derived state must travel with its original lineage too.
    pools[("a", "en")].append(
        Sample(
            state="translation of case 3",
            questions=[Question.noul("q", "test?", 1.0)],
            metadata={"source_family": "a", "source_example_id": "3"},
        )
    )
    held = split_eval(pools, 10, 0)
    train = [s for cell in pools.values() for s in cell]
    assert {s.state for s in held}.isdisjoint(s.state for s in train)

    def lineage(s):
        return s.metadata["source_family"], s.metadata["source_example_id"]

    assert {lineage(s) for s in held}.isdisjoint(lineage(s) for s in train)
    assert all(pools.values())


def test_calibration_reserve_fills_long_buckets_and_prefers_natural_sources():
    from ayaka.data.schema import Question, Sample
    from ayaka.training.run import reserve_calibration

    def sample(state, source, i):
        return Sample(
            state=state,
            questions=[Question.noul("q", "holds?", 1.0)],
            metadata={"source": source, "source_example_id": f"{source}-{i}"},
        )

    short = [sample(f"short case {i}", "natural", i) for i in range(40)]
    long_nat = [sample(f"long natural {i} " + "x" * 300, "natural", i + 100) for i in range(6)]
    long_gen = [sample(f"long generated {i} " + "y" * 300, "synth_rules", i) for i in range(30)]
    pools = {("nat", "en"): short + long_nat, ("gen", "en"): long_gen}
    probe = tiny_config()
    lengths = [sample_to_items(s, TOK, probe)[0].length for s in (short[0], long_nat[0])]
    mcfg = tiny_config(long_prompt_tokens=sum(lengths) // 2)

    items = reserve_calibration(pools, TOK, mcfg, per_bucket=8, seed=0)
    long_items = [it for it in items if it.length >= mcfg.long_prompt_tokens]
    short_items = [it for it in items if it.length < mcfg.long_prompt_tokens]
    assert len(long_items) == 8 and len(short_items) == 8
    # every natural long group is used before any generated one
    assert sum(it.source == "natural" for it in long_items) == 6
    assert sum(it.source == "synth_rules" for it in long_items) == 2
    # reserved groups leave training, and no cell is emptied
    remaining = {s.state for cell in pools.values() for s in cell}
    assert len(remaining) == 76 - 16 and all(pools.values())


def test_item_stream_exact_questions_retains_shared_prefix():
    import random

    from ayaka.data.mixture import MixtureSampler
    from ayaka.data.schema import Question, Sample
    from ayaka.training.run import item_stream

    sample = Sample(
        "state",
        [Question.noul(f"q{i}", "test?", 0.5) for i in range(28)],
        {"task_family": "many", "source": "multilabel"},
    )
    cfg = RunConfig(questions_per_step=8)
    stream = item_stream(
        {("many", "en"): [sample]},
        MixtureSampler({"many": 1.0}),
        TOK,
        tiny_config(),
        cfg,
        random.Random(0),
    )
    items = next(stream)
    assert len(items) == 8
    assert len({id(it.enc.prefix_ids) for it in items}) == 1
    assert all(it.family == "many" and it.source == "multilabel" for it in items)


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
    assert loaded.temperature[1].tolist() == pytest.approx([1.7, 1.7])


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


@pytest.mark.parametrize("prune", [True, False])
def test_shared_prefix_training_matches_full_rows(prune):
    """Encoding a multi-question state once (KV cache branched to every
    question) gives the same loss and gradients as one full row per question."""
    from ayaka.data.schema import Candidate, Question, Sample
    from ayaka.losses import decision_loss
    from ayaka.training.batching import plan_chunks

    torch.manual_seed(0)
    m = ElectraDecisionModel.from_config(tiny_config(), dtype=torch.float64)
    m.head.double()
    m.gate.data = m.gate.data.double().fill_(0.5)
    m.prune_shared_positions = prune
    state = {
        "ticket": "the parcel never arrived and the customer wants a refund " * 4,
        "tier": "gold",
    }
    qs = [
        Question.noul("q0", "Is a refund requested?", 1.0),
        Question(
            "q1",
            "choice",
            "Category?",
            [Candidate("a", "billing"), Candidate("b", "shipping")],
            {"a": 0.3, "b": 0.7},
        ),
        Question(
            "q2",
            "score",
            "Urgency?",
            [Candidate(f"s{i}", f"level {i}", ordinal=i) for i in range(3)],
            {"s0": 0.0, "s1": 0.2, "s2": 0.8},
        ),
        Question.noul("q3", "Is the customer gold tier?", 1.0),
    ]
    items = sample_to_items(Sample(state=state, questions=qs), TOK, m.cfg)
    tr = Trainer(m, TOK, TrainConfig(steps=1, bf16=False), "cpu")

    def run(share):
        m.zero_grad()
        plan = plan_chunks(items, 100_000, share=share)
        assert {k for k, _ in plan} == ({"shared"} if share else {"rows"})
        total = 0.0
        for kind, mb in plan:
            out, t = tr._forward(kind, mb)
            loss = (
                decision_loss(out, t.targets, ordinals=t.ordinals)["total"] * len(mb) / len(items)
            )
            loss.backward()
            total += float(loss.detach())
        return (
            total,
            m.text_model().layers[0].self_attn.q_proj.weight.grad.clone(),
            m.head.w_q.weight.grad.clone(),
        )

    a, b = run(True), run(False)
    assert a[0] == pytest.approx(b[0], abs=1e-10)
    # Gemma's RMSNorm computes in float32 even in a float64 model, so
    # reordered work differs at float32 precision (~1e-7 relative)
    for ga, gb in ((a[1], b[1]), (a[2], b[2])):
        assert float((ga - gb).norm() / gb.norm()) < 1e-5


def test_plan_chunks_shares_only_when_worth_it():
    from ayaka.training.batching import plan_chunks

    pools = synthetic_pools(3)
    singles = [
        it for cell in pools.values() for s in cell for it in sample_to_items(s, TOK, tiny_config())
    ]
    assert {k for k, _ in plan_chunks(singles, 4096)} == {"rows"}  # one question per sample


def test_time_budget_stops_training_and_is_reported(tmp_path):
    cfg = RunConfig(
        model_size="tiny",
        steps=50,
        questions_per_step=8,
        micro_batch_tokens=4096,
        eval_questions=8,
        eval_every=0,
        fidelity_questions=0,
        jevbench=False,
        decontaminate=False,
        artifacts_dir=str(tmp_path),
        run_name="budget",
        log_every=0,
        bf16=False,
        max_train_seconds=1e-9,
    )
    res = run_training(cfg, pools=synthetic_pools(8), verbose=False)
    assert res["steps"] == 1 and res["stopped_early"] is True
    meta = json.loads((Path(res["checkpoint"]) / "meta.json").read_text())
    assert meta["stopped_early"] is True


def _mixed_length_items(m):
    from ayaka.data.schema import Question, Sample

    def sample(text):
        qs = [Question.noul(f"q{i}", f"Does clause {i} apply?", 1.0) for i in range(3)]
        return Sample(state={"doc": text}, questions=qs)

    short = sample("refund within 30 days")
    long = sample("the vendor must deliver the parts before the cutoff " * 6)
    return [it for s in (short, long) for it in sample_to_items(s, TOK, m.cfg)]


def test_selective_checkpointing_matches_plain_gradients_and_splits_by_length():
    from ayaka.losses import decision_loss

    torch.manual_seed(0)
    m = ElectraDecisionModel.from_config(tiny_config(), dtype=torch.float64)
    m.head.double()
    m.gate.data = m.gate.data.double().fill_(0.5)
    items = _mixed_length_items(m)
    lengths = sorted({it.length for it in items})
    assert len(lengths) == 2
    tr = Trainer(m, TOK, TrainConfig(steps=1, bf16=False), "cpu")
    m.train()

    def run(threshold):
        tr.ckpt_threshold = threshold
        m.zero_grad()
        plan = tr._plan(items)
        seen = []
        for kind, mb, ckpt in plan:
            tr._set_checkpointing(ckpt)
            seen.append((kind, ckpt, {it.length for it in mb}))
            assert bool(getattr(m.backbone, "is_gradient_checkpointing", ckpt)) == ckpt
            out, t = tr._forward(kind, mb)
            (decision_loss(out, t.targets)["total"] * len(mb) / len(items)).backward()
        tr._set_checkpointing(False)
        return seen, m.text_model().layers[0].self_attn.q_proj.weight.grad.clone()

    plain_plan, plain = run(None)
    sel_plan, selective = run(lengths[1])
    all_plan, everything = run(0)
    assert all(not ckpt for _, ckpt, _ in plain_plan)
    # short questions keep prefix sharing without checkpointing; long ones are
    # checkpointed full rows
    assert ("shared", False, {lengths[0]}) in sel_plan
    assert all(kind == "rows" and lens == {lengths[1]} for kind, ckpt, lens in sel_plan if ckpt)
    assert all(ckpt and kind == "rows" for kind, ckpt, _ in all_plan)
    for g in (selective, everything):
        assert float((g - plain).norm() / plain.norm()) < 1e-5


def test_oom_backoff_enables_selective_checkpointing_before_shrinking_batches(monkeypatch):
    m = _model()
    tr = Trainer(m, TOK, TrainConfig(steps=1, bf16=False, micro_batch_tokens=8192), "cpu")
    calls = []

    def fake_step(items):
        calls.append((tr.ckpt_threshold, tr.micro_tokens, tr.micro_ckpt_tokens))
        if len(calls) == 1:
            tr._chunk_ckpt = False
            raise torch.cuda.OutOfMemoryError("plain chunk")
        if len(calls) == 2:
            tr._chunk_ckpt = True
            raise torch.cuda.OutOfMemoryError("checkpointed chunk")
        return {"ok": True}

    monkeypatch.setattr(tr, "_train_step", fake_step)
    assert tr.train_step([]) == {"ok": True}
    assert calls == [(None, 8192, 8192), (1024, 8192, 8192), (1024, 8192, 4096)]


def test_compact_checkpoint_halves_adapter_with_small_output_drift(tmp_path):
    from ayaka.checkpoint import compact_checkpoint

    torch.manual_seed(3)
    m = _model()
    with torch.no_grad():
        for name, p in m.backbone.named_parameters():
            if "lora_" in name:
                p.normal_(0, 0.05)  # LoRA B starts at zero: make the adapter matter
    save_checkpoint(m, str(tmp_path / "full"), {"step": 1})
    sizes = compact_checkpoint(str(tmp_path / "full"), str(tmp_path / "bf16"))
    assert sizes["adapter_bytes_after"] < 0.55 * sizes["adapter_bytes_before"]
    assert json.loads((tmp_path / "bf16" / "meta.json").read_text())["adapter_dtype"] == "bfloat16"
    q = [QuestionSpec("noul", "Is it late?", ["no", "yes"])]
    state = {"order": 7, "status": "shipped on day 3"}
    out = {}
    for name in ("full", "bf16"):
        loaded = load_checkpoint(str(tmp_path / name), dtype=torch.bfloat16)
        out[name] = Decision(loaded, TOK).decide(state, q)[0].probs
    # an approximation: PEFT computes adapters in fp32, so rounding them moves
    # outputs slightly (LoRA weights here are far larger than trained ones)
    assert out["full"] != out["bf16"]
    assert max(abs(a - b) for a, b in zip(out["full"], out["bf16"], strict=True)) < 0.01
    with pytest.raises(ValueError):
        compact_checkpoint(str(tmp_path / "full"), str(tmp_path / "full"))


def test_head_safetensors_and_config_alias_roundtrip_with_legacy_fallback(tmp_path):
    import os

    from ayaka.checkpoint import compact_checkpoint, load_head

    torch.manual_seed(5)
    m = _model()
    with torch.no_grad():
        m.gate.fill_(0.3)
        m.temperature.fill_(1.7)
    save_checkpoint(m, str(tmp_path / "ck"), {"step": 1})
    for name in ("ayaka_config.json", "electra_config.json", "head.safetensors", "head.pt"):
        assert (tmp_path / "ck" / name).exists()
    a = load_head(str(tmp_path / "ck"))
    b = torch.load(tmp_path / "ck" / "head.pt", weights_only=True)
    assert torch.equal(a["gate"], b["gate"]) and torch.equal(a["temperature"], b["temperature"])
    assert all(torch.equal(a["head"][k], v) for k, v in b["head"].items())

    # a pre-safetensors checkpoint (legacy names only) still loads
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for name in ("electra_config.json", "head.pt", "meta.json"):
        (legacy / name).write_bytes((tmp_path / "ck" / name).read_bytes())
    os.rename(tmp_path / "ck" / "adapter", legacy / "adapter")
    loaded = load_checkpoint(str(legacy), dtype=torch.float32)
    assert torch.allclose(loaded.temperature, torch.full_like(loaded.temperature, 1.7))

    compact_checkpoint(str(legacy), str(tmp_path / "rel"))
    assert not (tmp_path / "rel" / "head.pt").exists()
    assert (tmp_path / "rel" / "head.safetensors").exists()
    assert (tmp_path / "rel" / "ayaka_config.json").exists()
    rel = load_checkpoint(str(tmp_path / "rel"), dtype=torch.float32)
    assert torch.allclose(rel.gate, torch.full_like(rel.gate, 0.3))
