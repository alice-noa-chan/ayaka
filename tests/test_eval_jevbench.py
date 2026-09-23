"""JevBench adapter + training-pipeline stage tests (CPU, tiny config)."""

import json

import pytest
import torch

from ayaka.eval.jevbench import evaluate, load_model, record_to_item

REC_CHOICE = {
    "expected": "track_order",
    "family": "intent",
    "labels": ["track_order", "cancel_order"],
    "question": {
        "type": "choice",
        "instructions": "Which intent does the user's message express?",
        "criteria": {
            "track_order": "Wants to know where an order is or when it arrives",
            "cancel_order": "Wants to cancel an order",
        },
    },
    "state": "Where is my package?",
}

REC_NOUL = {
    "expected": "yes",
    "family": "entailment",
    "labels": ["no", "yes"],
    "question": {
        "type": "noul",
        "instructions": "Is the user asking about shipping?",
        "criteria": {"false": "not about shipping", "true": "about shipping"},
    },
    "state": "When will it arrive?",
}

REC_SCORE = {
    "expected": "2",
    "family": "sentiment",
    "labels": ["0", "1", "2"],
    "question": {
        "type": "score",
        "instructions": "Rate the sentiment.",
        "criteria": ["negative", "neutral", "positive"],
    },
    "state": "I absolutely loved it!",
}


def test_record_to_item_choice():
    item = record_to_item(REC_CHOICE)
    assert item.spec.type == "choice"
    assert item.labels == ["track_order", "cancel_order"]
    # candidate descriptions come from criteria, not labels
    assert item.spec.candidates[0].startswith("Wants to know")
    assert "track_order" not in item.spec.instruction


def test_record_to_item_noul():
    item = record_to_item(REC_NOUL)
    assert item.spec.type == "noul"
    assert item.spec.candidates == ["not about shipping", "about shipping"]
    assert item.labels == ["no", "yes"]


def test_record_to_item_score():
    item = record_to_item(REC_SCORE)
    assert item.spec.type == "score"
    assert item.spec.candidates == ["negative", "neutral", "positive"]
    assert item.spec.ordinals == [0, 1, 2]


def _tiny_decision():
    from ayaka.config import tiny_config
    from ayaka.model.model import ElectraDecisionModel
    from ayaka.primitives import Decision
    from ayaka.tokenizer import HashTokenizer

    mcfg = tiny_config()
    model = ElectraDecisionModel(mcfg).eval()
    return Decision(model, HashTokenizer(mcfg.vocab_size))


def test_evaluate_records_metrics_shape():
    dec = _tiny_decision()
    metrics = evaluate(dec, [REC_CHOICE, REC_NOUL, REC_SCORE])
    assert metrics["n"] == 3
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["brier"] >= 0.0
    assert metrics["latency_mean_s"] > 0
    assert set(metrics["by_type"]) == {"choice", "noul", "score"}
    r = metrics["results"][0]
    assert r["probs_source"] == "model"
    # probs cover every exact label and sum to ~1
    assert set(r["probs"]) == set(REC_CHOICE["labels"])
    assert sum(r["probs"].values()) == pytest.approx(1.0, abs=1e-5)


def test_load_model_roundtrip(tmp_path):
    from ayaka.config import tiny_config
    from ayaka.model.model import ElectraDecisionModel

    mcfg = tiny_config()
    model = ElectraDecisionModel(mcfg)
    torch.save({"model": model.state_dict()}, tmp_path / "checkpoint_final.pt")
    (tmp_path / "run_config.json").write_text(json.dumps({"model_size": "tiny"}))
    m2, tok, ckpt = load_model(str(tmp_path / "checkpoint_final.pt"))
    assert isinstance(tok.vocab_size, int) and tok.vocab_size > 0
    assert "model" in ckpt


# ---------------------------------------------------- pipeline stage wiring


def test_split_eval_proportional_and_exclusive():
    from ayaka.training.run import _split_eval, synthetic_pools

    pools = synthetic_pools(40)
    total_before = sum(len(v) for v in pools.values())
    ev = _split_eval(pools, 10, seed=0)
    assert 0 < len(ev) <= 10
    total_after = sum(len(v) for v in pools.values())
    assert total_before == total_after + len(ev)
    ev_ids = {id(s) for s in ev}
    assert all(id(s) not in ev_ids for cell in pools.values() for s in cell)


def test_augment_sample_permutation_keeps_target():
    import random

    from ayaka.data.schema import Candidate, Question, Sample
    from ayaka.training.run import RunConfig, _augment_sample

    q = Question(
        id="q",
        type="choice",
        instruction="pick",
        candidates=[Candidate(id="a", description="A"), Candidate(id="b", description="B")],
        target_distribution={"a": 1.0, "b": 0.0},
    )
    s = Sample(state="s", questions=[q], metadata={})
    cfg = RunConfig(augment_p=1.0, evidence_aug_p=0.0)
    out = _augment_sample(s, cfg, random.Random(0))
    # target stays attached to the same candidate id under permutation
    for cid, p in out.questions[0].target_distribution.items():
        assert p == (1.0 if cid == "a" else 0.0)


def test_augment_sample_evidence_deletion_uniform():
    import random

    from ayaka.training.run import RunConfig, _augment_sample, synthetic_pools

    s = synthetic_pools(1)[("nli", "en")][0] if synthetic_pools(1) else None
    if s is None:  # pragma: no cover - cell naming guard
        pytest.skip("synthetic pool cell missing")
    cfg = RunConfig(augment_p=0.0, evidence_aug_p=1.0)
    out = _augment_sample(s, cfg, random.Random(0))
    assert out.metadata.get("evidence_state") == "deleted"
    for q in out.questions:
        ps = list(q.target_distribution.values())
        assert all(abs(p - ps[0]) < 1e-9 for p in ps)


def test_run_training_full_pipeline(tmp_path):
    """pretrain + augment + eval + calibrate + BPE training in one run."""
    pytest.importorskip("tokenizers")
    from ayaka.training.run import RunConfig, run_training, synthetic_pools

    cfg = RunConfig(
        model_size="tiny",
        steps=3,
        samples_per_step=8,
        token_budget=2048,
        train_tokenizer=True,
        pretrain_steps=2,
        augment_p=0.5,
        evidence_aug_p=0.2,
        eval_samples=4,
        calibrate=True,
        artifacts_dir=str(tmp_path),
        run_name="full",
        log_every=0,
        bf16=False,
    )
    result = run_training(cfg, pools=synthetic_pools(24), verbose=False)
    assert result["steps"] == 3
    art = tmp_path / "full"
    assert (art / "tokenizer.json").exists()
    assert result["tokenizer"].endswith("tokenizer.json")
    assert result["eval_metrics"]
    assert (art / "pretrain_history.json").exists()
    # calibrated temps persisted in the checkpoint's model buffer
    ckpt = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert "temperature" in ckpt["model"]
