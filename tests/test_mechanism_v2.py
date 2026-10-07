import copy
import json
from dataclasses import replace

import pytest
import torch

from ayaka.collate import encode_decision, full_rows
from ayaka.config import tiny_config
from ayaka.data.recovery_holdout import independent_holdout
from ayaka.eval.mechanism_v2 import (
    CONDITIONS,
    HEADS,
    TimedTraceGenerator,
    complete_run_bound,
    evaluate_frozen,
    forced_trace,
    interaction_interval,
    prepare_cohort,
    reference_notes,
    score_heads,
    score_trace,
    validate_plan,
)
from ayaka.model.decision import AyakaDecisionModel
from ayaka.model.ragged import ragged_softmax
from ayaka.primitives import Decision, QuestionSpec
from ayaka.reasoning_pipeline import ControlledDecision, TraceGenerator
from ayaka.tokenization import ToyTokenizer


class Tok(ToyTokenizer):
    def decode(self, ids):
        return " ".join(map(str, ids))


def cohort(tmp_path, per_rule=3):
    path = tmp_path / "dev.jsonl"
    path.write_text(
        "\n".join(json.dumps(s.to_json()) for s in independent_holdout("dev", 18)),
        encoding="utf-8",
    )
    return prepare_cohort(path, per_rule)


def tiny():
    torch.set_num_threads(1)
    return AyakaDecisionModel.from_config(tiny_config(version=2), dtype=torch.float32).eval()


def test_cohort_balanced_repeatable_dev_only_and_oracles_recomputed(tmp_path):
    records = cohort(tmp_path)
    assert records == prepare_cohort(tmp_path / "dev.jsonl")
    assert len(records) == 18
    assert len({r["sample"]["metadata"]["source_lineage"] for r in records}) == 18
    assert sum(len(r["sample"]["questions"]) for r in records) == 54
    assert all(r["sample"]["metadata"]["language"] == "en" for r in records)
    assert all(
        r["distractor_source"] != r["sample"]["metadata"]["source_example_id"] for r in records
    )
    for sample in independent_holdout("dev", 240)[::3]:
        assert reference_notes(sample)
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(independent_holdout("test", 1)[0].to_json()))
    with pytest.raises(ValueError, match="evaluation-only dev"):
        prepare_cohort(path)


def test_oracle_refuses_corrupted_gold_or_visible_policy():
    sample = independent_holdout("dev", 1)[0]
    corrupted = copy.deepcopy(sample)
    corrupted.metadata["oracle_facts"]["value"] += 1
    with pytest.raises(ValueError, match="recomputation"):
        reference_notes(corrupted)
    corrupted = copy.deepcopy(sample)
    corrupted.questions[0].target_distribution = {"false": 0.5, "true": 0.5}
    with pytest.raises(ValueError, match="targets disagree"):
        reference_notes(corrupted)
    sample.state["policy"] += " Extra exception."
    with pytest.raises(ValueError, match="exact declared"):
        reference_notes(sample)


def test_complete_frozen_plan_and_full_cap_time_admission(tmp_path):
    from ayaka.training.prepare_v2 import canonical, sha256

    records = cohort(tmp_path, 2)
    plan = {
        "records": records,
        "budget": 512,
        "independent_cases": 12,
        "questions_per_checkpoint": 36,
        "cohort_sha256": sha256(canonical(records)),
    }
    validate_plan(plan)
    validate_plan({**plan, "active_checkpoints": ["parent"]})
    with pytest.raises(ValueError, match="checkpoint scope"):
        validate_plan({**plan, "active_checkpoints": ["pilot"]})
    corrupted = copy.deepcopy(plan)
    corrupted["records"][0]["distractor"] = "Invented answer"
    corrupted["cohort_sha256"] = sha256(canonical(corrupted["records"]))
    with pytest.raises(ValueError, match="distractor"):
        validate_plan(corrupted)
    bound = complete_run_bound(
        {
            "generated": {
                "tokens": 100,
                "seconds": 11.5,
                "decode_seconds": 10,
                "prefill_seconds": 1,
                "readout_seconds": 0.5,
            },
            "question_s": 13.5,
        },
        71,
        1,
    )
    assert bound == pytest.approx((51.2 + 1.5 + 2) * 71 * 1.10 + 240)
    short = complete_run_bound(
        {
            "generated": {
                "tokens": 7,
                "seconds": 2.2,
                "decode_seconds": 0.7,
                "prefill_seconds": 1,
                "readout_seconds": 0.5,
            },
            "question_s": 4.2,
        },
        71,
        1,
    )
    assert short == pytest.approx(bound)


def test_timed_generation_preserves_trace_and_cache_probabilities():
    model, tok = tiny(), Tok()
    spec = QuestionSpec("choice", "Which?", ["first", "second"])
    ordinary, timed = TraceGenerator(model, tok), TimedTraceGenerator(model, tok)
    messages = ordinary.messages_for("State", spec)
    ordinary.eos = timed.eos = set()
    first = ordinary.generate_trace(messages, 3)
    second = timed.generate_trace(messages, 3)
    assert first.token_ids == second.token_ids and first.text == second.text
    assert timed.prefill_seconds > 0 and timed.decode_seconds > 0
    assert ordinary.readout(first, spec) == pytest.approx(timed.readout(second, spec))


@pytest.mark.parametrize("gate", [0, 0.7])
@pytest.mark.parametrize("kind", ["noul", "choice", "score"])
def test_three_heads_same_encoding_match_production_and_full_cache(gate, kind):
    model, tok = tiny(), Tok()
    with torch.no_grad():
        model.gate.fill_(gate)
    candidates = ["false", "true"] if kind == "noul" else ["first", "second", "third"]
    q = QuestionSpec(kind, "Which?", candidates, [0, 1, 2] if kind == "score" else None)
    prefix, items = encode_decision("State", [q.view()], tok, 8192)
    batch = full_rows(items, tok.pad_id)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    heads = score_heads(model, batch, None)
    assert model.cfg.readout == "hybrid"
    for head in HEADS:
        model.cfg = replace(model.cfg, readout=head)
        with torch.inference_mode():
            out = model(batch, apply_temperature=False)
        assert heads[head] == pytest.approx(ragged_softmax(out.logits, out.cand_cu).tolist())
    model.cfg = replace(model.cfg, readout="hybrid")
    gen = TraceGenerator(model, tok, apply_temperature=False)
    trace = forced_trace(gen, "State", q, "A complete answer.")
    from ayaka.collate import EncodedQuestion
    from ayaka.reasoning_pipeline import readout_suffix

    full = full_rows(
        [
            EncodedQuestion(
                trace.input_ids + trace.token_ids,
                readout_suffix(tok, q),
                {"noul": 0, "choice": 1, "score": 2}[kind],
            )
        ],
        tok.pad_id,
    )
    cached = score_trace(gen, trace, q)
    uncached = score_heads(model, full, None)
    for head in HEADS:
        assert cached[head] == pytest.approx(uncached[head], abs=1e-5)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_readout_restored_on_failure_and_context_never_truncated(monkeypatch):
    model, tok = tiny(), Tok()
    q = QuestionSpec("noul", "True?", ["false", "true"])
    _, items = encode_decision("State", [q.view()], tok, 8192)
    batch = full_rows(items, tok.pad_id)
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise RuntimeError("failed")

        patch.setattr(model, "forward", fail)
        with pytest.raises(RuntimeError, match="failed"):
            score_heads(model, batch, None)
    assert model.cfg.readout == "hybrid"
    with pytest.raises(ValueError, match="refuse truncation"):
        forced_trace(TraceGenerator(model, tok, max_context=8), "State", q, "Notes")
    gen = TraceGenerator(model, tok)
    gen.eos = {1}
    assert forced_trace(gen, "State", q, "", termination_token=1).token_ids == [1]
    assert forced_trace(gen, "State", q, "").token_ids == []
    with pytest.raises(ValueError, match="recognized EOS"):
        forced_trace(gen, "State", q, "Notes", termination_token=3)


def test_end_to_end_one_generation_per_question_frozen_and_paired(tmp_path, monkeypatch):
    model, tok = tiny(), Tok()
    gen = TraceGenerator(model, tok, apply_temperature=False)
    original = gen.generate_trace
    calls = []

    def short(self, messages, budget, reserve=0):
        calls.append(budget)
        return original(messages, 1, reserve)

    monkeypatch.setattr(TraceGenerator, "generate_trace", short)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    decision = ControlledDecision(Decision(model, tok, max_seq_len=8192), gen)
    progress = []
    report = evaluate_frozen(decision, cohort(tmp_path, 1), progress=progress.append)
    assert calls == [512] * 18
    assert len(progress) == 18 and report["complete"]
    assert report["optimizer_steps"] == 0 and not report["calibration_fitted"]
    assert all(len(rows) == 18 for rows in report["rows"].values())
    assert set(report["rows"]) == {f"{c}/{h}" for c in CONDITIONS for h in HEADS}
    assert report["interaction"]["generated"]["independent_cases"] == 6
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    corrupted = copy.deepcopy(report["rows"])
    corrupted["generated/lm"][0]["target"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="identical question"):
        interaction_interval(corrupted, "generated", 10)
    with pytest.raises(TimeoutError, match="incomplete"):
        evaluate_frozen(decision, cohort(tmp_path, 1), deadline=0)
    decision.max_seq_len = 8
    with pytest.raises(ValueError, match="truncate"):
        evaluate_frozen(decision, cohort(tmp_path, 1))
