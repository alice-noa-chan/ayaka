import json

from ayaka.data.schema import Candidate, Question, Sample
from ayaka.eval.evidence_eval import apply_policy, collect, report, sample_to_records, select
from ayaka.eval.jevbench import record_to_item
from ayaka.evidence_ids import SYSTEM
from ayaka.evidence_policy import EvidencePolicy
from ayaka.primitives import DecisionResult

SOURCE = "Subtotal 240.00.\nDiscount 15 percent.\nTax 8 percent after discount.\nBudget 225.00."
PLAN = next(ln for ln in SYSTEM.splitlines() if ln.startswith("Output: "))[8:]


class Stub:
    """Baseline 0.7 for "no"; any readout with verified work 0.9 for "yes"."""

    def decide(self, state, questions, device=None):
        p = [0.1, 0.9] if "<verified_work>" in str(state) else [0.7, 0.3]
        return [
            DecisionResult(q.type, p, dict(zip(q.candidates, p, strict=True))) for q in questions
        ]


class Gen:
    def __init__(self):
        self.calls = 0

    def generate(self, batch):
        self.calls += len(batch)
        assert all("SECRET" not in json.dumps(m) for m in batch)
        return [PLAN for _ in batch]


def _record(instruction, rid, criteria=None):
    return {
        "id": rid,
        "tier": "dev",
        "family": "SECRET_FAMILY",
        "state": SOURCE,
        "question": {
            "type": "noul",
            "instructions": instruction,
            "criteria": criteria or {"false": "Over budget", "true": "Within budget"},
        },
        "labels": ["no", "yes"],
        "expected": "yes",
    }


def test_sample_records_map_back_to_items_with_gold_in_labels():
    s = Sample(
        state="ticket",
        questions=[
            Question.noul("n", "ok?", 1.0),
            Question("c", "choice", "pick", [Candidate("a", "A"), Candidate("b", "B")], {"b": 1.0}),
            Question(
                "s",
                "score",
                "rate",
                [Candidate(f"s{i}", f"level {i}", ordinal=i) for i in range(3)],
                {"s2": 0.8, "s1": 0.2},
            ),
        ],
        metadata={"source_example_id": "7"},
    )
    records = sample_to_records(s, "spec")
    assert [r["expected"] for r in records] == ["true", "b", "2"]
    for r in records:
        item = record_to_item(r)
        assert item.expected in item.labels


def test_collect_runs_gated_questions_and_policies_use_stored_variants():
    gen = Gen()
    rows = collect(
        Stub(),
        gen,
        [
            _record("Is the total within budget?", "q1"),
            _record("Is the tone polite?", "q2", {"false": "Rude", "true": "Polite"}),
        ],
        log=lambda _: None,
    )
    gated, plain = rows
    assert gen.calls == 1 and gated["gate"] and not plain["gate"]
    assert gated["verified"] and not gated["recovered"] and gated["computed"] == "yes"
    assert gated["executed"]["yes"] == 0.9 and gated["quotes"]["yes"] == 0.9

    baseline = EvidencePolicy(readout="baseline")
    assert apply_policy(gated, baseline)[0] == gated["baseline"]
    routed = EvidencePolicy(readout="executed", weight=1.0, baseline_cutoff=1.0, gate="calculation")
    probs, was_routed, _ = apply_policy(gated, routed)
    assert was_routed and probs["yes"] == 0.9
    # a confident baseline above the cutoff is never routed
    strict = EvidencePolicy(readout="executed", baseline_cutoff=0.6, gate="calculation")
    assert apply_policy(gated, strict)[1] is False

    policy, table = select(rows)
    # the ungated question keeps its (wrong) baseline under every policy
    assert policy.readout != "baseline" and table[0][1]["correct"] == 1
    summary = report(rows, policy)["dev"]
    assert summary["baseline"]["correct"] == 0 and summary["policy"]["correct"] == 1
    assert summary["valid_plans"] == 1


class ReasonStub:
    def decide(self, state, questions, device=None):
        p = [0.2, 0.8] if "<worked_steps>" in str(state) else [0.7, 0.3]
        return [
            DecisionResult(q.type, p, dict(zip(q.candidates, p, strict=True))) for q in questions
        ]


class Notes:
    def __init__(self):
        self.seen = []

    def generate(self, batch):
        self.seen += batch
        return ["discounted 204.00; taxed 220.32; budget 225.00" for _ in batch]


def test_reasoning_variant_is_collected_scored_and_never_sees_gold():
    from ayaka.eval.evidence_eval import available

    notes = Notes()
    rows = collect(
        ReasonStub(),
        None,
        [_record("Is the total within budget?", "q1")],
        log=lambda _: None,
        reasoner=notes,
    )
    row = rows[0]
    assert row["reasoned"] == {"no": 0.2, "yes": 0.8} and "raw_plan" not in row
    assert "SECRET" not in json.dumps(notes.seen) and "expected" not in json.dumps(notes.seen)
    reasoned = EvidencePolicy(readout="reasoned", weight=1.0, baseline_cutoff=1.0)
    probs, routed, _ = apply_policy(row, reasoned)
    assert routed and probs == {"no": 0.2, "yes": 0.8}
    # plan policies are not scored on rows collected without plans
    assert not available(rows, EvidencePolicy(readout="executed"))
    policy, _ = select(rows)
    assert policy.readout == "reasoned"
