from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal

from ayaka.data.recovery_v2 import recovery_curriculum, verified_case
from ayaka.training.prepare_v2 import audit_splits


def test_recovery_oracles_recompute_multistage_results_independently():
    for split in ("train", "dev", "test"):
        for index in range(60):
            facts, target, trace = verified_case(split, index)
            utc = datetime.fromisoformat(facts["utc"])
            if facts["family"] == "temporal_numeric":
                start = utc.date()
                eligible = [
                    start + timedelta(days=i)
                    for i in range(1, 40)
                    if (start + timedelta(days=i)).weekday() < 5
                    and str(start + timedelta(days=i)) not in facts["holidays"]
                ]
                deadline = eligible[facts["required"] - 1]
                actual = datetime.fromisoformat(facts["actual"]).date()
                assert target == max(0, (actual - deadline).days)
            elif facts["family"] == "numeric":
                rate = (
                    facts["new_tax"] if str(utc.date()) >= facts["effective"] else facts["old_tax"]
                )
                discounted_cents = (
                    Decimal(facts["unit_cents"] * facts["quantity"])
                    * Decimal(100 - facts["discount_percent"])
                    / 100
                ).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                taxed_cents = (discounted_cents * Decimal(100 + rate) / 100).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP
                )
                assert target == int(taxed_cents) + facts["shipping_cents"]
            else:
                limit = (
                    facts["new_limit"]
                    if str(utc.date()) >= facts["effective"]
                    else facts["old_limit"]
                )
                blocked = facts["exception"] and not (
                    facts["override"] and facts["credential"] >= facts["required"]
                )
                assert target == int(facts["cost"] <= limit and not blocked)
            assert str(target) in trace and "UTC" in trace


def test_evidence_excludes_computed_answers_and_translations_share_lineage():
    samples = recovery_curriculum("train", 6)
    for start in range(0, len(samples), 3):
        translated = samples[start : start + 3]
        assert len({sample.metadata["source_lineage"] for sample in translated}) == 1
        for sample in translated:
            assert set(sample.state["record"]).isdisjoint({"value", "utc", "deadline"})
            assert {q.type for q in sample.questions} == {"choice", "noul", "score"}
            assert len(sample.metadata["verified_traces"]) == 3
    splits = {
        split: recovery_curriculum(split, 6)
        for split in ("train", "router_train", "dev", "calibration", "test")
    }
    assert audit_splits(splits)
