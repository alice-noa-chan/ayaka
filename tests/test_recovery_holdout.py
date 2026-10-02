import copy
from datetime import date, timedelta

import pytest

from ayaka.data.recovery_audit import shortcut_audit
from ayaka.data.recovery_holdout import holdout_case, independent_holdout
from ayaka.data.recovery_v2 import recovery_curriculum
from ayaka.training.prepare_v2 import audit_splits


def half_up(numerator, denominator):
    return (2 * numerator + denominator) // (2 * denominator)


def test_holdout_oracles_recompute_with_integer_arithmetic_and_separate_calendar_walk():
    for split in ("dev", "test"):
        for index in range(240):
            r, facts, _ = holdout_case(split, index)
            kind = index % 6
            if kind == 0:
                current, count = date.fromisoformat(r["start"]), 0
                while True:
                    count += (
                        current.weekday() not in r["weekend_weekdays"]
                        and str(current) not in r["holidays"]
                    )
                    if count == r["required"]:
                        break
                    current += timedelta(days=1)
                expected = max(0, (date.fromisoformat(r["delivery"]) - current).days)
            elif kind == 1:
                current = date.fromisoformat(r["start"])
                y, m = divmod(current.year * 12 + current.month - 1 + r["months"], 12)
                first_next = date(y + (m == 11), (m + 1) % 12 + 1, 1)
                expected = min(current.day, (first_next - timedelta(days=1)).day)
            elif kind in {2, 3}:
                discounted = half_up(
                    r["unit_cents"] * r["quantity"] * (100 - r["discount_percent"]), 100
                )
                if kind == 2:
                    expected = half_up(
                        (discounted + r["shipping_cents"]) * (100 + r["tax_percent"]), 100
                    )
                else:
                    discounted = half_up(discounted * (100 - r["second_discount_percent"]), 100)
                    expected = (
                        half_up(discounted * (100 + r["tax_percent"]), 100) + r["shipping_cents"]
                    )
            elif kind == 4:
                expected = int(
                    not r["cancelled"]
                    and r["cost"] <= r["limit"]
                    and (r["override"] or r["credential"] >= r["required"])
                )
            else:
                expected = {"0": 0.5, "1": 0.5}
            assert expected == facts["value"]
            assert "value" not in r and "deadline" not in r


def test_independent_holdout_is_not_derived_from_training_and_keeps_translations_grouped():
    splits = {
        s: recovery_curriculum(s, 60, generation=3)
        for s in ("train", "router_train", "calibration")
    }
    splits.update({s: independent_holdout(s, 60) for s in ("dev", "test")})
    assert audit_splits(splits)
    for split in ("dev", "test"):
        samples = splits[split]
        assert all(
            s.metadata["evaluation_only"] and "verified_traces" not in s.metadata for s in samples
        )
        assert len({s.metadata["source_lineage"] for s in samples}) == 60
        assert all(sum(q.target_distribution.values()) == 1 for s in samples for q in s.questions)
    assert (
        recovery_curriculum("train", 3, generation=3)[0].state
        != recovery_curriculum("train", 3)[0].state
    )


def test_shortcut_audit_accepts_corrected_data_and_rejects_reintroduced_defects():
    samples = recovery_curriculum("dev", 600, generation=3)
    assert shortcut_audit(samples)["passed"]
    assert shortcut_audit(independent_holdout("test", 600))["passed"]
    changed = copy.deepcopy(samples)
    for sample in changed:
        for question in sample.questions:
            if question.type == "choice" and sample.metadata["task_family"] == "numeric":
                middle = sorted(question.candidates, key=lambda c: int(c.id))[2].id
                question.target_distribution = {
                    c.id: float(c.id == middle) for c in question.candidates
                }
    with pytest.raises(ValueError, match="shortcuts"):
        shortcut_audit(changed)
    changed = copy.deepcopy(samples)
    for sample in changed:
        truth = bool(date.fromisoformat(sample.metadata["oracle_facts"]["local"][:10]).year % 2)
        q = next(q for q in sample.questions if q.type == "noul")
        q.target_distribution = {"false": float(not truth), "true": float(truth)}
    with pytest.raises(ValueError, match="shortcuts"):
        shortcut_audit(changed)
