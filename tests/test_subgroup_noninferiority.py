"""Clustered non-inferiority for per-source and per-language CC checks."""

import pytest

from ayaka.eval.heldout_tuning import gate
from ayaka.eval.quality_hierarchy import _comparison, subgroup_cc_interval
from ayaka.eval.v2 import typed_row


class _Spec:
    def __init__(self, kind):
        self.type, self.ordinals = kind, None


def choice_row(qid, source, correct, cluster=None, language="en"):
    probs = [0.7, 0.1, 0.1, 0.1]
    target = [1.0, 0.0, 0.0, 0.0] if correct else [0.0, 1.0, 0.0, 0.0]
    return {
        **typed_row(_Spec("choice"), probs, target),
        "id": qid,
        "cluster_id": cluster or qid,
        "source": source,
        "language": language,
        "tier": "standard",
    }


def other_rows():
    """Unchanged Noul and Score rows, so every type the screen checks is present."""
    rows = []
    for i in range(40):
        noul = typed_row(_Spec("noul"), [0.1, 0.9], [0.0, 1.0])
        spec = _Spec("score")
        spec.ordinals = [0, 1, 2]
        score = typed_row(spec, [0.1, 0.8, 0.1], [0.0, 1.0, 0.0])
        for kind, row in (("noul", noul), ("score", score)):
            rows.append(
                {
                    **row,
                    "id": f"other/{kind}/{i}",
                    "cluster_id": f"other/{kind}/{i}",
                    "source": "other",
                    "language": "en",
                    "tier": "standard",
                }
            )
    return rows


def systems(*, small_drop, large_drop):
    """Two Choice sources of 60 questions; v2 gains on ``easy`` and loses on ``mixed``."""
    before, after = other_rows(), other_rows()
    for i in range(60):
        before.append(choice_row(f"easy/{i}", "easy", i < 30))
        after.append(choice_row(f"easy/{i}", "easy", i < 50))
    lost = 1 if small_drop else 0
    lost = 25 if large_drop else lost
    for i in range(60):
        before.append(choice_row(f"mixed/{i}", "mixed", i < 50))
        after.append(choice_row(f"mixed/{i}", "mixed", i < 50 - lost))
    return before, after


def test_one_question_drop_fails_zero_tolerance_but_passes_noninferiority():
    before, after = systems(small_drop=True, large_drop=False)
    name = "source:mixed/choice/cc_not_worse"
    strict = _comparison(before, after, major_gain=False, replicates=400)
    assert strict["checks"][name] is False and strict["subgroup_rule"] == "zero_tolerance"
    relaxed = _comparison(
        before,
        after,
        major_gain=False,
        replicates=400,
        subgroup_rule="clustered_noninferiority",
    )
    assert relaxed["checks"][name] is True
    low, high = relaxed["groups"]["source"]["mixed"]["cc_delta_95ci"]["choice"]
    assert low < 0 <= high


def test_a_significant_subgroup_regression_still_fails():
    before, after = systems(small_drop=False, large_drop=True)
    relaxed = _comparison(
        before,
        after,
        major_gain=False,
        replicates=400,
        subgroup_rule="clustered_noninferiority",
    )
    assert relaxed["checks"]["source:mixed/choice/cc_not_worse"] is False
    assert relaxed["groups"]["source"]["mixed"]["cc_delta_95ci"]["choice"][1] < 0


def test_global_checks_are_identical_under_both_rules():
    before, after = systems(small_drop=True, large_drop=False)
    strict = _comparison(before, after, major_gain=False, replicates=200)
    relaxed = _comparison(
        before, after, major_gain=False, replicates=200, subgroup_rule="clustered_noninferiority"
    )
    for name, value in strict["checks"].items():
        if not name.startswith(("source:", "language:")):
            assert relaxed["checks"][name] == value


def test_clusters_are_resampled_as_whole_cases():
    before = [choice_row(f"q{i}", "s", True, cluster=f"c{i // 5}") for i in range(20)]
    after = [choice_row(f"q{i}", "s", i % 5 != 0, cluster=f"c{i // 5}") for i in range(20)]
    low, high = subgroup_cc_interval(before, after, "choice", 300)
    # Every case loses exactly one of its five questions, so the change never varies.
    assert low == pytest.approx(high)
    assert high < 0


def test_unknown_rules_and_misaligned_systems_are_rejected():
    before, after = systems(small_drop=True, large_drop=False)
    with pytest.raises(ValueError):
        _comparison(before, after, major_gain=False, replicates=10, subgroup_rule="lenient")
    with pytest.raises(ValueError):
        _comparison(before, list(reversed(after)), major_gain=False, replicates=10)


def test_heldout_gate_reports_its_subgroup_rule():
    before, after = systems(small_drop=True, large_drop=False)
    result = gate(
        before, after, major_gain=False, replicates=200, subgroup_rule="clustered_noninferiority"
    )
    assert result["subgroup_rule"] == "clustered_noninferiority"
    assert "source:mixed/choice/cc_not_worse" not in result["failed_checks"]
