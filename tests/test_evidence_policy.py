import pytest

from ayaka.evidence_policy import EvidencePolicy, fuse_probabilities


def test_invalid_plan_and_high_confidence_preserve_baseline_exactly():
    baseline = {"no": 0.97, "yes": 0.03}
    policy = EvidencePolicy()
    assert fuse_probabilities(baseline, {}, policy, usable=False) == (baseline, "baseline")
    assert fuse_probabilities(baseline, {"no": 0.1, "yes": 0.9}, policy, usable=True) == (
        baseline,
        "baseline",
    )


def test_boolean_polarity_and_fusion_align_with_actual_labels():
    b = {"FALSE": 0.6, "TRUE": 0.4}
    p = EvidencePolicy(boolean_confidence=0.8, baseline_cutoff=1)
    out, reason = fuse_probabilities(
        b, {"FALSE": 0.7, "TRUE": 0.3}, p, computed_label="TRUE", usable=True
    )
    assert out["TRUE"] == pytest.approx(0.6)
    assert sum(out.values()) == pytest.approx(1)
    assert reason == "computed_predicate"


def test_label_permutation_preserves_semantics():
    b = {"A": 0.3, "B": 0.5, "C": 0.2}
    a = {"B": 0.1, "C": 0.2, "A": 0.7}
    p = EvidencePolicy(baseline_cutoff=1)
    out, _ = fuse_probabilities(b, a, p, usable=True)
    flipped, _ = fuse_probabilities(dict(reversed(list(b.items()))), a, p, usable=True)
    assert out == flipped


def test_unmatched_or_nonprobability_outputs_cannot_be_promoted():
    p = EvidencePolicy(baseline_cutoff=1)
    with pytest.raises(ValueError, match="mismatch"):
        fuse_probabilities({"a": 0.5, "b": 0.5}, {"x": 0.4, "y": 0.6}, p, usable=True)
    with pytest.raises(ValueError, match="auxiliary"):
        fuse_probabilities({"a": 0.5, "b": 0.5}, {"a": float("nan"), "b": 0.6}, p, usable=True)
