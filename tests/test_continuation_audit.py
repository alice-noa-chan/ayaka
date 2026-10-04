import copy

import pytest

from ayaka.eval.continuation_audit import audit_continuation


def report(probs):
    return {
        "complete": True,
        "split": "dev",
        "model_id": "a" * 64,
        "rows": {
            "off": [
                {
                    "id": "case/q",
                    "cluster_id": "case",
                    "model_id": "a" * 64,
                    "split": "dev",
                    "route": "direct",
                    "budget": 0,
                    "reasoning_tokens": 0,
                    "language": "en",
                    "family": "calendar",
                    "modality": "text",
                    "partition": "fixed",
                    "type": "noul",
                    "probs": probs,
                    "target": [0.0, 1.0],
                    "ordinals": None,
                    "correct": 999,
                    "nll": -999,
                }
            ]
        },
    }


def test_noul_decomposition_separates_abstention_from_wrong_direction():
    parent, candidate = report([0.1, 0.9]), report([0.3, 0.7])
    original = copy.deepcopy(candidate)
    result = audit_continuation(parent, candidate, replicates=20)
    noul = result["overall"]["by_type"]["noul"]
    assert noul["transitions"] == {"yes->abstain": 1}
    assert noul["argmax_credit_delta"] == 0
    assert noul["abstention_credit_penalty_delta"] == 1
    assert noul["thresholded_credit_delta"] == -1
    assert result["overall"]["cc_delta"] == -200
    assert candidate == original and result["execution"]["new_gpu_seconds"] == 0
    assert result["test_opened"] is False and "cannot identify" in result["scope"]


def test_wrong_to_abstention_can_improve_nll_without_fixing_decision():
    result = audit_continuation(report([0.99, 0.01]), report([0.4, 0.6]), replicates=20)
    noul = result["overall"]["by_type"]["noul"]
    assert noul["nll_delta"] < 0 and noul["cc_delta"] == 0
    assert noul["argmax_credit_delta"] == noul["abstention_credit_penalty_delta"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("route", "reasoned"), ("budget", 128), ("generated_tokens", 2), ("model_id", "b" * 64)],
)
def test_refuses_wrong_execution_or_checkpoint_binding(field, value):
    candidate = report([0.1, 0.9])
    candidate["rows"]["off"][0][field] = value
    with pytest.raises(ValueError):
        audit_continuation(report([0.1, 0.9]), candidate, replicates=20)


def test_refuses_test_partial_duplicate_and_intersection_comparisons():
    for field, value in [("split", "test"), ("complete", False)]:
        candidate = report([0.1, 0.9])
        candidate[field] = value
        with pytest.raises(ValueError, match="dev"):
            audit_continuation(report([0.1, 0.9]), candidate)
    candidate = report([0.1, 0.9])
    candidate["rows"]["off"].append(copy.deepcopy(candidate["rows"]["off"][0]))
    with pytest.raises(ValueError, match="unique"):
        audit_continuation(report([0.1, 0.9]), candidate)
    candidate = report([0.1, 0.9])
    candidate["rows"]["off"][0]["id"] = "different/q"
    with pytest.raises(ValueError, match="identical cohort"):
        audit_continuation(report([0.1, 0.9]), candidate)


def test_refuses_non_sha256_checkpoint_identity():
    candidate = report([0.1, 0.9])
    candidate["model_id"] = candidate["rows"]["off"][0]["model_id"] = "z" * 64
    with pytest.raises(ValueError, match="checkpoint fingerprint"):
        audit_continuation(report([0.1, 0.9]), candidate, replicates=20)


def test_workload_and_history_are_validated_before_reporting_exposure():
    workload = {"total_rows": 4, "steps": 2, "counts": {"data_kind": {"natural": 1, "authored": 3}}}
    history = [{"step": 1, "total": 2.0}, {"step": 2, "total": 1.0}]
    result = audit_continuation(
        report([0.1, 0.9]), report([0.1, 0.9]), workload, history, replicates=20
    )
    assert result["training_exposure"]["fractions"]["data_kind"]["natural"] == 0.25
    workload["counts"]["data_kind"]["natural"] = 5
    with pytest.raises(ValueError, match="scheduled row"):
        audit_continuation(report([0.1, 0.9]), report([0.1, 0.9]), workload, replicates=20)


def test_score_error_changes_are_separate_and_ordinals_are_respected():
    parent, candidate = report([0.9, 0.1]), report([0.1, 0.9])
    for value in (parent, candidate):
        row = value["rows"]["off"][0]
        row.update(type="score", ordinals=[5, -2])
    result = audit_continuation(parent, candidate, replicates=20)
    score = result["overall"]["by_type"]["score"]
    assert score["improved"] == 1 and "fixed" not in score
    candidate["rows"]["off"][0]["ordinals"] = [5, 5]
    with pytest.raises(ValueError, match="ordinals"):
        audit_continuation(parent, candidate, replicates=20)
