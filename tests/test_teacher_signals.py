import copy
import json
import math

import pytest

from ayaka.training.teacher_signals import prompt_teacher_signal_report


def probabilities(logit):
    true = 1 / (1 + math.exp(-logit))
    return [1 - true, true]


def pair(student, teacher, *, source="natural", kind="noul", accepted=True):
    return {
        "source": source,
        "type": kind,
        "teacher_present": True,
        "teacher_accepted": accepted,
        "candidate_ids": ["false", "true"] if kind == "noul" else ["a", "b"],
        "direct_probs": student,
        "teacher_probs": teacher,
    }


def test_sourcewise_constant_and_variable_shifts_remain_separate_and_diagnostic():
    records = [
        pair(probabilities(x), probabilities(x + 2), source="authored") for x in (-2, -1, 1)
    ] + [
        pair(probabilities(x), probabilities(x + delta), source="natural")
        for x, delta in ((-2, -2), (-1, 0), (1, 2))
    ]
    report = prompt_teacher_signal_report(records)
    authored, natural = [s["noul_true_logit"] for s in report["by_source_type"]]
    assert authored["mean_teacher_minus_student"] == pytest.approx(2)
    assert authored["residual_rmse_after_slice_constant"] == pytest.approx(0, abs=1e-12)
    assert authored["teacher_on_student_ols_slope"] == pytest.approx(1)
    assert authored["pearson_correlation"] == pytest.approx(1)
    assert natural["mean_teacher_minus_student"] == pytest.approx(0)
    assert natural["residual_rmse_after_slice_constant"] == pytest.approx(math.sqrt(8 / 3))
    assert report["eligibility_or_authored_cap_applied"] is False
    assert report["policy_refit_required_before_trained_comparison"] is True
    assert report["promotable"] is report["execution_attested"] is False


def test_filter_coverage_source_mix_entropy_and_kl_do_not_change_targets():
    records = [
        pair([0.6, 0.4], [0.9, 0.1], kind="score"),
        pair([0.6, 0.4], [0.1, 0.9], kind="score", accepted=False),
        {"source": "natural", "type": "score", "teacher_present": False, "teacher_accepted": False},
        pair([0.7, 0.3], [0.9, 0.1], source="authored", kind="choice"),
    ]
    before = copy.deepcopy(records)
    report = prompt_teacher_signal_report(records)
    assert records == before
    natural = next(s for s in report["by_source_type"] if s["source"] == "natural")
    assert natural["observed_pairs"] == 2
    assert natural["accepted_teacher_questions"] == natural["rejected_teacher_questions"] == 1
    assert natural["acceptance_fraction"] == 0.5
    assert natural["no_saved_teacher_gold_replay_questions"] == 1
    assert natural["mean_entropy_change_nats"] < 0
    assert natural["mean_forward_kl_finite_nats"] == pytest.approx(
        (
            0.9 * math.log(0.9 / 0.6)
            + 0.1 * math.log(0.1 / 0.4)
            + 0.1 * math.log(0.1 / 0.6)
            + 0.9 * math.log(0.9 / 0.4)
        )
        / 2
    )
    assert natural["argmax_flips"] == 1
    assert natural["argmax_flip_fraction_of_untied"] == 0.5
    assert natural["accepted_pair_statistics"]["observed_pairs"] == 1
    assert natural["accepted_pair_statistics"]["argmax_flips"] == 0
    assert natural["rejected_pair_statistics"]["argmax_flips"] == 1
    assert report["accepted_source_mix"] == [
        {"source": "authored", "questions": 1, "fraction": 0.5},
        {"source": "natural", "questions": 1, "fraction": 0.5},
    ]


def test_zero_support_endpoints_ties_and_constant_logits_are_explicit_and_json_finite():
    endpoint = prompt_teacher_signal_report(
        [
            pair([0, 1], [0.5, 0.5]),
            pair([0.5, 0.5], [0.5, 0.5]),
        ]
    )
    result = endpoint["by_source_type"][0]
    assert result["forward_kl_nonfinite_pairs"] == 1
    assert result["forward_kl_finite_pairs"] == 1
    assert result["argmax_tied_pairs"] == 2
    assert result["argmax_flip_fraction_of_untied"] is None
    logits = result["noul_true_logit"]
    assert logits["finite_pairs"] == logits["endpoint_pairs"] == 1
    assert logits["residual_rmse_after_slice_constant"] == 0
    assert logits["teacher_on_student_ols_slope"] is logits["pearson_correlation"] is None
    json.dumps(endpoint, allow_nan=False)
    all_endpoints = prompt_teacher_signal_report([pair([0, 1], [1, 0])])["by_source_type"][0]
    assert all_endpoints["mean_forward_kl_finite_nats"] is None
    assert all_endpoints["noul_true_logit"]["mean_teacher_minus_student"] is None


def test_reversed_noul_coordinates_keep_true_logit_orientation():
    original = pair([0.8, 0.2], [0.3, 0.7])
    reverse = copy.deepcopy(original)
    for field in ("candidate_ids", "direct_probs", "teacher_probs"):
        reverse[field].reverse()
    assert prompt_teacher_signal_report([original]) == prompt_teacher_signal_report([reverse])


def test_missing_only_cell_reports_zero_observations_without_fake_pair_statistics():
    report = prompt_teacher_signal_report(
        [
            {
                "source": "natural",
                "type": "choice",
                "teacher_present": False,
                "teacher_accepted": False,
            },
        ]
    )
    row = report["by_source_type"][0]
    assert row["observed_pairs"] == 0 and row["acceptance_fraction"] is None
    assert row["no_saved_teacher_gold_replay_questions"] == 1
    assert "mean_forward_kl_finite_nats" not in row
    assert report["accepted_source_mix"] == []


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(source=""),
        lambda r: r.update(type=[]),
        lambda r: r.update(teacher_present=1),
        lambda r: r.update(teacher_accepted=1),
        lambda r: r.update(teacher_present=False),
        lambda r: r.update(candidate_ids=["false", "false"]),
        lambda r: r.update(candidate_ids=["no", "yes"]),
        lambda r: r.update(direct_probs=[True, False]),
        lambda r: r.update(direct_probs=[float("nan"), 0]),
        lambda r: r.update(direct_probs=[0.9, 0.9]),
        lambda r: r.update(direct_probs=[-0.1, 1.1]),
        lambda r: r.update(teacher_probs=[1.00000001, 0]),
    ],
)
def test_invalid_diagnostic_contract_is_rejected(change):
    row = pair([0.4, 0.6], [0.3, 0.7])
    change(row)
    with pytest.raises(ValueError, match="teacher signal|missing teacher"):
        prompt_teacher_signal_report([row])
