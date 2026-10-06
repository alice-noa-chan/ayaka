import math

import pytest

from ayaka.data.reasoning_v2 import SPLITS, curriculum
from ayaka.eval.v2 import assert_isolated, paired_report, select_candidates, summarize, typed_row
from ayaka.primitives import QuestionSpec
from ayaka.training.path_calibration import PathCalibration, path_key


def test_logit_nll_preserves_underflow_and_soft_targets_without_changing_legacy_rows():
    spec = QuestionSpec("choice", "Pick?", ["a", "b"])
    row = typed_row(spec, [1, 0], [0.5, 0.5], logits=[0, -1000])
    assert row["nll"] == 500 and row["nll_source"] == "logits"
    legacy = typed_row(spec, [1, 0], [0.5, 0.5])
    assert legacy["nll"] == pytest.approx(-0.5 * math.log(1e-12))
    assert "nll_source" not in legacy
    assert {k: v for k, v in row.items() if k not in {"nll", "nll_source"}} == {
        k: v for k, v in legacy.items() if k != "nll"
    }


def test_logit_nll_shifts_before_normalizing_large_equal_values():
    spec = QuestionSpec("choice", "Pick?", ["a", "b"])
    assert typed_row(spec, [0.5, 0.5], [1, 0], logits=[1e30, 1e30])["nll"] == pytest.approx(
        math.log(2)
    )


@pytest.mark.parametrize(
    "logits,error",
    [
        ([0], "aligned"),
        ([0, float("nan")], "finite"),
        ([0, float("inf")], "finite"),
        ([1e308, -1e308], "span"),
        ([0, 1], "different distributions"),
    ],
)
def test_logit_nll_rejects_nonfinite_misaligned_or_inconsistent_inputs(logits, error):
    spec = QuestionSpec("choice", "Pick?", ["a", "b"])
    with pytest.raises(ValueError, match=error):
        typed_row(spec, [0.5, 0.5], [1, 0], logits=logits)


def test_noul_abstention_thresholds_and_yes_calibration():
    spec = QuestionSpec("noul", "Test?", ["no", "yes"])
    for p, correct in [(0.2, 0), (0.8, 1), (0.7, 0), (0.5, 0)]:
        row = typed_row(spec, [1 - p, p], [0, 1])
        assert row["correct"] == correct
        assert row["confidence"] == p and row["calibration_target"] == 1
    assert typed_row(spec, [0.8, 0.2], [1, 0])["correct"] == 1


def test_score_expected_position_uses_order_not_argmax_or_ordinal_spacing():
    spec = QuestionSpec("score", "Rating?", ["low", "mid", "high"], [0, 5, 100])
    row = typed_row(spec, [0.4, 0.35, 0.25], [0, 1, 0])
    assert row["expected_position"] == pytest.approx(0.85)
    assert row["nmae"] == pytest.approx(0.075)
    assert row["nmae_chance"] == pytest.approx(1 / 3)
    assert row["rps"] == pytest.approx((0.4**2 + 0.25**2) / 2)


def test_type_balancing_paired_fixes_and_invalid_probabilities():
    spec = QuestionSpec("choice", "Pick?", ["a", "b"])
    direct = [
        dict(typed_row(spec, [0.9, 0.1], [1, 0]), id="a"),
        dict(typed_row(spec, [0.8, 0.2], [0, 1]), id="b"),
    ]
    reasoned = [
        dict(typed_row(spec, [0.1, 0.9], [1, 0]), id="a"),
        dict(typed_row(spec, [0.2, 0.8], [0, 1]), id="b"),
    ]
    report = paired_report(direct, reasoned, replicates=30)
    assert report["fixed"] == report["broken"] == 1
    assert summarize(direct)["official_composite"] is None
    with pytest.raises(ValueError):
        typed_row(spec, [float("nan"), 0], [1, 0])
    with pytest.raises(ValueError):
        paired_report(direct, list(reversed(reasoned)))


def test_paired_uncertainty_clusters_repeated_underlying_cases():
    spec = QuestionSpec("choice", "Pick?", ["a", "b"])
    direct, reasoned = [], []
    for case in range(3):
        target = [0, 1] if case == 0 else [1, 0]
        for copy in range(10):
            metadata = {"id": f"{case}/{copy}", "cluster_id": f"case/{case}"}
            direct.append(dict(typed_row(spec, [0.9, 0.1], target), **metadata))
            reasoned.append(
                dict(typed_row(spec, [0.1, 0.9] if case < 2 else [0.9, 0.1], target), **metadata)
            )
    clustered = paired_report(direct, reasoned)
    naive = paired_report(
        [{k: v for k, v in row.items() if k != "cluster_id"} for row in direct], reasoned
    )
    assert clustered["independent_cases"] == 3
    assert naive["independent_cases"] == 30
    assert clustered["cc_delta_95ci"][1] > naive["cc_delta_95ci"][1]


def test_score_error_changes_are_not_reported_as_broken_correct_answers():
    spec = QuestionSpec("score", "Rating?", ["low", "high"], [0, 1])
    direct = [
        dict(typed_row(spec, probs, [1, 0]), id=str(i))
        for i, probs in enumerate(([0.8, 0.2], [0.9, 0.1], [1, 0]))
    ]
    reasoned = [
        dict(typed_row(spec, probs, [1, 0]), id=str(i))
        for i, probs in enumerate(([1, 0], [0.6, 0.4], [1 - 1e-12, 1e-12]))
    ]
    report = paired_report(direct, reasoned, replicates=30)
    assert report["fixed"] == report["broken"] == 0
    assert report["score_improved"] == report["score_worsened"] == 1
    assert report["score_count_tolerance"] == 1e-6
    assert report["mean_nll_gain"] != 0  # continuous metrics keep actual differences


def test_curriculum_splits_do_not_overlap_and_detect_shared_templates():
    splits = {split: [s for s, _ in curriculum(split)] for split in SPLITS}
    assert_isolated(splits)
    splits["test"][0].metadata["generator_template_id"] = splits["train"][0].metadata[
        "generator_template_id"
    ]
    with pytest.raises(ValueError, match="template"):
        assert_isolated(splits)


def test_case_isolation_checks_facts_even_when_document_voice_differs():
    splits = {split: [s for s, _ in curriculum(split, 128)] for split in SPLITS}
    assert_isolated(splits)
    splits["test"][0].metadata["case_facts_sha256"] = splits["train"][0].metadata[
        "case_facts_sha256"
    ]
    with pytest.raises(ValueError, match="case_facts"):
        assert_isolated(splits)


def test_calibration_does_not_fit_dev_or_test_and_preserves_permutation():
    with pytest.raises(ValueError, match="reserved"):
        PathCalibration.fit([{"split": "dev"}])
    cal = PathCalibration({"choice/reasoned/high": 2})
    assert path_key("choice", "reasoned", 1000) == "choice/reasoned/high"
    assert cal.apply([0.8, 0.2], "choice", "reasoned", 1024) == pytest.approx(
        list(reversed(cal.apply([0.2, 0.8], "choice", "reasoned", 1024)))
    )


def test_candidate_selection_never_promotes_incomplete_results():
    r = {
        "status": "complete",
        "n": 96,
        "by_type": {t: {"n": 32} for t in ("choice", "noul", "score")},
        "cc_equal_types": 80,
        "proper_loss": 0.5,
        "p95_s": 1,
    }
    best = dict(r, name="best", cc_equal_types=80.5, proper_loss=0.6)
    incomplete = dict(r, status="incomplete", cc_equal_types=100)
    assert select_candidates([best, r, incomplete]) == [r, best]


def test_serial_effort_runner_reports_deadline_incomplete_and_never_uses_gold():
    import time

    from ayaka.eval.reasoning_v2 import evaluate_efforts
    from ayaka.primitives import DecisionResult

    class Fake:
        from ayaka.tokenization import ToyTokenizer

        tok = ToyTokenizer()
        calls = []

        def decide(self, state, questions, reasoning):
            self.calls.append(reasoning[0].mode)
            assert not hasattr(questions[0], "target_distribution")
            q = questions[0]
            p = [1 / len(q.candidates)] * len(q.candidates)
            return [
                DecisionResult(
                    q.type,
                    p,
                    {},
                    extras={
                        "reasoning": {
                            "generated_tokens": 0,
                            "input_tokens": 12,
                            "route": "direct",
                            "budget": reasoning[0].budget,
                            "finish_reason": "disabled",
                            "error": None,
                        }
                    },
                )
            ]

    samples = [s for s, _ in curriculum("dev", 1)]
    report = evaluate_efforts(Fake(), samples, efforts=())
    assert report["status"] == "complete" and report["reports"]["off"]["n"] == 3
    stopped = evaluate_efforts(Fake(), samples, deadline=time.monotonic() - 1)
    assert stopped["status"] == "incomplete" and not stopped["reports"]
    restored = evaluate_efforts(
        Fake(), samples, efforts=(), deadline=time.monotonic() - 1, resume_rows=report["rows"]
    )
    assert restored["status"] == "complete" and restored["rows"] == report["rows"]
    Fake.calls.clear()
    partial = {"off": report["rows"]["off"][:2]}
    continued = evaluate_efforts(Fake(), samples, efforts=(), resume_rows=partial)
    assert Fake.calls == ["off"] and continued["status"] == "complete"
    assert continued["rows"]["off"][:2] == partial["off"]
    bad = {"off": [dict(report["rows"]["off"][0], budget=128)]}
    with pytest.raises(ValueError, match="budgets"):
        evaluate_efforts(Fake(), samples, efforts=(), resume_rows=bad)
