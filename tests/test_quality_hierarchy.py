"""A development quality claim needs three matched systems and no hidden regression."""

import copy
import json

import pytest

from ayaka.data.schema import Candidate, Question, Sample
from ayaka.eval.pretraining_v2 import cohort_fingerprint
from ayaka.eval.quality_hierarchy import hierarchy_screen, main
from ayaka.eval.v2 import typed_row
from ayaka.primitives import QuestionSpec

MODEL_IDS = {"v1": "a" * 64, "v2": "b" * 64}


@pytest.fixture(scope="module")
def observations():
    samples = []
    v1, v2 = [], {"off": [], "high": []}
    for index in range(240):
        kind = ("choice", "noul", "score")[index % 3]
        width = 3 if kind == "score" else 2
        q = (
            Question.noul("q", "Is the proposition true?", 0)
            if kind == "noul"
            else Question(
                "q",
                kind,
                "Use evidence",
                [
                    Candidate(str(i), f"Option {i}", i if kind == "score" else None)
                    for i in range(width)
                ],
                {"0": 1.0},
            )
        )
        metadata = {
            "source_example_id": str(index),
            "source_lineage": f"case/{index}",
            "split": "dev",
            "language": ("en", "ko", "ja")[(index // 3) % 3],
            "source": "natural" if index % 2 else "procedural",
            "task_family": "calendar",
        }
        samples.append(Sample(f"Evidence {index}", [q], metadata))
        target = [1.0, *([0.0] * (width - 1))]
        ordinals = list(range(width)) if kind == "score" else None
        fields = {
            "id": f"{index}/q",
            "type": kind,
            "target": target,
            "candidate_ids": [c.id for c in q.candidates],
            "ordinals": ordinals,
            "cluster_id": metadata["source_lineage"],
            "split": "dev",
            "language": metadata["language"],
            "family": "calendar",
            "source": metadata["source"],
            "tier": "standard",
            "modality": "text",
            "partition": "fixed",
        }
        for system, mode, probability in (
            ("v1", "high", 0.1),
            ("v2", "off", 0.85),
            ("v2", "high", 0.95),
        ):
            probs = [probability, *([(1 - probability) / (width - 1)] * (width - 1))]
            row = {
                **fields,
                **typed_row(
                    QuestionSpec(kind, "", [c.description for c in q.candidates], ordinals),
                    probs,
                    target,
                ),
                "probs": probs,
                "model_id": MODEL_IDS[system],
                "route": "direct" if mode == "off" else "reasoned",
                "budget": 0 if mode == "off" else 1024,
                "reasoning_tokens": 0 if mode == "off" else 4,
                "finish_reason": "direct" if mode == "off" else "eos",
                "error": None,
            }
            (v1 if system == "v1" else v2[mode]).append(row)
    common = {"complete": True, "split": "dev", "cohort_sha256": cohort_fingerprint(samples)}
    return samples, {**common, "rows": {"high": v1}}, {**common, "rows": v2}


def screen(samples, v1, v2):
    return hierarchy_screen(
        v1, v2, samples, v1_model_id=MODEL_IDS["v1"], v2_model_id=MODEL_IDS["v2"], replicates=30
    )


def test_three_system_gain_passes_only_as_an_experimental_dev_screen(observations):
    samples, v1, v2 = observations
    original = copy.deepcopy(observations)
    result = screen(samples, v1, v2)
    assert result["screen_passed"] and result["final_test_required"]
    assert not result["promotable"] and result["official_composite"] is None
    assert result["comparisons"]["v2_on_over_v2_off"]["paired"]["independent_cases"] == 240
    assert result["comparisons"]["v2_off_over_v1_on"]["checks"]["classification_credit_gain"]
    assert observations == original


def test_cached_metrics_and_row_order_cannot_determine_the_result(observations):
    samples, v1, v2 = copy.deepcopy(observations)
    for rows in v2["rows"].values():
        rows.reverse()
        for row in rows:
            row.update(nmae=1000, rps=1000, nll=1000, brier=1000, correct=0)
    assert screen(samples, v1, v2)["screen_passed"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("target", [0, 1]),
        ("candidate_ids", ["1", "0"]),
        ("cluster_id", "other"),
        ("type", "noul"),
        ("tier", "hard"),
        ("source", "other"),
        ("ordinals", [0, 1]),
    ],
)
def test_scoring_fields_must_match_canonical_prepared_questions(observations, field, value):
    samples, v1, v2 = copy.deepcopy(observations)
    v2["rows"]["off"][0][field] = value
    with pytest.raises(ValueError, match="canonical field"):
        screen(samples, v1, v2)


@pytest.mark.parametrize(
    "mutation",
    ["partial", "duplicate", "foreign_model", "wrong_cohort", "incomplete", "off_generation"],
)
def test_partial_foreign_or_wrong_mode_results_are_rejected(observations, mutation):
    samples, v1, v2 = copy.deepcopy(observations)
    if mutation == "partial":
        v2["rows"]["high"].pop()
    elif mutation == "duplicate":
        v2["rows"]["high"].append(v2["rows"]["high"][0])
    elif mutation == "foreign_model":
        v2["rows"]["high"][0]["model_id"] = "c" * 64
    elif mutation == "wrong_cohort":
        v2["cohort_sha256"] = "c" * 64
    elif mutation == "incomplete":
        v2["complete"] = False
    else:
        v2["rows"]["off"][0]["reasoning_tokens"] = 1
    with pytest.raises(ValueError):
        screen(samples, v1, v2)


def test_better_rps_cannot_hide_worse_score_nmae(observations):
    samples, v1, v2 = copy.deepcopy(observations)
    for sample in samples:
        if sample.questions[0].type == "score":
            sample.questions[0].target_distribution = {"1": 1.0}
    for report in (v1, v2):
        report["cohort_sha256"] = cohort_fingerprint(samples)
        for mode, rows in report["rows"].items():
            for row in rows:
                if row["type"] == "score":
                    row["target"] = [0, 1, 0]
                    row["probs"] = [0.4, 0.2, 0.4] if mode == "off" else [0.1, 0.7, 0.2]
                elif mode == "off" and row["type"] == "choice":
                    row["probs"] = [0.4, 0.6]
    comparison = screen(samples, v1, v2)["comparisons"]["v2_on_over_v2_off"]
    assert comparison["cc_delta"] > 0
    assert comparison["checks"]["score/rps_not_worse"]
    assert not comparison["checks"]["score/nmae_not_worse"]
    assert not comparison["screen_passed"]


def test_no_gain_and_all_fallback_reasoning_do_not_pass(observations):
    samples, v1, v2 = copy.deepcopy(observations)
    v2["rows"]["high"] = copy.deepcopy(v2["rows"]["off"])
    for row in v2["rows"]["high"]:
        row.update(route="fallback", budget=1024)
    result = screen(samples, v1, v2)
    assert not result["screen_passed"] and not result["v2_reasoning_completed"]
    assert not result["comparisons"]["v2_on_over_v2_off"]["checks"]["cc_gain"]


def test_independent_case_count_is_not_the_number_of_questions(observations):
    samples, v1, v2 = copy.deepcopy(observations)
    for sample in samples:
        sample.metadata["source_lineage"] = "one-shared-case"
    for report in (v1, v2):
        report["cohort_sha256"] = cohort_fingerprint(samples)
        for rows in report["rows"].values():
            for row in rows:
                row["cluster_id"] = "one-shared-case"
    result = screen(samples, v1, v2)
    assert not result["screen_passed"]
    assert not result["comparisons"]["v2_off_over_v1_on"]["checks"]["enough_independent_cases"]


def test_overall_improvement_cannot_hide_a_natural_source_regression(observations):
    samples, v1, v2 = copy.deepcopy(observations)
    for row in v2["rows"]["high"]:
        if row["source"] == "natural" and row["type"] == "score":
            row["probs"] = [0.83, 0.085, 0.085]
    comparison = screen(samples, v1, v2)["comparisons"]["v2_on_over_v2_off"]
    assert comparison["cc_delta"] > 0 and comparison["checks"]["score/cc_not_worse"]
    assert not comparison["checks"]["source:natural/score/cc_not_worse"]
    assert not comparison["screen_passed"]


@pytest.mark.parametrize("value", [0, True, -1])
def test_invalid_bootstrap_repetition_counts_are_rejected(observations, value):
    samples, v1, v2 = observations
    with pytest.raises(ValueError, match="positive integer"):
        hierarchy_screen(
            v1,
            v2,
            samples,
            v1_model_id=MODEL_IDS["v1"],
            v2_model_id=MODEL_IDS["v2"],
            replicates=value,
        )


def test_empty_reasoned_results_cannot_claim_completed_reasoning(observations):
    samples, v1, v2 = copy.deepcopy(observations)
    v2["rows"]["high"][0]["reasoning_tokens"] = 0
    with pytest.raises(ValueError, match="reasoning controls"):
        screen(samples, v1, v2)


def test_holdout_samples_are_refused(observations):
    samples, v1, v2 = copy.deepcopy(observations)
    samples[0].metadata["split"] = "test"
    with pytest.raises(ValueError, match="test stays unopened"):
        screen(samples, v1, v2)


def test_cli_records_consumed_inputs_and_never_overwrites_a_report(observations, tmp_path):
    samples, v1, v2 = observations
    args = []
    for name, report in (("v1-report", v1), ("v2-report", v2)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        args.extend([f"--{name}", str(path)])
    sample_path = tmp_path / "dev.jsonl"
    sample_path.write_text(
        "".join(json.dumps(s.to_json()) + "\n" for s in samples), encoding="utf-8"
    )
    output = tmp_path / "screen.json"
    args += [
        "--samples",
        str(sample_path),
        "--out",
        str(output),
        "--replicates",
        "30",
        "--v1-model-id",
        MODEL_IDS["v1"],
        "--v2-model-id",
        MODEL_IDS["v2"],
    ]
    assert main(args) == 0
    result = json.loads(output.read_bytes())
    assert set(result["input_sha256"]) == {"v1_report", "v2_report", "samples"}
    with pytest.raises(ValueError, match="written once"):
        main(args)
