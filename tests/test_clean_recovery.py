import copy

import pytest

from ayaka.data.recovery_v2 import recovery_curriculum
from ayaka.eval.v2 import typed_row
from ayaka.primitives import QuestionSpec
from ayaka.training.prepare_recovery import clean_splits
from scripts.runpod_v2.clean_recovery import clean_screen


def report(probability):
    rows = []
    for i in range(240):
        kind = ("choice", "noul", "score")[i % 3]
        spec = QuestionSpec(kind, "", ["0", "1"], [0, 1] if kind == "score" else None)
        target, probs = [1, 0], [probability, 1 - probability]
        row = typed_row(spec, probs, target)
        row.update(
            id=str(i),
            cluster_id=str(i),
            type=kind,
            target=target,
            probs=probs,
            ordinals=spec.ordinals,
            language=("en", "ko", "ja")[(i // 3) % 3],
            family="temporal_numeric",
            latency_s=0,
        )
        rows.append(row)
    return {"complete": True, "split": "dev", "cohort_sha256": "a" * 64, "rows": {"off": rows}}


def test_clean_gate_cannot_hide_raw_regression_behind_bad_parent_calibration():
    parent, pilot = report(0.9), report(0.3)
    bad_parent_cal, pilot_cal = report(0.1), report(0.95)
    result = clean_screen(parent, pilot, bad_parent_cal, pilot_cal, parent, replicates=20)
    assert not result["screen_passed"] and not result["raw"]["screen_passed"]
    assert result["calibrated"]["screen_passed"]
    with pytest.raises(ValueError, match="content-bound"):
        foreign = copy.deepcopy(pilot)
        foreign["cohort_sha256"] = "b" * 64
        clean_screen(parent, foreign, parent, pilot, parent, replicates=20)


def test_clean_gate_accepts_matched_improvements_in_all_three_comparisons():
    parent, pilot = report(0.3), report(0.95)
    result = clean_screen(parent, pilot, parent, pilot, parent, replicates=20)
    assert result["screen_passed"] and not result["full_training_authorized"]


def test_clean_preparation_excludes_old_authored_data_and_inspected_final_test():
    old = {
        split: recovery_curriculum(split, 2)
        for split in ("train", "router_train", "dev", "calibration", "test")
    }
    natural = copy.deepcopy(old["train"][0])
    natural.metadata["data_kind"] = "natural"
    old["train"].append(natural)
    splits, checks = clean_splits(old)
    assert natural in splits["train"]
    assert all(
        s.metadata.get("generation") == 3 or s.metadata.get("data_kind") == "natural"
        for s in splits["train"]
    )
    assert all(s.metadata.get("evaluation_only") for s in splits["test"])
    assert not {s.metadata["source_lineage"] for s in old["test"]} & {
        s.metadata["source_lineage"] for s in splits["test"]
    }
    assert all(r["passed"] for r in checks.values())
