"""Noul discrimination must separate ranking failures from band abstentions."""

import json

import pytest

from ayaka.eval.noul_discrimination import auc, discrimination, main, noul_rows, state


def row(i, p, y, family="calc"):
    return {"id": f"q{i}", "type": "noul", "p": p, "y": y, "family": family}


def test_auc_counts_ranking_and_ties():
    assert auc([row(0, 0.9, 1), row(1, 0.1, 0)]) == 1.0
    assert auc([row(0, 0.1, 1), row(1, 0.9, 0)]) == 0.0
    assert auc([row(0, 0.5, 1), row(1, 0.5, 0)]) == 0.5
    assert auc([row(0, 0.5, 1)]) is None


def test_band_states_use_the_fixed_thresholds():
    assert state(row(0, 0.8, 1)) == "correct"
    assert state(row(0, 0.2, 0)) == "correct"
    assert state(row(0, 0.79, 1)) == "abstain"
    assert state(row(0, 0.95, 0)) == "wrong"


def test_bias_shift_without_ranking_skill_shows_as_lost_lucky_credit():
    # No ranking skill: P(true) does not depend on gold. A false-leaning
    # parent is credited on gold-false questions; centering removes that.
    parent = {r["id"]: r for r in [row(0, 0.1, 0), row(1, 0.1, 1), row(2, 0.1, 0)]}
    pilot = {r["id"]: r for r in [row(0, 0.5, 0), row(1, 0.5, 1), row(2, 0.5, 0)]}
    report = discrimination({"parent": parent, "pilot": pilot}, "family")
    calc = report["groups"]["calc"]
    assert calc["parent"]["auc"] == calc["pilot"]["auc"] == 0.5
    assert (calc["parent"]["correct"], calc["pilot"]["correct"]) == (2, 0)
    assert calc["pilot"]["abstain"] == 3
    assert report["transitions"] == {"correct->abstain": 2, "wrong->abstain": 1}


def test_soft_targets_are_counted_but_excluded_from_hard_metrics():
    rows = {r["id"]: r for r in [row(0, 0.5, 0.5), row(1, 0.9, 1), row(2, 0.1, 0)]}
    summary = discrimination({"a": rows})["groups"]["all"]["a"]
    assert summary["soft_target_questions"] == 1
    assert summary["hard_target_questions"] == 2
    assert summary["mean_p_true_by_gold"] == {"0": 0.1, "1": 0.9}


def test_systems_must_cover_the_same_questions():
    with pytest.raises(ValueError, match="same Noul questions"):
        discrimination({"a": {"q0": row(0, 0.1, 0)}, "b": {"q1": row(1, 0.1, 0)}})


def test_both_saved_report_formats_load(tmp_path):
    comparison = tmp_path / "rows.jsonl"
    comparison.write_text(
        json.dumps(
            {
                "id": "q0",
                "type": "noul",
                "probs": [0.3, 0.7],
                "candidate_ids": ["false", "true"],
                "target": [0.0, 1.0],
            }
        )
        + "\n"
        + json.dumps({"id": "c0", "type": "choice", "probs": [1.0], "target": [1.0]})
        + "\n"
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "rows": {
                    "off": [
                        {"id": "q0", "type": "noul", "confidence": 0.7, "calibration_target": 1.0}
                    ]
                }
            }
        )
    )
    assert noul_rows(comparison)["q0"]["p"] == noul_rows(report)["q0"]["p"] == 0.7
    out = tmp_path / "out.json"
    main(["--report", f"a={comparison}", "--report", f"b={report}", "--out", str(out)])
    assert json.loads(out.read_text())["transitions"] == {"abstain->abstain": 1}
    with pytest.raises(ValueError, match="new output"):
        main(["--report", f"a={comparison}", "--out", str(out)])
