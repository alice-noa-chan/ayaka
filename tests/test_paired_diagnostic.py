"""Bound fixtures for subset bias, probability regressions and case uncertainty."""

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from test_swift_bias import bound_synthetic
from test_swift_router import paired_rows

from ayaka.eval.read_artifact import fingerprint
from scripts.direct_v2.paired_diagnostic import (
    _clusters,
    _interval,
    _metrics,
    _summary,
    diagnose_roles,
)


def rebind(row, **changes):
    row.update(changes)
    row["binding"].update(changes)
    row["binding"].pop("binding_sha256")
    row["binding"]["binding_sha256"] = fingerprint(row["binding"])
    row["binding_sha256"] = row["binding"]["binding_sha256"]
    row.pop("record_sha256")
    row["record_sha256"] = fingerprint(row)


def cohorts(tmp_path):
    roles = {}
    for split in ("calibration", "dev"):
        specs = [
            (("choice", "noul", "score")[i % 3], [0, 1] if i < 6 else [1, 0], 0) for i in range(9)
        ]
        direct = bound_synthetic(tmp_path, split, "min", specs)
        for i, row in enumerate(direct):
            rebind(row, source="authored" if i < 6 else "natural")
        paired = paired_rows(tmp_path, direct, [[2, 0]] * 6 + [[-4, 0]] * 3)
        roles[split] = {"direct": direct, "paired": paired}
    return roles


def anchors(roles):
    return {
        role: fingerprint(sorted(block["direct"], key=lambda row: (row["source"], row["id"])))
        for role, block in roles.items()
    }


def test_exposes_natural_regression_hidden_by_pooled_brier_and_typed_denominators(tmp_path):
    roles = cohorts(tmp_path)
    report = diagnose_roles(roles, anchors(roles), iterations=100)
    dev = report["results"]["dev"]
    all_pairs = dev["cohorts"]["all_observed_pairs"]
    assert all_pairs["all"]["metrics"]["brier"]["reasoned_minus_direct"] < 0
    natural = all_pairs["by_source"]["natural"]
    assert natural["metrics"]["brier"]["reasoned_minus_direct"] > 0
    assert natural["metrics"]["nll"]["reasoned_minus_direct"] > 0
    assert natural["metrics"]["rps"]["reasoned_minus_direct"] > 0
    assert all_pairs["all"]["metrics"]["rps"]["pairs"] == 3
    assert all_pairs["all"]["metrics"]["nll"]["pairs"] == 9
    assert natural["connected_cases"] == 1
    assert natural["metrics"]["nll"]["delta_ci95"] is None
    assert dev["coverage"]["by_source"]["natural"]["canonical_observed_fraction"] == 1
    assert report["promotable"] is False and report["training_data_created"] is False
    assert report["policy_fitted_or_changed"] is False and report["execution_attested"] is False
    serialized = json.dumps(report)
    assert "PRIVATE" not in serialized and '"state"' not in serialized
    assert "question-weighted" in report["bootstrap"]["estimand"]


def test_sparse_and_empty_observations_remain_visible_without_invented_gain(tmp_path):
    roles = cohorts(tmp_path)
    roles["calibration"]["paired"] = roles["calibration"]["paired"][:3]
    roles["dev"]["paired"] = []
    report = diagnose_roles(roles, anchors(roles), iterations=100)
    assert report["results"]["calibration"]["coverage"]["all"]["canonical_unobserved"] == 6
    natural = report["results"]["calibration"]["coverage"]["by_source"]["natural"]
    assert natural["canonical_observed_fraction"] == 0 and natural["observed_pairs"] == 0
    dev = report["results"]["dev"]
    assert dev["coverage"]["all"]["canonical_unobserved"] == 9
    assert dev["trace_tokens_all_observed"] == 0
    assert dev["cohorts"]["all_observed_pairs"]["all"]["metrics"] == {}


def test_cap_and_empty_trace_stay_in_usage_and_observed_cohort_not_completed_cohort(tmp_path):
    roles = cohorts(tmp_path)
    pair = roles["dev"]["paired"][0]
    nested = pair["reasoned_read"]
    nested.update(finish_reason="length", length_capped=True)
    pair.pop("record_sha256")
    pair["record_sha256"] = fingerprint(pair)
    pair = roles["dev"]["paired"][1]
    nested = pair["reasoned_read"]
    nested["pass_inputs"][0]["messages"][1]["content"] = " "
    nested["trace_tokens"] = 0
    nested["output_tokens"] = 1
    pair.pop("record_sha256")
    pair["record_sha256"] = fingerprint(pair)
    report = diagnose_roles(roles, anchors(roles), iterations=100)
    dev = report["results"]["dev"]
    assert dev["cohorts"]["all_observed_pairs"]["all"]["pairs"] == 9
    assert dev["completed_nonempty_eos_pairs"] == 7
    assert dev["finish_reasons"] == {"eos": 8, "length": 1}
    assert dev["trace_tokens_all_observed"] == sum(
        row["reasoned_read"]["trace_tokens"] for row in roles["dev"]["paired"]
    )


@pytest.mark.parametrize(
    "problem",
    [
        "anchor",
        "public",
        "test",
        "nested_direct",
        "tampered_pair",
        "mixed_model",
        "mixed_recipe",
        "overlap",
    ],
)
def test_rejects_invalid_scope_anchors_pairs_or_comparability(tmp_path, problem):
    roles = cohorts(tmp_path)
    expected = anchors(roles)
    if problem == "anchor":
        expected["dev"] = "0" * 64
    elif problem in ("public", "test"):
        roles["dev"]["direct"][0]["public" if problem == "public" else "split"] = (
            True if problem == "public" else "test"
        )
    elif problem == "nested_direct":
        roles["dev"]["direct"][0]["reasoned_read"] = {}
    elif problem == "tampered_pair":
        roles["dev"]["paired"][0]["reasoned_read"]["raw_probs"]["label-0"] = 0.9
    elif problem == "mixed_model":
        rebind(roles["dev"]["direct"][0], model="other")
        expected = anchors(roles)
    elif problem == "mixed_recipe":
        roles["dev"]["paired"][0]["reasoned_read"]["recipe"]["max_tokens"] = 1025
    else:
        roles["dev"] = copy.deepcopy(roles["calibration"])
        for row in roles["dev"]["direct"]:
            rebind(row, split="dev")
        roles["dev"]["paired"] = []
        expected = anchors(roles)
    with pytest.raises(ValueError):
        diagnose_roles(roles, expected, iterations=100)


def test_sorting_contract_does_not_depend_on_artifact_line_order(tmp_path):
    roles = cohorts(tmp_path)
    expected = anchors(roles)
    first = diagnose_roles(roles, expected, iterations=100, seed=123)
    for block in roles.values():
        block["direct"].reverse()
        block["paired"].reverse()
    assert first == diagnose_roles(roles, expected, iterations=100, seed=123)


def test_unobserved_bridge_joins_ancestry_and_exact_state_transitively(tmp_path):
    rows = cohorts(tmp_path)["dev"]["direct"]
    # A bridge need not have a saved trace to make the other two rows correlated.
    a, bridge, c = copy.deepcopy([rows[0], rows[3], rows[6]])
    bridge["lineage_ids"] = [a["case_id"]]
    c["binding"]["state"] = bridge["binding"]["state"]
    result = _clusters([a, bridge, c])
    assert len(set(result.values())) == 1
    same = _clusters([c, a, bridge])
    assert len(set(same.values())) == 1


def test_explicit_unqualified_ancestry_joins_across_sources_conservatively(tmp_path):
    rows = copy.deepcopy(cohorts(tmp_path)["dev"]["direct"])
    rows[0]["lineage_ids"] = ["shared-document"]
    rows[-1]["lineage_ids"] = ["shared-document"]
    components = _clusters(rows)
    assert rows[0]["source"] != rows[-1]["source"]
    assert components[rows[0]["id"]] == components[rows[-1]["id"]]


def test_same_global_ancestry_relation_rejects_cross_source_role_leakage(tmp_path):
    roles = cohorts(tmp_path)
    rebind(roles["calibration"]["direct"][0], lineage_ids=["shared-document"])
    rebind(roles["dev"]["direct"][-1], lineage_ids=["shared-document"])
    roles["calibration"]["paired"] = []
    roles["dev"]["paired"] = []
    with pytest.raises(ValueError, match="connected ancestry"):
        diagnose_roles(roles, anchors(roles), iterations=100)


def test_cluster_bootstrap_retains_question_weighting_and_refuses_false_precision():
    items = [
        {
            "cluster": 0,
            "direct": {"nll": 0.0, "hard_argmax_matches_gold": 0.0},
            "reasoned": {"nll": 1.0, "hard_argmax_matches_gold": 0.0},
        }
    ] * 9
    items += [
        {
            "cluster": 1,
            "direct": {"nll": 1.0, "hard_argmax_matches_gold": 0.0},
            "reasoned": {"nll": 0.0, "hard_argmax_matches_gold": 0.0},
        }
    ]
    result = _summary(items, iterations=100, seed=31)
    assert result["metrics"]["nll"]["reasoned_minus_direct"] == pytest.approx(0.8)
    assert result["metrics"]["nll"]["delta_ci95"] == [-1, 1]
    assert result["connected_cases"] == 2
    one = _interval(items[:9], "nll", 100, 31)
    assert one["delta_ci95"] is None
    assert _interval(items, "nll", 100, 31) == _interval(items, "nll", 100, 31)


def test_extreme_logits_use_logspace_nll_and_soft_score_ordinal_metrics():
    row = {
        "type": "score",
        "labels": ["0", "2", "10"],
        "gold": "0",
        "gold_distribution": {"0": 0.5, "2": 0.5},
        "raw_probs": {"0": 0.0, "2": 0.0, "10": 1.0},
        "candidate_log_masses": {"0": -2000, "2": -1000, "10": 0},
    }
    result = _metrics(row)
    assert result["nll"] == pytest.approx(1500)
    assert result["brier"] == pytest.approx(1.5)
    assert result["rps"] == pytest.approx(0.625)
    assert result["normalized_expected_absolute_error"] == pytest.approx(0.9)
    row["labels"] = ["0", "10", "2"]
    with pytest.raises(ValueError, match="numerically ordered"):
        _metrics(row)


@pytest.mark.parametrize(
    "iterations,seed", [(0, 1), (True, 1), (10001, 1), (100, True), (100, math.inf)]
)
def test_rejects_invalid_bootstrap_settings(iterations, seed):
    with pytest.raises(ValueError, match="bootstrap"):
        diagnose_roles(
            {role: {} for role in ("calibration", "dev")},
            dict.fromkeys(("calibration", "dev"), "a" * 64),
            iterations=iterations,
            seed=seed,
        )


def test_cli_checks_physical_anchors_and_preserves_existing_report(tmp_path):
    import hashlib

    roles = cohorts(tmp_path)
    expected = anchors(roles)
    command = [sys.executable, "-m", "scripts.direct_v2.paired_diagnostic", "--iterations", "100"]
    for role, block in roles.items():
        command += [f"--expected-{role}-cohort-sha256", expected[role]]
        for kind, rows in block.items():
            path = tmp_path / f"{role}-{kind}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            command += [
                f"--{role}-{kind}",
                str(path),
                f"--expected-{role}-{kind}-sha256",
                hashlib.sha256(path.read_bytes()).hexdigest(),
            ]
    out = tmp_path / "report.json"
    command += ["--out", str(out)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    original = out.read_bytes()
    report = json.loads(original)
    assert (
        report["file_anchors"]["dev_paired"]["sha256"]
        == hashlib.sha256((tmp_path / "dev-paired.jsonl").read_bytes()).hexdigest()
    )
    assert (
        report["diagnostic_script_sha256"]
        == hashlib.sha256(Path("scripts/direct_v2/paired_diagnostic.py").read_bytes()).hexdigest()
    )
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode != 0 and out.read_bytes() == original
    command[-1] = str(tmp_path / "wrong-hash.json")
    command[command.index("--expected-dev-paired-sha256") + 1] = "0" * 64
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode != 0 and not (tmp_path / "wrong-hash.json").exists()
