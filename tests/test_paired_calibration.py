"""Calibration leakage, immutable observations and underflow-safe diagnostics."""

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest
from test_paired_diagnostic import anchors, cohorts
from test_swift_bias import bound_synthetic
from test_swift_router import paired_rows

from ayaka.eval.read_artifact import fingerprint
from scripts.direct_v2 import paired_calibration as diagnostic
from scripts.direct_v2.paired_diagnostic import diagnose_roles


def soft_roles(tmp_path, *, shifted_dev=False):
    roles = {}
    for split in ("calibration", "dev"):
        shifted = split == "dev" and shifted_dev
        specs = [
            (
                ("choice", "noul", "score")[i % 3],
                [2 if shifted else 1, 0],
                [0.4, 0.6] if shifted else [0.8, 0.2],
            )
            for i in range(9)
        ]
        direct = bound_synthetic(tmp_path, split, "min", specs)
        paired = paired_rows(tmp_path, direct, [[-4, 0] if shifted else [4, 0]] * 9)
        roles[split] = {"direct": direct, "paired": paired}
    return roles


def test_path_specific_logmass_fits_use_identical_calibration_membership_only(
    tmp_path, monkeypatch
):
    roles = soft_roles(tmp_path)
    before = copy.deepcopy(roles)
    calls = []
    fitter = diagnostic.fit_temperature

    def observed_fit(rows):
        calls.append(copy.deepcopy(rows))
        assert all(row["split"] == "calibration" and row["public"] is False for row in rows)
        return fitter(rows)

    monkeypatch.setattr(diagnostic, "fit_temperature", observed_fit)
    report = diagnostic.diagnose_calibration(roles, anchors(roles), iterations=100)
    assert roles == before
    assert len(calls) == 6
    for direct, reasoned in zip(calls[:3], calls[3:], strict=True):
        assert [row["id"] for row in direct] == [row["id"] for row in reasoned]
        assert [row["gold_distribution"] for row in direct] == [
            row["gold_distribution"] for row in reasoned
        ]
        assert all(list(row["candidate_log_masses"].values()) == [1, 0] for row in direct)
        assert all(list(row["candidate_log_masses"].values()) == [4, 0] for row in reasoned)
    fits = report["fit"]["parameters"]
    for kind in diagnostic.TYPES:
        assert fits["direct"][kind]["temperature"] == pytest.approx(1 / math.log(4), abs=1e-6)
        assert fits["reasoned"][kind]["temperature"] == pytest.approx(4 / math.log(4), abs=1e-6)
        for path in fits:
            assert fits[path][kind]["completed_pairs"] == 3
            assert fits[path][kind]["source_counts"] == {"synthetic-private": 3}
    assert report["promotable"] is False
    assert report["training_data_created"] is False
    assert report["serving_policy_changed"] is False
    assert report["execution_attested"] is False
    assert '"state"' not in json.dumps(report)
    assert set(report["dev_comparisons"]) == set(diagnostic.ARMS) | {"uniform_baseline"}


def test_dev_targets_and_logits_do_not_influence_fits(tmp_path):
    ordinary = soft_roles(tmp_path)
    alternative = tmp_path / "different-dev"
    alternative.mkdir()
    shifted = soft_roles(alternative, shifted_dev=True)
    first = diagnostic.diagnose_calibration(ordinary, anchors(ordinary), iterations=100)
    second = diagnostic.diagnose_calibration(shifted, anchors(shifted), iterations=100)
    assert first["fit"] == second["fit"]
    assert first["dev_comparisons"] != second["dev_comparisons"]


def test_line_permutation_preserves_fits_and_all_connected_case_intervals(tmp_path):
    roles = cohorts(tmp_path)
    expected = anchors(roles)
    first = diagnostic.diagnose_calibration(roles, expected, iterations=100, seed=119)
    for block in roles.values():
        block["direct"].reverse()
        block["paired"].reverse()
    assert first == diagnostic.diagnose_calibration(roles, expected, iterations=100, seed=119)


def test_raw_baseline_and_ordering_are_preserved_after_positive_temperature(tmp_path):
    roles = cohorts(tmp_path)
    report = diagnostic.diagnose_calibration(roles, anchors(roles), iterations=100)
    raw = diagnose_roles(roles, anchors(roles), iterations=100)
    assert report["dev_comparisons"]["raw_paths"] == raw["results"]["dev"]["cohorts"]
    for name in ("all_observed_pairs", "completed_nonempty_eos"):
        baseline = report["dev_comparisons"]["raw_paths"][name]["all"]
        for arm in diagnostic.ARMS:
            current = report["dev_comparisons"][arm][name]["all"]
            assert (
                current["hard_argmax_wrong_to_correct"] == baseline["hard_argmax_wrong_to_correct"]
            )
            assert (
                current["hard_argmax_correct_to_wrong"] == baseline["hard_argmax_correct_to_wrong"]
            )


def test_caps_and_empty_calibration_traces_are_excluded_from_fit_but_not_usage(tmp_path):
    roles = soft_roles(tmp_path)
    pairs = roles["calibration"]["paired"]
    pairs[0]["reasoned_read"].update(finish_reason="length", length_capped=True)
    pairs[1]["reasoned_read"]["pass_inputs"][0]["messages"][1]["content"] = " "
    pairs[1]["reasoned_read"].update(trace_tokens=0, output_tokens=1)
    for pair in pairs[:2]:
        pair.pop("record_sha256")
        pair["record_sha256"] = fingerprint(pair)
    report = diagnostic.diagnose_calibration(roles, anchors(roles), iterations=100)
    assert report["fit"]["completed_pairs"] == 7
    for path in report["fit"]["parameters"].values():
        assert [path[kind]["completed_pairs"] for kind in diagnostic.TYPES] == [2, 2, 3]
    provenance = report["observation_provenance"]["calibration"]
    assert provenance["coverage"]["all"]["observed_pairs"] == 9
    assert provenance["trace_tokens_all_observed"] == sum(
        row["reasoned_read"]["trace_tokens"] for row in pairs
    )


def test_missing_calibration_pairs_produce_explicit_identity_not_invented_fit(tmp_path):
    roles = soft_roles(tmp_path)
    roles["calibration"]["paired"] = []
    report = diagnostic.diagnose_calibration(roles, anchors(roles), iterations=100)
    assert report["fit"]["completed_pairs"] == 0
    for path in report["fit"]["parameters"].values():
        for fit in path.values():
            assert fit["temperature"] == 1
            assert fit["completed_pairs"] == 0
            assert fit["nll_after"] is None
            assert fit["status"] == "no_completed_calibration_pairs_identity"
    assert (
        report["dev_comparisons"]["raw_paths"]
        == report["dev_comparisons"]["separately_calibrated_paths"]
    )


def test_underflow_is_recovered_from_log_masses_without_probability_floor():
    row = {
        "type": "score",
        "labels": ["0", "10"],
        "gold": "0",
        "raw_probs": {"0": 0, "10": 1},
        "candidate_log_masses": {"0": -2000, "10": 0},
    }
    result = diagnostic._scaled_metrics(row, 10)
    assert result["nll"] == pytest.approx(200)
    assert result["hard_argmax_matches_gold"] == 0
    assert result["normalized_expected_absolute_error"] == pytest.approx(1)
    uniform = diagnostic._uniform_metrics(row)
    assert uniform["nll"] == pytest.approx(math.log(2))
    assert uniform["rps"] == pytest.approx(0.25)
    assert uniform["normalized_expected_absolute_error"] == pytest.approx(0.5)


@pytest.mark.parametrize("problem", ["anchor", "public", "test", "invalid_pair"])
def test_scope_and_binding_failure_happens_before_fitting(tmp_path, monkeypatch, problem):
    roles = soft_roles(tmp_path)
    expected = anchors(roles)
    if problem == "anchor":
        expected["dev"] = "0" * 64
    elif problem in ("public", "test"):
        roles["dev"]["direct"][0]["public" if problem == "public" else "split"] = (
            True if problem == "public" else "test"
        )
    else:
        roles["dev"]["paired"][0]["reasoned_read"]["candidate_log_masses"]["label-0"] = 7

    def forbidden(_rows):
        raise AssertionError("fitter called before input validation")

    monkeypatch.setattr(diagnostic, "fit_temperature", forbidden)
    with pytest.raises(ValueError):
        diagnostic.diagnose_calibration(roles, expected, iterations=100)


def cli_args(tmp_path):
    roles = soft_roles(tmp_path)
    expected = anchors(roles)
    args = ["paired_calibration", "--iterations", "100"]
    for role, block in roles.items():
        args += [f"--expected-{role}-cohort-sha256", expected[role]]
        for kind, rows in block.items():
            path = tmp_path / f"{role}-{kind}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            args += [
                f"--{role}-{kind}",
                str(path),
                f"--expected-{role}-{kind}-sha256",
                hashlib.sha256(path.read_bytes()).hexdigest(),
            ]
    out = tmp_path / "report.json"
    return [*args, "--out", str(out)], out


def test_cli_anchors_fitter_and_consumed_inputs_and_preserves_existing_report(
    tmp_path, monkeypatch
):
    args, out = cli_args(tmp_path)
    monkeypatch.setattr(sys, "argv", args)
    diagnostic.main()
    saved = out.read_bytes()
    report = json.loads(saved)
    assert (
        report["source_sha256"]["ayaka/swift/fit.py"]
        == hashlib.sha256(Path("ayaka/swift/fit.py").read_bytes()).hexdigest()
    )
    assert (
        report["file_anchors"]["dev_paired"]["sha256"]
        == hashlib.sha256((tmp_path / "dev-paired.jsonl").read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="fresh report"):
        diagnostic.main()
    assert out.read_bytes() == saved
    args[-1] = str(tmp_path / "invalid.json")
    args[args.index("--expected-dev-paired-sha256") + 1] = "0" * 64
    with pytest.raises(ValueError, match="external file anchor"):
        diagnostic.main()
    assert not (tmp_path / "invalid.json").exists()


def test_cli_source_drift_refuses_report_without_editing_real_sources(tmp_path, monkeypatch):
    args, out = cli_args(tmp_path)
    monkeypatch.setattr(sys, "argv", args)
    read_bytes = Path.read_bytes
    calls = 0

    def changed_source(path):
        nonlocal calls
        raw = read_bytes(path)
        if path.resolve() == Path(diagnostic.__file__).resolve():
            calls += 1
            if calls > 1:
                return raw + b"\n# simulated drift\n"
        return raw

    monkeypatch.setattr(Path, "read_bytes", changed_source)
    with pytest.raises(ValueError, match="source changed"):
        diagnostic.main()
    assert not out.exists()
