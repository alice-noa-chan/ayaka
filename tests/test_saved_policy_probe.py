import copy
import json

import pytest

from ayaka.eval.continuation_audit import checked_rows
from ayaka.eval.saved_policy_probe import commit_noul, main, probe_policies


def report(split="dev"):
    rows = []
    for i, target in enumerate([1] * 6 + [0] * 4):
        rows.append(
            {
                "id": f"{split}/{i}",
                "cluster_id": f"{split}-case-{i}",
                "model_id": "a" * 64,
                "split": split,
                "route": "direct",
                "budget": 0,
                "reasoning_tokens": 0,
                "language": "en",
                "family": "calendar",
                "modality": "text",
                "partition": "fixed",
                "type": "noul",
                "probs": [0.4, 0.6],
                "target": [1 - target, target],
                "ordinals": None,
            }
        )
    return {
        "complete": True,
        "split": split,
        "model_id": "a" * 64,
        "cohort_sha256": split,
        "rows": {"off": rows},
    }


def inputs():
    raw, fitted, calibration = report(), report(), report("calibration")
    fitted["calibration"] = "reserved_split_scoped_temperature"
    return raw, fitted, calibration


def test_emitted_commit_can_improve_competence_while_worsening_proper_losses():
    data = inputs()
    original = copy.deepcopy(data)
    result = probe_policies(*data, replicates=20)
    change = result["comparisons"]["calibrated->calibrated_commit"]
    noul = change["by_type"]["noul"]
    assert noul["cc_delta"] == pytest.approx(120)
    assert noul["nll_delta"] > 0 and noul["brier_delta"] > 0 and noul["ece_delta"] > 0
    assert result["execution"] == {"model_forwards": 0, "fitting_steps": 0, "new_gpu_seconds": 0}
    assert result["test_opened"] is False and result["selected_policy"] is None
    assert result["calibration_independent_cases"] == 10 and data == original


@pytest.mark.parametrize("p", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_commit_endpoints_and_half_tie(p):
    raw = report()
    for row in raw["rows"]["off"]:
        row["probs"] = [1 - p, p]
    emitted = commit_noul(checked_rows(raw))
    assert emitted[0]["probs"][1] == (0.801 if p == 0.5 else p)
    assert emitted[0]["argmax_credit"] == emitted[0]["target"][int(emitted[0]["probs"][1] > 0.5)]


@pytest.mark.parametrize("field", ["id", "cluster_id"])
def test_calibration_dev_overlap_is_refused(field):
    raw, fitted, calibration = inputs()
    calibration["rows"]["off"][0][field] = raw["rows"]["off"][0][field]
    with pytest.raises(ValueError, match="overlap"):
        probe_policies(raw, fitted, calibration, replicates=20)


def test_model_split_cohort_and_provenance_binding():
    for mutation, message in [
        (lambda raw, fitted, cal: cal.update(model_id="b" * 64), "checkpoint"),
        (lambda raw, fitted, cal: cal.update(split="test"), "calibration"),
        (lambda raw, fitted, cal: fitted.update(cohort_sha256="different"), "fingerprints"),
        (lambda raw, fitted, cal: fitted.pop("calibration"), "provenance"),
        (lambda raw, fitted, cal: raw.update(calibration="fitted"), "raw arm"),
    ]:
        data = inputs()
        mutation(*data)
        with pytest.raises(ValueError, match=message):
            probe_policies(*data, replicates=20)
    with pytest.raises(ValueError, match="test stays unopened"):
        checked_rows(report("test"), split="test")


def test_score_and_choice_readouts_are_unchanged_by_noul_commit():
    for kind in ("choice", "score"):
        source = report()
        for row in source["rows"]["off"]:
            row.update(type=kind, ordinals=[5, -2] if kind == "score" else None)
        before = checked_rows(source)
        after = commit_noul(before)
        assert all(
            a["probs"] == b["probs"] and a["nll"] == b["nll"]
            for a, b in zip(before, after, strict=True)
        )


def test_cli_hashes_inputs_and_never_overwrites_a_receipt(tmp_path):
    args = []
    for name, data in zip(("raw", "calibrated", "calibration"), inputs(), strict=True):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        args += [f"--{name}", str(path)]
    output = tmp_path / "probe.json"
    args += ["--out", str(output), "--replicates", "20"]
    result = main(args)
    assert len(result["input_sha256"]) == 3 and output.is_file()
    with pytest.raises(ValueError, match="fresh output"):
        main(args)
