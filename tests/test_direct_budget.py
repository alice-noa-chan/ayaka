import copy

import pytest

from ayaka.training.direct_budget import STAGES, VERSION, admit_workflow


def budget():
    return {
        "version": VERSION,
        "hourly_usd": 2,
        "prepaid_usd": 4,
        "other_reserved_usd": 0.5,
        "billing_quantum_seconds": 60,
        "safety_factor": 1.25,
        "stages": {
            stage: {"seconds": 120, "basis": "declared fixture allowance"} for stage in STAGES
        },
    }


def test_admission_counts_setup_download_test_recovery_and_billing_rounding():
    cfg = budget()
    result = admit_workflow(cfg)
    assert result["forecast_total_seconds"] == 1650
    assert result["rounded_billed_seconds"] == 1680
    assert result["forecast_total_usd"] == pytest.approx(1680 * 2 / 3600 + 0.5)
    assert result["fits"] and result["partial_curriculum"] is False
    cfg["prepaid_usd"] = result["forecast_total_usd"] - 0.001
    assert not admit_workflow(cfg)["fits"]
    assert cfg["stages"]["training"]["seconds"] == 120


def test_remeasurement_keeps_every_pending_stage_and_already_billed_time():
    cfg = budget()
    result = admit_workflow(
        cfg,
        measured_training_seconds=2000,
        elapsed_seconds=450,
        completed=(
            "environment_setup",
            "weight_download",
            "teacher_collection",
            "model_load",
            "preflight",
        ),
    )
    assert result["forecast_total_seconds"] == 450 + (2000 + 5 * 120) * 1.25
    assert result["stages"]["evaluation"]["seconds"] == 120
    assert "preflight" not in result["stages"]
    assert "artifact_download" in result["stages"]
    assert cfg["stages"]["training"]["seconds"] == 120


@pytest.mark.parametrize(
    "bad", ["missing_stage", "nan", "blank_basis", "quantum", "margin", "balance"]
)
def test_incomplete_or_nonfinite_budget_is_an_error(bad):
    cfg = copy.deepcopy(budget())
    if bad == "missing_stage":
        del cfg["stages"]["weight_download"]
    elif bad == "nan":
        cfg["stages"]["evaluation"]["seconds"] = float("nan")
    elif bad == "blank_basis":
        cfg["stages"]["training"]["basis"] = " "
    elif bad == "quantum":
        cfg["billing_quantum_seconds"] = 0
    elif bad == "margin":
        cfg["safety_factor"] = 0.5
    else:
        cfg["prepaid_usd"] = True
    with pytest.raises(ValueError):
        admit_workflow(cfg)


def test_exact_prepaid_boundary_and_invalid_remeasurement():
    cfg = budget()
    cfg.update(hourly_usd=3.6, prepaid_usd=2.18)
    assert admit_workflow(cfg)["fits"]
    cfg["prepaid_usd"] = 2.179
    assert not admit_workflow(cfg)["fits"]
    with pytest.raises(ValueError):
        admit_workflow(cfg, completed=("training",), measured_training_seconds=10)
    with pytest.raises(ValueError):
        admit_workflow(cfg, completed=("model_load", "model_load"))
