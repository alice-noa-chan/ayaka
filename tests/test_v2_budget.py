import copy
import json

import pytest

from ayaka.experiments.budget import reconcile_closed_windows


def evidence():
    return {
        "stage_index": 0,
        "container_id": "container",
        "app_id": "app",
        "closure_evidence": "observed stop succeeded",
        "started_at": "2025-01-01T00:00:00+00:00",
        "ended_before": "2025-01-01T00:10:00+00:00",
        "startup_shutdown_margin_s": 120,
    }


def test_closed_window_reconciliation_is_conservative_and_idempotent(tmp_path):
    ledger = {
        "elapsed_s": 7300,
        "stages": [{"stage": "screen", "status": "running", "allocation_s": 7200}],
    }
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(ledger))
    measured = reconcile_closed_windows(tmp_path, [evidence()])
    assert measured["elapsed_s"] == 820  # 600 observed + 120 overhead + 100 prior overrun
    assert measured["stages"][0]["allocation_s"] == 7200
    assert measured["stages"][0]["charged_s"] == 720
    assert reconcile_closed_windows(tmp_path, [evidence()]) == measured


@pytest.mark.parametrize(
    "field,value",
    [
        ("startup_shutdown_margin_s", 0),
        ("closure_evidence", ""),
        ("ended_before", "2025-01-01T00:00:00+00:00"),
        ("started_at", "2025-01-01T00:00:00"),
        ("stage_index", -1),
    ],
)
def test_unverified_or_invalid_closure_cannot_release_reservations(tmp_path, field, value):
    ledger = {
        "elapsed_s": 7200,
        "stages": [{"stage": "screen", "status": "running", "allocation_s": 7200}],
    }
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(ledger))
    observation = copy.deepcopy(evidence())
    observation[field] = value
    with pytest.raises(ValueError):
        reconcile_closed_windows(tmp_path, [observation])
    assert json.loads(path.read_text()) == ledger


def test_closed_sequential_window_keeps_completed_durations_and_full_tail(tmp_path):
    ledger = {
        "elapsed_s": 10800,
        "stages": [
            {
                "stage": "screen",
                "status": "complete",
                "allocation_s": 3600,
                "actual_elapsed_s": 250,
            },
            {"stage": "sft", "status": "complete", "allocation_s": 3600, "actual_elapsed_s": 100},
            {"stage": "evaluate", "status": "running", "allocation_s": 3600},
        ],
    }
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(ledger))
    observed = evidence()
    del observed["stage_index"]
    observed.update(stage_indices=[0, 1, 2], sequential_stages=True)
    settled = reconcile_closed_windows(tmp_path, [observed])
    assert settled["elapsed_s"] == 720
    assert [r["charged_s"] for r in settled["stages"]] == [250, 100, 370]
    assert settled["stages"][0]["status"] == "complete"
    assert settled["stages"][2]["status"] == "interrupted"
    assert reconcile_closed_windows(tmp_path, [observed]) == settled
    with pytest.raises(ValueError, match="second reservation"):
        reconcile_closed_windows(tmp_path, [{**evidence(), "stage_index": 0}])


def test_sequential_audit_rejects_unmeasured_or_noncontiguous_prefixes(tmp_path):
    ledger = {
        "elapsed_s": 7200,
        "stages": [
            {"stage": "screen", "status": "running", "allocation_s": 3600},
            {"stage": "evaluate", "status": "running", "allocation_s": 3600},
        ],
    }
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(ledger))
    observed = {**evidence(), "stage_indices": [0, 1], "sequential_stages": True}
    with pytest.raises(ValueError, match="measured completion"):
        reconcile_closed_windows(tmp_path, [observed])
    assert json.loads(path.read_text()) == ledger
    with pytest.raises(ValueError, match="contiguous"):
        reconcile_closed_windows(tmp_path, [{**observed, "stage_indices": [1, 0]}])
