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
