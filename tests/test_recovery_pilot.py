import copy

import pytest

from ayaka.data.recovery_v2 import recovery_curriculum
from ayaka.eval.pretraining_v2 import cohort_fingerprint
from scripts.runpod_v2.recovery import question_ids
from scripts.runpod_v2.recovery_pilot import validate_parent_reports


def test_reused_parent_reports_reject_wrong_identity_partial_or_changed_cohort():
    dev = recovery_curriculum("dev", 2)
    report = {
        "model_id": "parent",
        "complete": True,
        "split": "dev",
        "cohort_sha256": cohort_fingerprint(dev),
        "rows": {"off": [{"id": x} for x in question_ids(dev)]},
    }
    validate_parent_reports(report, report, "parent", dev)
    for key, value in (("model_id", "other"), ("complete", False), ("split", "test")):
        changed = copy.deepcopy(report)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_parent_reports(report, changed, "parent", dev)
    changed = copy.deepcopy(report)
    changed["rows"]["off"].reverse()
    with pytest.raises(ValueError):
        validate_parent_reports(changed, report, "parent", dev)
    changed_inputs = copy.deepcopy(dev)
    changed_inputs[0].questions[0].candidates[0].description += " changed"
    with pytest.raises(ValueError):
        validate_parent_reports(report, report, "parent", changed_inputs)
