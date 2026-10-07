"""Direct Noul correction respects independent evidence and probability losses."""

import math

import pytest

from ayaka.training import noul_calibration as calibration


def rows(count=80, target=1.0):
    return [
        {
            "type": "noul",
            "split": "calibration",
            "candidate_ids": ["false", "true"],
            "cluster_id": f"source-{i}",
            "tokens": 100,
            "logits": [0.0, 0.0],
            "target": [1 - target, target],
        }
        for i in range(count)
    ]


def stub_fits(monkeypatch, observed):
    def temperature(logits, targets, types, lengths, threshold):
        return {"noul": 1.0}

    def affine(group, regularization):
        observed.append({r["cluster_id"] for r in group})
        return 1.0, 2.0

    monkeypatch.setattr(calibration, "fit_temperatures", temperature)
    monkeypatch.setattr(calibration, "_fit_affine", affine)


def test_cross_validation_keeps_whole_sources_out_and_selects_improved_correction(monkeypatch):
    observed = []
    stub_fits(monkeypatch, observed)
    data = rows()
    data += [{**r, "tokens": 500} for r in data]
    result = calibration.fit_noul_calibration(data, {"noul": 1.0})
    assert result.selected
    assert result.report["independent_cases"] == 80
    assert result.report["questions"] == 160
    assert len(observed) == 11
    assert len(observed[-1]) == 80
    for f in range(5):
        expected = {
            r["cluster_id"]
            for r in data
            if int(calibration.hashlib.sha256(r["cluster_id"].encode()).hexdigest(), 16) % 5 != f
        }
        assert observed[2 * f] == observed[2 * f + 1] == expected
    reference = result.report["out_of_fold_baseline"]
    selected = result.report["out_of_fold_candidates"][str(result.report["regularization"])]
    assert selected["nll"] < reference["nll"]
    assert selected["brier"] < reference["brier"]
    assert selected["abstentions"] == 0 < reference["abstentions"]


def test_fewer_abstentions_cannot_override_worse_probability_losses(monkeypatch):
    stub_fits(monkeypatch, [])
    result = calibration.fit_noul_calibration(rows(target=0.5), {"noul": 1.0})
    assert not result.selected
    assert result.report["out_of_fold_candidates"]["0.01"]["abstentions"] == 0
    assert result.apply([0.5, 0.5], "noul", 100) == [0.5, 0.5]


@pytest.mark.parametrize("target,accepted", [(1.0, True), (0.0, False), (0.5, False)])
def test_frozen_fit_needs_independent_validation_gains(target, accepted):
    fitted = calibration.NoulCalibration(bias=2.0, report={"status": "selected"})
    heldout = [{**r, "split": "evaluation", "probs": [0.5, 0.5]} for r in rows(40, target=target)]
    parameters = fitted.scale, fitted.bias
    fitted.validate(heldout)
    assert fitted.selected is accepted
    assert (fitted.scale, fitted.bias) == parameters
    if not accepted:
        assert fitted.report["status"] == "rejected_validation"
        assert fitted.apply([0.5, 0.5], "noul", 100) == [0.5, 0.5]


def test_insufficient_validation_and_overlapping_sources_are_not_accepted():
    fitted = calibration.NoulCalibration(bias=2.0, report={"status": "selected"})
    heldout = [{**r, "split": "evaluation", "probs": [0.5, 0.5]} for r in rows(40)]
    fitted.validate(heldout[:2] * 100)
    assert fitted.report["status"] == "insufficient_validation"
    assert not fitted.selected
    fitted.report = {
        "status": "selected",
        "source_groups_sha256": [calibration.hashlib.sha256(b"source-0").hexdigest()],
    }
    with pytest.raises(ValueError, match="overlaps"):
        fitted.validate(heldout)


def test_repeated_questions_do_not_create_independent_sources(monkeypatch):
    def forbidden(*args):
        pytest.fail("insufficient source groups must not be fitted")

    monkeypatch.setattr(calibration, "_fit_affine", forbidden)
    result = calibration.fit_noul_calibration(rows(2) * 100, {"noul": 1.0})
    assert result.report["status"] == "insufficient_data"
    assert result.report["questions"] == 200
    assert result.report["independent_cases"] == 2


def test_temperature_is_reversed_once_and_only_for_direct_noul():
    fitted = calibration.NoulCalibration(
        2.0, 0.4, {"noul@short": 0.5, "noul@long": 2.0}, 1024, {"status": "selected"}
    )
    raw_margin = 0.7
    for tokens, temperature in ((100, 0.5), (2048, 2.0)):
        p = calibration._sigmoid(raw_margin / temperature)
        actual = fitted.apply([1 - p, p], "noul", tokens)
        assert actual[1] == pytest.approx(calibration._sigmoid(2 * raw_margin + 0.4))
        assert sum(actual) == pytest.approx(1)
        assert fitted.apply([1 - p, p], "noul", tokens, route="reasoned") == [1 - p, p]
        assert fitted.apply([1 - p, p], "score", tokens) == [1 - p, p]
    assert calibration.NoulCalibration.from_dict(fitted.as_dict()) == fitted
    assert all(math.isfinite(p) and p > 0 for p in fitted.apply([0.0, 1.0], "noul", 100))


@pytest.mark.parametrize("split", ["dev", "test", "train", None])
def test_non_calibration_rows_are_rejected(split):
    data = rows(1)
    data[0]["split"] = split
    with pytest.raises(ValueError, match="reserved calibration"):
        calibration.fit_noul_calibration(data, {})


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_ids": ["true", "false"]},
        {"cluster_id": ""},
        {"logits": [0, float("nan")]},
        {"target": [0.4, 0.4]},
        {"tokens": 0},
    ],
)
def test_invalid_source_or_binary_contract_is_rejected(changes):
    with pytest.raises(ValueError, match="binary raw logits"):
        calibration.fit_noul_calibration([{**rows(1)[0], **changes}], {})


@pytest.mark.parametrize("scale,bias", [(0, 0), (1, float("nan")), (1, 6)])
def test_invalid_correction_parameters_are_rejected(scale, bias):
    with pytest.raises(ValueError, match="invalid direct Noul"):
        calibration.NoulCalibration(scale, bias)


def test_regularized_affine_fit_learns_a_binary_bias():
    data = rows()
    for i, row in enumerate(data):
        margin = (i % 9 - 4) / 2
        target = calibration._sigmoid(1.7 * margin + 1.0)
        row.update(logits=[0.0, margin], target=[1 - target, target])
    scale, bias = calibration._fit_affine(data, 0.01)
    assert 1 < scale < 2
    assert 0.5 < bias < 1.2
    old = calibration._metrics(data, [calibration._sigmoid(r["logits"][1]) for r in data])
    new = calibration._metrics(
        data, [calibration._sigmoid(scale * r["logits"][1] + bias) for r in data]
    )
    assert new["nll"] < old["nll"]
    assert new["brier"] < old["brier"]
