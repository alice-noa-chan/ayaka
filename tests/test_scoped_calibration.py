import pytest

from ayaka.training.scoped_calibration import ScopedCalibration


def rows(count=24):
    return [
        {
            "model_id": "a" * 64,
            "modality": "image",
            "partition": "fixed",
            "split": "calibration",
            "cluster_id": f"source-{i}",
            "type": "choice",
            "route": "direct",
            "budget": 0,
            "probs": [0.9, 0.1],
            "target": [float(i % 2), float(1 - i % 2)],
        }
        for i in range(count)
    ]


def test_image_calibration_is_bound_to_checkpoint_and_domain(tmp_path):
    calibration = ScopedCalibration.fit(rows(), "a" * 64, "image", "fixed")
    assert calibration.apply([0.9, 0.1], "choice", "direct", 0)[0] < 0.9
    assert calibration.apply([0.9, 0.1], "choice", "reasoned", 1024) == [0.9, 0.1]
    calibration.validate_binding("a" * 64, "image", "fixed")
    for identity, modality, partition in [
        ("b" * 64, "image", "fixed"),
        ("a" * 64, "text", "fixed"),
        ("a" * 64, "image", "generated_finite"),
    ]:
        with pytest.raises(ValueError, match="binding"):
            calibration.validate_binding(identity, modality, partition)
    file = tmp_path / "image.json"
    calibration.save(file)
    loaded = ScopedCalibration.load(file)
    assert loaded.calibration.temperatures == calibration.calibration.temperatures


def test_translations_and_repeated_traces_cannot_inflate_independent_calibration_count():
    repeated = rows(3) * 16
    calibration = ScopedCalibration.fit(repeated, "a" * 64, "image", "fixed")
    assert not calibration.calibration.temperatures


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "test"),
        ("modality", "text"),
        ("partition", "generated_finite"),
        ("model_id", "b" * 64),
    ],
)
def test_fitting_rejects_other_split_or_model_domain(field, value):
    data = rows()
    data[0][field] = value
    with pytest.raises(ValueError, match="reserved"):
        ScopedCalibration.fit(data, "a" * 64, "image", "fixed")
