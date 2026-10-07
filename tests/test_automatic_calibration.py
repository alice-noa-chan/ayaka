"""Automatic calibration uses the user's reserved data and survives reloads."""

import json
from pathlib import Path

import pytest
import torch

from ayaka.checkpoint import load_checkpoint
from ayaka.data.schema import Question, Sample
from ayaka.training import run
from ayaka.training.trainer import Trainer


def customer_pools(size):
    return {
        ("noul", "en"): [
            Sample(
                {"case": i, "allowed": bool(i % 2)},
                [Question.noul("q", "Is access allowed?", float(i % 2))],
                {
                    "source": "customer",
                    "source_family": "customer",
                    "source_example_id": f"customer-{i}",
                    "task_family": "noul",
                },
            )
            for i in range(size)
        ]
    }


def config(tmp_path, name, **overrides):
    return run.RunConfig(
        model_size="tiny",
        steps=1,
        questions_per_step=4,
        eval_questions=8,
        eval_every=0,
        fidelity_questions=0,
        jevbench=False,
        decontaminate=False,
        artifacts_dir=str(tmp_path),
        run_name=name,
        log_every=0,
        bf16=False,
        **overrides,
    )


def test_custom_data_is_automatically_calibrated_refitted_and_saved(tmp_path, monkeypatch):
    training_ids, calibration_ids, evaluation_ids = set(), set(), set()
    original_step, original_predict, original_evaluate = (
        Trainer.train_step,
        Trainer.predict,
        Trainer.evaluate,
    )
    fits = []
    temperatures = iter((1.7, 0.6))
    evaluating = False

    def train_step(self, items):
        training_ids.update(it.sample_id for it in items)
        return original_step(self, items)

    def predict(self, items, **kwargs):
        if kwargs.get("apply_temperature") is False and not evaluating:
            calibration_ids.update(it.sample_id for it in items)
        return original_predict(self, items, **kwargs)

    def evaluate(self, items, **kwargs):
        nonlocal evaluating
        evaluation_ids.update(it.sample_id for it in items)
        evaluating = True
        try:
            return original_evaluate(self, items, **kwargs)
        finally:
            evaluating = False

    def fit(logits, targets, types, lengths, long_threshold):
        fits.append(len(logits))
        assert len(logits) == len(targets) == len(types) == len(lengths) == 24
        assert set(types) == {"noul"}
        return {"noul": next(temperatures)}

    def no_external_data(*args, **kwargs):
        raise AssertionError("default calibration must use only the customer's reserved data")

    monkeypatch.setattr(Trainer, "train_step", train_step)
    monkeypatch.setattr(Trainer, "predict", predict)
    monkeypatch.setattr(Trainer, "evaluate", evaluate)
    monkeypatch.setattr(run, "fit_temperatures", fit)
    monkeypatch.setattr(run, "items_from_spec", no_external_data)
    for name, expected_temperature in (("first", 1.7), ("second", 0.6)):
        training_ids.clear()
        calibration_ids.clear()
        evaluation_ids.clear()
        pools = customer_pools(240)
        result = run.run_training(config(tmp_path, name), pools=pools, verbose=False)
        assert training_ids and calibration_ids and evaluation_ids
        assert training_ids.isdisjoint(calibration_ids | evaluation_ids)
        assert calibration_ids.isdisjoint(evaluation_ids)
        assert len(pools[("noul", "en")]) == 208
        report = json.loads((tmp_path / name / "calibration.json").read_text())
        assert report == result["calibration"]
        assert report["status"] == "fitted"
        assert report["source"] == "reserved_training_groups"
        assert report["fitted_types"] == ["noul"]
        assert report["unfitted_types"] == []
        assert report["noul_correction"]["report"]["status"] == "insufficient_data"
        assert result["raw_heldout"]["n"] == result["heldout"]["n"] == 8
        assert result["raw_heldout"]["accuracy"] == result["heldout"]["accuracy"]
        assert result["raw_heldout"]["nll"] != pytest.approx(result["heldout"]["nll"])
        loaded = load_checkpoint(result["checkpoint"], device="cpu", dtype=torch.float32)
        assert loaded.noul_calibration is not None and not loaded.noul_calibration.selected
        from ayaka.model.decision import PRIMITIVE_INDEX

        assert loaded.temperature[PRIMITIVE_INDEX["noul"]].tolist() == pytest.approx(
            [expected_temperature, expected_temperature]
        )
        metadata = json.loads((Path(result["checkpoint"]) / "meta.json").read_text())
        assert metadata["calibration"] == report
    assert fits == [24, 24]


def test_small_custom_dataset_retains_training_and_reports_insufficient_calibration(tmp_path):
    pools = customer_pools(10)
    result = run.run_training(config(tmp_path, "small"), pools=pools, verbose=False)
    assert len(pools[("noul", "en")]) == 8
    assert result["calibration"]["questions"] == 1
    assert result["calibration"]["status"] == "insufficient_data"
    assert result["calibration"]["unfitted_types"] == ["noul"]
    assert result["temperatures"] == {}


def test_eval_and_calibration_caps_preserve_cross_language_lineages():
    from ayaka.config import tiny_config
    from ayaka.tokenization import ToyTokenizer

    pools = customer_pools(100)
    pools[("noul", "ko")] = [
        Sample("Translated " + str(s.state), s.questions, dict(s.metadata))
        for s in pools[("noul", "en")]
    ]
    calibration = run.reserve_calibration(pools, ToyTokenizer(), tiny_config(), max_fraction=0.1)
    evaluation = run.split_eval(pools, 1000, 0, max_fraction=0.1)
    calibrated = {it.sample_id for it in calibration}
    evaluated = {s.metadata["source_example_id"] for s in evaluation}
    trained = {s.metadata["source_example_id"] for cell in pools.values() for s in cell}
    assert len(calibrated) == 10 and len(evaluated) == 9 and len(trained) == 81
    assert calibrated.isdisjoint(evaluated | trained) and evaluated.isdisjoint(trained)
    assert all(len(cell) == 81 for cell in pools.values())
    groups = {}
    for item in calibration:
        groups.setdefault(item.sample_id, set()).add(item.calibration_group)
    assert all(len(group) == 1 for group in groups.values())


def test_new_training_refits_and_validates_automatic_noul_correction(tmp_path, monkeypatch):
    import math

    from ayaka.training.noul_calibration import NoulCalibration

    fits = []
    biases = iter((0.1, -0.1))

    def predict(self, items, *, apply_temperature=True, return_logits=False):
        probabilities = [[0.3, 0.7] if it.target[1] else [0.7, 0.3] for it in items]
        logits = [[math.log(p) for p in row] for row in probabilities]
        correction = getattr(self.model, "noul_calibration", None)
        if apply_temperature and correction is not None:
            probabilities = [
                correction.apply(p, it.type, it.length)
                for it, p in zip(items, probabilities, strict=True)
            ]
        return (probabilities, logits) if return_logits else probabilities

    def fit(rows, temperatures, long_threshold):
        assert len(rows) == 40
        assert all(r["split"] == "calibration" for r in rows)
        assert len({r["cluster_id"] for r in rows}) == 40
        fits.append({r["cluster_id"] for r in rows})
        return NoulCalibration(
            3.0, next(biases), temperatures, long_threshold, {"status": "selected"}
        )

    monkeypatch.setattr(Trainer, "predict", predict)
    monkeypatch.setattr(run, "fit_temperatures", lambda *args: {"noul": 1.0})
    monkeypatch.setattr(run, "fit_noul_calibration", fit)
    for name, bias in (("calibrated-first", 0.1), ("calibrated-second", -0.1)):
        cfg = config(tmp_path, name)
        cfg.eval_questions = 32
        result = run.run_training(cfg, pools=customer_pools(400), verbose=False)
        loaded = load_checkpoint(result["checkpoint"], dtype=torch.float32)
        correction = loaded.noul_calibration
        assert correction.selected and correction.bias == bias
        validation = correction.report["validation"]
        assert validation["independent_cases"] == 32
        assert validation["corrected"]["abstentions"] == 0
        assert validation["baseline"]["abstentions"] == 32
        assert result["calibration"]["noul_correction"] == correction.as_dict()
    assert len(fits) == 2
