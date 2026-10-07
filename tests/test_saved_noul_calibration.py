"""Saved probability corrections reach native serving without changing weights."""

import json

import pytest
import torch
from test_checkpoint_serving_inputs import ENCODING, QUESTIONS, STATE, expected, setup_checkpoint

from ayaka.checkpoint import compact_checkpoint, load_checkpoint, save_checkpoint
from ayaka.model.decision import AyakaDecisionModel
from ayaka.primitives import Decision
from ayaka.reasoning import ReasoningSettings
from ayaka.reasoning_pipeline import controlled_decision
from ayaka.serve import parse_question
from ayaka.training.noul_calibration import NoulCalibration
from ayaka.training.path_calibration import PathCalibration
from ayaka.training.scoped_calibration import checkpoint_fingerprint
from ayaka.training.trainer import TrainConfig, Trainer


def attach_correction(model):
    short, long = model.temperature[0].tolist()
    model.noul_calibration = NoulCalibration(
        scale=1.5,
        bias=0.4,
        input_temperatures={"noul@short": short, "noul@long": long},
        long_threshold=model.cfg.long_prompt_tokens,
        report={"status": "selected"},
    )
    return model.noul_calibration


def test_saved_native_correction_matches_training_and_service_after_reload(tmp_path):
    _, model, tok, checkpoint, _ = setup_checkpoint(tmp_path)
    correction = attach_correction(model)
    save_checkpoint(model, str(checkpoint))
    base_id = checkpoint_fingerprint(checkpoint, include_calibration=False)
    assert json.loads((checkpoint / "noul_calibration.json").read_text())["model_id"] == base_id
    assert checkpoint_fingerprint(checkpoint) != base_id
    loaded = load_checkpoint(str(checkpoint), dtype=torch.float32, merge=False)
    assert loaded.noul_calibration.as_dict() == correction.as_dict()
    questions = [parse_question(q)[0] for q in QUESTIONS.values()]
    raw = Decision(loaded, tok)
    raw.apply_probability_calibration = False
    direct = Decision(loaded, tok)
    trainer = Trainer(loaded, tok, TrainConfig(steps=1), torch.device("cpu"))
    for state in (STATE, {"facts": "Evidence. " * 40}):
        items, _ = expected(loaded, tok, state, questions, ENCODING)
        before = raw.decide(state, questions)
        after = direct.decide(state, questions)
        probabilities, logits = trainer.predict(items, return_logits=True)
        for index, (item, old, new) in enumerate(zip(items, before, after, strict=True)):
            want = correction.apply(old.probs, item.type, item.length)
            assert new.probs == pytest.approx(want, abs=3e-6)
            assert probabilities[index] == pytest.approx(want, abs=3e-6)
            assert torch.tensor(logits[index]).softmax(0).tolist() == pytest.approx(want, abs=3e-6)
            assert list(new.distribution.values()) == new.probs
            if item.type == "noul":
                assert new.extras["p_true"] == new.probs[1]
                assert new.probs != pytest.approx(old.probs)
            else:
                assert new.probs == old.probs
        uncalibrated = trainer.predict(items, apply_temperature=False)
        loaded.noul_calibration = None
        assert trainer.predict(items, apply_temperature=False) == uncalibrated
        loaded.noul_calibration = correction


def test_changed_weights_reject_binding_before_backbone_load(tmp_path, monkeypatch):
    _, model, _, checkpoint, _ = setup_checkpoint(tmp_path)
    attach_correction(model)
    save_checkpoint(model, str(checkpoint))
    with (checkpoint / "head.safetensors").open("ab") as stream:
        stream.write(b"changed")

    def forbid(*args, **kwargs):
        pytest.fail("mismatched correction reached native weight loading")

    monkeypatch.setattr(AyakaDecisionModel, "from_config", forbid)
    with pytest.raises(ValueError, match="binding mismatch"):
        load_checkpoint(str(checkpoint))


def test_resaving_without_correction_removes_stale_artifact(tmp_path):
    _, model, _, checkpoint, _ = setup_checkpoint(tmp_path)
    attach_correction(model)
    save_checkpoint(model, str(checkpoint))
    model.noul_calibration = None
    save_checkpoint(model, str(checkpoint))
    assert not (checkpoint / "noul_calibration.json").exists()
    loaded = load_checkpoint(str(checkpoint), dtype=torch.float32, merge=False)
    assert loaded.noul_calibration is None


def test_resumed_training_does_not_keep_an_old_probability_correction(tmp_path):
    _, model, _, checkpoint, _ = setup_checkpoint(tmp_path)
    attach_correction(model)
    save_checkpoint(model, str(checkpoint))
    loaded = load_checkpoint(str(checkpoint), dtype=torch.float32, trainable=True)
    assert loaded.noul_calibration is None


def test_changed_serving_recipe_rejects_correction_before_loading(tmp_path, monkeypatch):
    from ayaka.eval.read_artifact import fingerprint
    from ayaka.training.swift_direct import input_serving_recipe

    _, model, tok, checkpoint, _ = setup_checkpoint(tmp_path)
    attach_correction(model)
    save_checkpoint(model, str(checkpoint))
    meta = json.loads((checkpoint / "meta.json").read_text())
    meta["input_encoding"]["prompt_variant"] = "min"
    meta["input_recipe"] = input_serving_recipe(tok, meta["input_encoding"])
    meta["input_recipe_sha256"] = fingerprint(meta["input_recipe"])
    (checkpoint / "meta.json").write_text(json.dumps(meta))

    def forbid(*args, **kwargs):
        pytest.fail("changed serving recipe reached weight loading")

    monkeypatch.setattr(AyakaDecisionModel, "from_config", forbid)
    with pytest.raises(ValueError, match="recipe mismatch"):
        load_checkpoint(str(checkpoint))


def test_explicit_path_calibration_replaces_automatic_direct_correction(tmp_path):
    _, model, tok, _, _ = setup_checkpoint(tmp_path)
    attach_correction(model)
    path = PathCalibration({"noul/direct/low": 1.0})
    controlled = controlled_decision(model, tok, calibration=path)
    assert not controlled.original.apply_probability_calibration
    raw = Decision(model, tok)
    raw.apply_probability_calibration = False
    question = parse_question(QUESTIONS["judge"])[0]
    result = controlled.decide(STATE, [question], reasoning=[ReasoningSettings(mode="off")])[0]
    assert result.probs == pytest.approx(raw.decide(STATE, [question])[0].probs)


def test_reasoned_read_does_not_reuse_direct_correction(tmp_path):
    from test_checkpoint_serving_inputs import favor_trace_token

    _, model, tok, _, _ = setup_checkpoint(tmp_path)
    favor_trace_token(model, tok)
    controlled = controlled_decision(model, tok)
    question = parse_question(QUESTIONS["judge"])[0]
    settings = [ReasoningSettings(mode="on", max_tokens=3)]
    before = controlled.decide(STATE, [question], reasoning=settings)[0]
    attach_correction(model)
    after = controlled.decide(STATE, [question], reasoning=settings)[0]
    assert before.extras["reasoning"]["route"] == after.extras["reasoning"]["route"] == "reasoned"
    assert after.probs == before.probs


def test_malformed_and_wrong_domain_corrections_are_rejected(tmp_path):
    _, model, _, checkpoint, _ = setup_checkpoint(tmp_path)
    attach_correction(model)
    save_checkpoint(model, str(checkpoint))
    path = checkpoint / "noul_calibration.json"
    value = json.loads(path.read_text())
    value["modality"] = "image"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="domain"):
        load_checkpoint(str(checkpoint))


def test_compacting_corrected_weights_requires_recalibration(tmp_path):
    _, model, _, checkpoint, _ = setup_checkpoint(tmp_path)
    attach_correction(model)
    save_checkpoint(model, str(checkpoint))
    with pytest.raises(ValueError, match="recalibrat"):
        compact_checkpoint(str(checkpoint), str(tmp_path / "compact"))
    assert not (tmp_path / "compact").exists()
