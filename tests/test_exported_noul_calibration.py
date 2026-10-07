"""Standalone exports retain their calibrated probabilities and binding."""

import pytest
import torch
from test_checkpoint_serving_inputs import QUESTIONS, STATE, setup_checkpoint
from test_saved_noul_calibration import attach_correction

from ayaka.export import export_model, load_exported
from ayaka.primitives import Decision
from ayaka.serve import parse_question


def test_native_merged_export_preserves_automatic_noul_correction(tmp_path):
    _, model, tok, _, _ = setup_checkpoint(tmp_path, "gemma")
    correction = attach_correction(model)
    model.backbone = model.backbone.merge_and_unload()
    questions = [parse_question(q)[0] for q in QUESTIONS.values()]
    expected = Decision(model, tok).decide(STATE, questions)
    path = export_model(model, str(tmp_path / "export"), model.cfg.backbone)
    loaded, exported_tok = load_exported(path, dtype=torch.float32)
    assert loaded.noul_calibration.as_dict() == correction.as_dict()
    actual = Decision(loaded, exported_tok).decide(STATE, questions)
    for before, after in zip(expected, actual, strict=True):
        assert after.probs == pytest.approx(before.probs, abs=3e-6)
    with (tmp_path / "export" / "head.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="binding mismatch"):
        load_exported(path, dtype=torch.float32)


def test_quantized_export_does_not_silently_reuse_an_unvalidated_correction(tmp_path):
    _, model, _, _, _ = setup_checkpoint(tmp_path, "gemma")
    attach_correction(model)
    model.backbone = model.backbone.merge_and_unload()
    with pytest.raises(ValueError, match="recalibrat"):
        export_model(model, str(tmp_path / "quantized"), model.cfg.backbone, quantize=True)
    assert not (tmp_path / "quantized").exists()
