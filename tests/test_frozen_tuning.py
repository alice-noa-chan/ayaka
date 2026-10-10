"""Serving a checkpoint with its frozen held-out tuning fit."""

import json

import pytest
import torch

from ayaka import frozen_tuning as ft
from ayaka import serve
from ayaka.config import tiny_config
from ayaka.eval import final_gate as fg
from ayaka.model.decision import AyakaDecisionModel
from ayaka.routing import BenefitRouter
from ayaka.tokenization import ToyTokenizer
from ayaka.training.path_calibration import PathCalibration


def frozen(policy="noul_always+router"):
    router = BenefitRouter(
        [0.0] * 9,
        [1.0] * 9,
        [-1.0] + [0.0] * 9,
        [0.1] + [0.0] * 9,
        0.0005,
        True,
        {"validated_budgets": [384]},
    )
    return fg.frozen_fit(PathCalibration({"noul": 1.1}), router, policy, "merged")


@pytest.mark.parametrize(
    ("policy", "kinds"),
    [
        ("direct", ()),
        ("router", ()),
        ("noul_always", ("noul",)),
        ("noul_always+router", ("noul",)),
        ("noul+choice_always+router", ("noul", "choice")),
    ],
)
def test_always_types_follow_the_policy_name(policy, kinds):
    assert ft.always_types(policy) == kinds


def test_frozen_decision_reproduces_the_gate_settings():
    model = AyakaDecisionModel.from_config(tiny_config(version=2), dtype=torch.float32).eval()
    decision = ft.frozen_decision(model, ToyTokenizer(), frozen())
    assert decision.always_types == {"noul"}
    assert decision.router.penalty == 0.0005
    assert decision.calibration.temperatures == {"noul": 1.1}
    assert decision.original.apply_probability_calibration is True
    assert decision.max_seq_len == ft.READ_MAX_SEQ_LEN


def test_policies_without_a_router_serve_without_one():
    model = AyakaDecisionModel.from_config(tiny_config(version=2), dtype=torch.float32).eval()
    assert ft.frozen_decision(model, ToyTokenizer(), frozen("noul_always")).router is None


def test_a_model_folder_ships_its_own_frozen_tuning(tmp_path):
    assert ft.find_frozen(tmp_path) is None
    (tmp_path / ft.FROZEN_FILE).write_text(json.dumps(frozen()), encoding="utf-8")
    assert ft.load_frozen(ft.find_frozen(tmp_path))["policy"] == "noul_always+router"


def test_serve_uses_the_folder_file_unless_reasoning_is_set_by_hand(tmp_path):
    (tmp_path / ft.FROZEN_FILE).write_text(json.dumps(frozen()), encoding="utf-8")
    assert serve._frozen_tuning("auto", tmp_path, False)["policy"] == "noul_always+router"
    assert serve._frozen_tuning("auto", tmp_path, True) is None
    assert serve._frozen_tuning("off", tmp_path, False) is None
    explicit = tmp_path / "other.json"
    explicit.write_text(json.dumps(frozen("noul_always")), encoding="utf-8")
    assert serve._frozen_tuning(str(explicit), None, False)["policy"] == "noul_always"


def test_an_invalid_frozen_file_is_refused(tmp_path):
    broken = {**frozen(), "extra": 1}
    (tmp_path / ft.FROZEN_FILE).write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown or missing fields"):
        serve._frozen_tuning("auto", tmp_path, False)
