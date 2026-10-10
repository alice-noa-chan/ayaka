"""The validated v2 recipe defaults."""

import argparse
import json

import pytest

from ayaka.config import AyakaConfig
from ayaka.training import v2_recipe

# The model config the final v2 run of 2026-10-10 trained with.
FINAL_RUN_MODEL = {
    "backbone": "google/gemma-4-12B-it",
    "backbone_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    "input_contract_required": False,
    "long_prompt_tokens": 1024,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "lora_r": 64,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "max_label_candidates": 26,
    "max_seq_len": 4096,
    "name": "ayaka-v2-final-direct",
    "pointer_dim": 512,
    "readout": "lm",
    "reasoning_defaults": {},
    "serve_max_seq_len": 8192,
    "set_mixer_heads": 8,
    "set_mixer_layers": 3,
    "tiny_overrides": {},
    "version": 2,
}


def test_model_config_cli_writes_the_final_run_config(tmp_path):
    out = tmp_path / "direct-config.json"
    v2_recipe.main(["model-config", "--out", str(out)])
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == FINAL_RUN_MODEL
    AyakaConfig(**written)
    with pytest.raises(SystemExit):
        v2_recipe.main(["model-config", "--out", str(out)])


def parse(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--contractnli-train")
    v2_recipe.add_extra_natural_arguments(parser, ("strategyqa",))
    return parser.parse_args(argv)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ((), ()),
        (("--contractnli-train", "train.json"), ("strategyqa",)),
        (("--contractnli-train", "train.json", "--no-extra-natural"), ()),
        (("--extra-natural", "strategyqa"), ("strategyqa",)),
    ],
)
def test_strategyqa_is_the_plan3_default(argv, expected):
    assert v2_recipe.extra_natural(parse(*argv)) == expected


def test_explicit_and_disabled_extra_sources_conflict():
    with pytest.raises(ValueError, match="exclusive"):
        v2_recipe.extra_natural(parse("--extra-natural", "strategyqa", "--no-extra-natural"))


def test_recipe_learning_rates_are_below_the_diverged_ones():
    assert (v2_recipe.LORA_LR, v2_recipe.HEAD_LR) == (3e-5, 1.5e-4)
    assert v2_recipe.SELECT_EVERY == 250 and v2_recipe.SELECT_PATIENCE == 3
