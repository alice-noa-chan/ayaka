from dataclasses import replace

import pytest
import torch

from ayaka.backbone import tiny_text_config
from ayaka.config import AYAKA_LARGE, tiny_config
from ayaka.training.direct_preflight import inspect_direct_model


def test_meta_inspection_preserves_rng_and_does_not_load_weights(monkeypatch):
    from transformers import AutoModelForCausalLM

    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **k: pytest.fail("weights must not load"),
    )
    torch.manual_seed(27)
    state = torch.random.get_rng_state().clone()
    report = inspect_direct_model(tiny_config(readout="lm"))
    assert torch.equal(state, torch.random.get_rng_state())
    assert report["materialized_parameter_bytes"] == 0
    assert report["weights_loaded"] is False
    assert report["forward_calls"] == report["backward_calls"] == report["optimizer_steps"] == 0
    assert report["conservative_total_parameters"] > report["native_text_parameters"]
    assert set(report["lora_targets"]) == set(tiny_config().lora_targets)
    assert report["output_tied_to_input"]
    assert (
        inspect_direct_model(tiny_config(readout="lm", max_seq_len=2048))["serving_context"] == 2048
    )


def test_cached_untied_head_and_actual_lora_positions_are_counted(tmp_path, monkeypatch):
    import huggingface_hub
    from transformers import AutoConfig

    calls = []
    text = tiny_text_config()
    text.tie_word_embeddings = False
    text.save_pretrained(tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(tmp_path))

    def config(repo, **kwargs):
        calls.append((repo, kwargs))
        return text

    monkeypatch.setattr(AutoConfig, "from_pretrained", config)
    cfg = replace(
        AYAKA_LARGE,
        readout="lm",
        pointer_dim=32,
        set_mixer_heads=2,
        set_mixer_layers=1,
        max_seq_len=512,
        serve_max_seq_len=512,
        lora_r=4,
    )
    report = inspect_direct_model(cfg)
    assert calls == [
        (
            str(tmp_path.resolve()),
            {
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
    ]
    assert report["output_tied_to_input"] is False
    assert report["native_output_shape"] == [512, 64]
    assert report["lora_targets"] == {
        "q_proj": 6,
        "k_proj": 4,
        "v_proj": 4,
        "o_proj": 6,
        "gate_proj": 6,
        "up_proj": 6,
        "down_proj": 6,
    }


def test_full_weight_count_plus_real_adapter_must_fit_limit():
    cfg = tiny_config(readout="lm")
    with pytest.raises(ValueError, match="exceeds 14B"):
        inspect_direct_model(cfg, official_weight_elements=14_000_000_000)
    with pytest.raises(ValueError, match="below native"):
        inspect_direct_model(cfg, official_weight_elements=1)


@pytest.mark.parametrize(
    "cfg",
    [
        tiny_config(readout="hybrid"),
        tiny_config(readout="lm", lora_r=0),
        tiny_config(readout="lm", max_seq_len=10_000_000),
        replace(AYAKA_LARGE, readout="lm", backbone_revision="main"),
    ],
)
def test_rejects_bad_model_requests_before_configuration_or_parameter_creation(monkeypatch, cfg):
    from transformers import AutoConfig, AutoModelForCausalLM

    monkeypatch.setattr(
        AutoConfig, "from_pretrained", lambda *a, **k: pytest.fail("configuration must not load")
    )
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_config",
        lambda *a, **k: pytest.fail("parameters must not create"),
    )
    with pytest.raises(ValueError):
        inspect_direct_model(cfg)


def test_missing_lora_targets_fail_on_meta_before_training():
    with pytest.raises(ValueError, match="Target modules"):
        inspect_direct_model(tiny_config(readout="lm", lora_targets=("not_a_projection",)))
