from ayaka.config import (
    AYAKA_BASE,
    AYAKA_LARGE,
    AYAKA_SMALL,
    MODEL_FAMILY,
    model_config,
    tiny_config,
)


def test_family_backbones_are_gemma4_instruct():
    assert AYAKA_SMALL.backbone == "google/gemma-4-E2B-it"
    assert AYAKA_BASE.backbone == "google/gemma-4-E4B-it"
    assert AYAKA_LARGE.backbone == "google/gemma-4-12B-it"
    assert set(MODEL_FAMILY) == {"electra-small", "electra-base", "electra-large"}


def test_capacity_grows_with_size_topology_fixed():
    s, b, lg = AYAKA_SMALL, AYAKA_BASE, AYAKA_LARGE
    assert s.pointer_dim < b.pointer_dim < lg.pointer_dim
    assert s.lora_r <= b.lora_r <= lg.lora_r
    for c in (s, b, lg):
        assert c.set_mixer_layers >= 2  # the Set Mixer is never dropped
        assert c.pointer_dim % c.set_mixer_heads == 0
        assert c.max_label_candidates == 26


def test_model_config_lookup():
    assert model_config("tiny").backbone == "tiny"
    assert model_config("electra-base") is AYAKA_BASE
    assert tiny_config(pointer_dim=16).pointer_dim == 16
