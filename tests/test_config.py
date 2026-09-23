from ayaka.config import (
    ELECTRA_BASE,
    ELECTRA_LARGE,
    ELECTRA_SMALL,
    MODEL_FAMILY,
    tiny_config,
)


def test_family_shapes_match_spec():
    # docs.md section 39 table
    assert ELECTRA_LARGE.hidden == 1536 and ELECTRA_LARGE.state_layers == 24
    assert ELECTRA_BASE.hidden == 1024 and ELECTRA_BASE.state_layers == 18
    assert ELECTRA_SMALL.hidden == 768 and ELECTRA_SMALL.state_layers == 12
    for cfg in (ELECTRA_LARGE, ELECTRA_BASE, ELECTRA_SMALL):
        assert cfg.head_dim == 64
        assert cfg.max_state == 65_536
        assert cfg.max_candidates == 255
        assert cfg.vocab_size == 64_000


def test_param_estimates_in_documented_ballpark():
    # spec targets: ~1.4B / ~500M / ~220M (order of magnitude check)
    assert 1.0e9 < ELECTRA_LARGE.estimate_params() < 2.0e9
    assert 3.5e8 < ELECTRA_BASE.estimate_params() < 7.5e8
    assert 1.4e8 < ELECTRA_SMALL.estimate_params() < 3.2e8


def test_topology_preserved_across_sizes():
    # Small keeps Set Mixer / pointer head / router (docs.md section 28.1)
    for cfg in (ELECTRA_LARGE, ELECTRA_BASE, ELECTRA_SMALL):
        assert cfg.set_mixer_layers >= 1
        assert cfg.cross_attn_blocks >= 1
        assert cfg.pointer_dim > 0
        assert cfg.block_topk >= 1
        assert cfg.question_latents >= 1 and cfg.candidate_latents >= 1


def test_family_registry_and_tiny():
    assert set(MODEL_FAMILY) == {"electra-large", "electra-base", "electra-small"}
    cfg = tiny_config()
    assert cfg.head_dim == 16
    assert cfg.max_blocks == cfg.max_state // cfg.block_size
