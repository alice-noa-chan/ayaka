"""Electra model family on pretrained Gemma 4 backbones.

The family keeps Electra's decision contract — shared state encoded
once, isolated question branches, dynamic candidate sets, pointer
readout, per-primitive calibration — but replaces the from-scratch
encoder with an instruction-tuned Gemma 4 text stack. Only capacity
changes across sizes; the decision topology is identical.

| size  | backbone              | effective params | train GPU  |
| ----- | --------------------- | ---------------- | ---------- |
| small | google/gemma-4-E2B-it | ~2B              | A10G 24GB  |
| base  | google/gemma-4-E4B-it | ~4B              | L40S 48GB  |
| large | google/gemma-4-12B-it | ~12B             | H100 80GB  |
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ElectraConfig:
    """Backbone choice + decision-head capacity for one family member."""

    name: str
    backbone: str  # HF repo id, or "tiny" for a random CPU test model
    pointer_dim: int  # Dp — pointer/set-mixer width
    set_mixer_layers: int
    set_mixer_heads: int
    lora_r: int
    lora_alpha: int
    max_seq_len: int = 4096  # prompt tokens per question (state + question + options)
    max_label_candidates: int = 26  # A..Z single-token readout; larger sets use the pointer
    long_prompt_tokens: int = 1024  # long bucket for pointer mixing and temperatures
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    tiny_overrides: dict = field(default_factory=dict)  # only for backbone == "tiny"


ELECTRA_SMALL = ElectraConfig(
    name="electra-small",
    backbone="google/gemma-4-E2B-it",
    pointer_dim=256,
    set_mixer_layers=2,
    set_mixer_heads=4,
    lora_r=32,
    lora_alpha=64,
)

ELECTRA_BASE = ElectraConfig(
    name="electra-base",
    backbone="google/gemma-4-E4B-it",
    pointer_dim=384,
    set_mixer_layers=2,
    set_mixer_heads=6,
    lora_r=32,
    lora_alpha=64,
)

ELECTRA_LARGE = ElectraConfig(
    name="electra-large",
    backbone="google/gemma-4-12B-it",
    pointer_dim=512,
    set_mixer_layers=3,
    set_mixer_heads=8,
    lora_r=64,
    lora_alpha=128,
)

MODEL_FAMILY: dict[str, ElectraConfig] = {
    c.name: c for c in (ELECTRA_LARGE, ELECTRA_BASE, ELECTRA_SMALL)
}


def tiny_config(**overrides) -> ElectraConfig:
    """Random-weight Gemma 4 text stack for CPU tests (no downloads)."""
    base = {
        "name": "electra-tiny",
        "backbone": "tiny",
        "pointer_dim": 32,
        "set_mixer_layers": 1,
        "set_mixer_heads": 2,
        "lora_r": 4,
        "lora_alpha": 8,
        "max_seq_len": 512,
    }
    base.update(overrides)
    return ElectraConfig(**base)


def model_config(name: str) -> ElectraConfig:
    return tiny_config() if name in ("tiny", "electra-tiny") else MODEL_FAMILY[name]
