"""Ayaka model family on pretrained Gemma 4 backbones.

The family keeps the decision contract of the original ELECTRA-based
design — shared state encoded once, isolated question branches, dynamic
candidate sets, pointer readout, per-primitive calibration — but replaces
the from-scratch encoder with an instruction-tuned Gemma 4 text stack.
Only capacity changes across sizes; the decision topology is identical.

The ``electra-*`` size names are kept as stable identifiers: published
checkpoints store them in their config, and the CLIs accept them.

| size  | backbone              | effective params | train GPU  |
| ----- | --------------------- | ---------------- | ---------- |
| small | google/gemma-4-E2B-it | ~2B              | A10G 24GB  |
| base  | google/gemma-4-E4B-it | ~4B              | L40S 48GB  |
| large | google/gemma-4-12B-it | ~12B             | H100 80GB  |
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AyakaConfig:
    """Backbone choice + decision-head capacity for one family member."""

    name: str
    backbone: str  # HF repo id, or "tiny" for a random CPU test model
    pointer_dim: int  # Dp — pointer/set-mixer width
    set_mixer_layers: int
    set_mixer_heads: int
    lora_r: int
    lora_alpha: int
    max_seq_len: int = 4096  # training prompt tokens per question (state + question + options)
    # Inference prompt budget (Decision / serve / eval). Longer states are
    # otherwise cut in the middle; JevBench hard documents reach ~6K tokens and
    # every Gemma 4 backbone supports far longer contexts than training uses.
    serve_max_seq_len: int = 8192
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
    # exact Hub revision of ``backbone`` (None = latest). Pinned so a later
    # upstream update cannot silently change a released model.
    backbone_revision: str | None = None
    # Old checkpoints remain v1/direct. v2 configs opt into auto + medium.
    version: int = 1
    reasoning_defaults: dict = field(default_factory=dict)
    readout: str = "hybrid"  # lm | pointer | hybrid; set_mixer_layers=0 is simple pointer
    input_contract_required: bool = (
        False  # saved recipe-bound checkpoints fail closed if meta is lost
    )

    def __post_init__(self):
        if self.readout not in ("lm", "pointer", "hybrid"):
            raise ValueError("readout must be lm, pointer, or hybrid")
        if type(self.input_contract_required) is not bool:
            raise ValueError("input_contract_required must be boolean")


AYAKA_SMALL = AyakaConfig(
    name="electra-small",
    backbone="google/gemma-4-E2B-it",
    backbone_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    pointer_dim=256,
    set_mixer_layers=2,
    set_mixer_heads=4,
    lora_r=32,
    lora_alpha=64,
)

AYAKA_BASE = AyakaConfig(
    name="electra-base",
    backbone="google/gemma-4-E4B-it",
    backbone_revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
    pointer_dim=384,
    set_mixer_layers=2,
    set_mixer_heads=6,
    lora_r=32,
    lora_alpha=64,
)

AYAKA_LARGE = AyakaConfig(
    name="electra-large",
    backbone="google/gemma-4-12B-it",
    backbone_revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    pointer_dim=512,
    set_mixer_layers=3,
    set_mixer_heads=8,
    lora_r=64,
    lora_alpha=128,
)

MODEL_FAMILY: dict[str, AyakaConfig] = {c.name: c for c in (AYAKA_LARGE, AYAKA_BASE, AYAKA_SMALL)}


def tiny_config(**overrides) -> AyakaConfig:
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
        "serve_max_seq_len": 512,
    }
    base.update(overrides)
    return AyakaConfig(**base)


def model_config(name: str) -> AyakaConfig:
    return tiny_config() if name in ("tiny", "electra-tiny") else MODEL_FAMILY[name]
