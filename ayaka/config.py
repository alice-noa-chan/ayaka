"""Electra model-family configuration (docs.md section 39).

Large / Base / Small share the same topology and output semantics — only
capacity changes. Nothing structural (Set Mixer, Pointer Head, question
isolation, dynamic candidates, evidence routing) is ever removed at
smaller sizes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ElectraConfig:
    """Topology + capacity for one Electra family member.

    Notation follows docs.md section 22: D = hidden, S = state length,
    Mq = question latents, Mc = candidate latents, Dp = pointer dim.
    """

    name: str
    hidden: int  # D
    state_layers: int  # shared evidence encoder depth
    heads: int  # attention heads (head_dim = hidden // heads)
    ffn: int  # SwiGLU intermediate dim
    text_encoder_layers: int  # shared question/candidate text encoder
    cross_attn_blocks: int  # candidate <-> state evidence blocks
    set_mixer_layers: int
    question_latents: int  # Mq
    candidate_latents: int  # Mc
    pointer_dim: int  # Dp
    block_topk: int  # long-context routed blocks per candidate

    vocab_size: int = 64_000
    max_state: int = 65_536
    max_candidates: int = 255
    block_size: int = 1024  # tokens per state block for long context
    short_context_threshold: int = 8192  # <= uses full attention path
    resid_dropout: float = 0.05
    attn_dropout: float = 0.0
    rope_theta: float = 10_000.0

    @property
    def head_dim(self) -> int:
        if self.hidden % self.heads != 0:
            raise ValueError(f"hidden {self.hidden} not divisible by heads {self.heads}")
        return self.hidden // self.heads

    @property
    def max_blocks(self) -> int:
        return self.max_state // self.block_size

    def estimate_params(self) -> int:
        """Rough parameter estimate for the documented targets."""
        d = self.hidden
        layer = 4 * d * d + 3 * d * self.ffn  # qkvo + swiglu
        state = self.state_layers * layer
        text = self.text_encoder_layers * layer
        # cross-attn block: self-attn on latents + cross-attn + ffn
        cross = self.cross_attn_blocks * (4 * d * d + 4 * d * d + 3 * d * self.ffn)
        mixer = self.set_mixer_layers * (4 * d * d + 3 * d * self.ffn)
        embed = self.vocab_size * d
        latents = (self.question_latents + self.candidate_latents) * d
        pointer = 2 * d * self.pointer_dim
        misc = 8 * d * d  # router, pools, small projections
        return embed + state + text + cross + mixer + latents + pointer + misc


ELECTRA_LARGE = ElectraConfig(
    name="electra-large",
    hidden=1536,
    state_layers=24,
    heads=24,
    ffn=6144,
    text_encoder_layers=4,
    cross_attn_blocks=4,
    set_mixer_layers=3,
    question_latents=8,
    candidate_latents=4,
    pointer_dim=512,
    block_topk=8,
)

ELECTRA_BASE = ElectraConfig(
    name="electra-base",
    hidden=1024,
    state_layers=18,
    heads=16,
    ffn=4096,
    text_encoder_layers=3,
    cross_attn_blocks=3,
    set_mixer_layers=2,
    question_latents=6,
    candidate_latents=4,
    pointer_dim=384,
    block_topk=6,
)

ELECTRA_SMALL = ElectraConfig(
    name="electra-small",
    hidden=768,
    state_layers=12,
    heads=12,
    ffn=3072,
    text_encoder_layers=2,
    cross_attn_blocks=2,
    set_mixer_layers=2,
    question_latents=4,
    candidate_latents=2,
    pointer_dim=256,
    block_topk=4,
)

MODEL_FAMILY: dict[str, ElectraConfig] = {
    c.name: c for c in (ELECTRA_LARGE, ELECTRA_BASE, ELECTRA_SMALL)
}


def tiny_config(**overrides) -> ElectraConfig:
    """Minimal config for CPU tests."""
    base = {
        "name": "electra-tiny",
        "hidden": 64,
        "state_layers": 2,
        "heads": 4,
        "ffn": 128,
        "text_encoder_layers": 1,
        "cross_attn_blocks": 1,
        "set_mixer_layers": 1,
        "question_latents": 2,
        "candidate_latents": 2,
        "pointer_dim": 32,
        "block_topk": 2,
        "vocab_size": 512,
        "max_state": 4096,
        "block_size": 64,
        "short_context_threshold": 256,
        "resid_dropout": 0.0,
    }
    base.update(overrides)
    return ElectraConfig(**base)
