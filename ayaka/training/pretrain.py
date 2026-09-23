"""Foundation pretraining objectives (docs.md section 40.1, A10).

    L_pretrain = 1.00 L_RTD + 0.30 L_structured + 0.20 L_span
               + 0.15 L_multilingual + 0.15 L_evidence_retrieval

v1 generator policy (A10): tokens are corrupted by a unigram/uniform
sampler — a learned MLM generator is a follow-up. The discriminator
objectives (the actual Electra contribution) are fully implemented.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..model.model import ElectraDecisionModel
from ..special_tokens import NUM_SPECIAL_TOKENS, SPECIAL_TOKEN_IDS


def rtd_corrupt(
    token_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    vocab_lo: int,
    vocab_hi: int,
    rate: float = 0.15,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace ~rate non-special tokens with random vocab ids.

    Returns (corrupted_ids, labels) where labels[i]=1 marks replaced
    positions. Special tokens are never corrupted.
    """
    corrupted = token_ids.clone()
    special = token_ids < NUM_SPECIAL_TOKENS
    mask = (
        torch.rand(token_ids.shape, generator=generator, device=token_ids.device) < rate
    ) & ~special
    random_ids = torch.randint(
        vocab_lo, vocab_hi, token_ids.shape, generator=generator, device=token_ids.device
    )
    corrupted[mask] = random_ids[mask]
    return corrupted, mask.float()


class RTDHead(nn.Module):
    """Per-token replaced/not-replaced binary discriminator."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        return self.proj(hs).squeeze(-1)


def rtd_loss(
    model: ElectraDecisionModel,
    rtd_head: RTDHead,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    labels: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """BCE on per-token replacement predictions."""
    mem = model.state_encoder(corrupted_ids, cu_seqlens)
    logits = rtd_head(mem.hs)
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)


# ------------------------------------------------- structured corruption

_TYPE_MARKERS = [
    SPECIAL_TOKEN_IDS["<num>"],
    SPECIAL_TOKEN_IDS["<str>"],
    SPECIAL_TOKEN_IDS["<bool_true>"],
    SPECIAL_TOKEN_IDS["<bool_false>"],
    SPECIAL_TOKEN_IDS["<null>"],
]


def structured_corrupt(
    token_ids: torch.Tensor,
    rate: float = 0.10,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Swap primitive-type markers (<num>/<str>/<bool_*>) so that
    e.g. 42 becomes "42" at the typed-serialization level."""
    corrupted = token_ids.clone()
    labels = torch.zeros_like(token_ids, dtype=torch.float)
    is_marker = torch.isin(token_ids, torch.tensor(_TYPE_MARKERS, device=token_ids.device))
    pick = (
        torch.rand(token_ids.shape, generator=generator, device=token_ids.device) < rate
    ) & is_marker
    alt = torch.tensor(_TYPE_MARKERS, device=token_ids.device)[
        torch.randint(
            0, len(_TYPE_MARKERS), token_ids.shape, generator=generator, device=token_ids.device
        )
    ]
    # keep original where alt == current (avoid fake labels)
    same = alt == token_ids
    pick = pick & ~same
    corrupted[pick] = alt[pick]
    labels[pick] = 1.0
    return corrupted, labels


def structured_corrupt_loss(
    model: ElectraDecisionModel,
    head: RTDHead,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    labels: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    mem = model.state_encoder(corrupted_ids, cu_seqlens)
    logits = head(mem.hs)
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)


# -------------------------------------------------------- span relation


class SpanRelationHead(nn.Module):
    """3-way relation classifier over pooled segment pairs."""

    def __init__(self, dim: int, n_classes: int = 3):
        super().__init__()
        self.proj = nn.Linear(2 * dim, n_classes)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([a, b], dim=-1))


def span_relation_loss(
    head: SpanRelationHead,
    span_a: torch.Tensor,
    span_b: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """labels: [n_pairs] in {0=entail, 1=contradict, 2=irrelevant}."""
    return nn.functional.cross_entropy(head(span_a, span_b), labels)


# -------------------------------------------------- evidence retrieval


def evidence_retrieval_loss(
    model: ElectraDecisionModel,
    item_queries: torch.Tensor,
    mem,
    item_state_index: torch.Tensor,
    block_target: torch.Tensor,
    block_mask: torch.Tensor,
) -> torch.Tensor:
    """Foundation-stage router pretraining on synthetic evidence
    labels — same BlockRouter the decision stage uses."""
    from ..losses import router_bce

    route = model.router(item_queries, mem.bs, item_state_index, mem.block_state_index)
    return router_bce(route["route_probs"], block_target, block_mask)


# ------------------------------------------------- multilingual align


def multilingual_alignment_loss(
    latents_a: torch.Tensor, latents_b: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """InfoNCE pulling parallel (en/ko/ja) question latents together."""
    a = nn.functional.normalize(latents_a, dim=-1)
    b = nn.functional.normalize(latents_b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (
        nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.T, labels)
    )
