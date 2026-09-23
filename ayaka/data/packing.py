"""Token-budget packing + batch shape contract (sec 45.3-45.4, 46).

Batches are built by token budget, not sample count. Each batch gets
a fixed-topology BatchDescriptor so tensor payloads stay ragged while
control flow stays static — torch.compile sees a stable graph per
bucket.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..collate import encode_candidate, encode_question, encode_state
from ..tokenizer import Tokenizer
from .schema import Sample

STATE_BUCKETS = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
CAND_COUNT_BUCKETS = ((2, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 255))
CAND_LEN_BUCKETS = (16, 32, 64, 128)


def _bucket(value: int, buckets: tuple[int, ...]) -> int:
    for b in buckets:
        if value <= b:
            return b
    return buckets[-1]


def _count_bucket(k: int) -> tuple[int, int]:
    for lo, hi in CAND_COUNT_BUCKETS:
        if lo <= k <= hi:
            return (lo, hi)
    return CAND_COUNT_BUCKETS[-1]


@dataclass(frozen=True)
class BatchDescriptor:
    """Fixed execution-path descriptor for a packed batch (sec 46)."""

    state_bucket: int
    cand_count_bucket: tuple[int, int]
    cand_len_bucket: int
    primitive_mix: tuple[int, int, int]  # (noul, choice, score) counts
    long_context_mode: bool


def sample_token_stats(sample: Sample, tokenizer: Tokenizer) -> dict:
    """Token counts for packing decisions (collation-side work)."""
    state_len = len(encode_state(sample.state, tokenizer))
    n_cand, max_cand_len = 0, 0
    q_tokens = 0
    primitive_mix = [0, 0, 0]
    prim_idx = {"noul": 0, "choice": 1, "score": 2}
    for q in sample.questions:
        q_tokens += len(encode_question(q.instruction, tokenizer))
        primitive_mix[prim_idx[q.type]] += 1
        for c in q.candidates:
            n_cand += 1
            max_cand_len = max(max_cand_len, len(encode_candidate(c.description, tokenizer)))
    return {
        "state_len": state_len,
        "q_tokens": q_tokens,
        "n_candidates": n_cand,
        "max_cand_len": max_cand_len,
        "primitive_mix": tuple(primitive_mix),
        "total": state_len + q_tokens + n_cand * max_cand_len,
    }


def descriptor_for(
    samples: list[Sample], stats: list[dict], long_threshold: int = 8192
) -> BatchDescriptor:
    max_state = max(s["state_len"] for s in stats)
    max_cand = max(s["n_candidates"] for s in stats)
    max_clen = max(s["max_cand_len"] for s in stats)
    mix = [0, 0, 0]
    for s in stats:
        for i in range(3):
            mix[i] += s["primitive_mix"][i]
    return BatchDescriptor(
        state_bucket=_bucket(max_state, STATE_BUCKETS),
        cand_count_bucket=_count_bucket(max_cand),
        cand_len_bucket=_bucket(max_clen, CAND_LEN_BUCKETS),
        primitive_mix=tuple(mix),
        long_context_mode=max_state > long_threshold,
    )


def pack_by_token_budget(
    samples: list[Sample],
    tokenizer: Tokenizer,
    token_budget: int,
    max_samples: int | None = None,
) -> list[tuple[list[Sample], BatchDescriptor]]:
    """Greedy first-fit-decreasing bin-pack into token budgets."""
    stats = [sample_token_stats(s, tokenizer) for s in samples]
    order = sorted(range(len(samples)), key=lambda i: -stats[i]["total"])
    batches: list[list[int]] = []
    loads: list[int] = []
    for i in order:
        t = stats[i]["total"]
        placed = False
        for b, load in enumerate(loads):
            if load + t <= token_budget and (max_samples is None or len(batches[b]) < max_samples):
                batches[b].append(i)
                loads[b] += t
                placed = True
                break
        if not placed:
            batches.append([i])
            loads.append(t)
    out = []
    for batch in batches:
        ss = [samples[i] for i in batch]
        st = [stats[i] for i in batch]
        out.append((ss, descriptor_for(ss, st)))
    return out
