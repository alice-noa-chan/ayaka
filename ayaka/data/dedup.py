"""Dedup and lineage splitting (docs.md section 32, 43.5).

Three layers:

1. Cross-dataset dedup — canonical key over (state, question,
   unordered candidates) merges e.g. jev-distill-corpus-v3 with
   Open-Jev duplicates.
2. Benchmark-family lineage — metadata fields link translations /
   derived rows to their source example so family splits hold.
3. Synthetic-template leakage — group keys (domain, template,
   scenario, generator seed) split before row-level splits.
"""

from __future__ import annotations

import hashlib

from ..serialization import dedup_key
from .schema import Sample


def sample_dedup_key(sample: Sample, question_index: int) -> str:
    q = sample.questions[question_index]
    return dedup_key(
        sample.state,
        q.instruction,
        [c.description for c in q.candidates],
    )


def dedup_samples(samples: list[Sample]) -> list[Sample]:
    """Drop duplicate (state, question, candidate-set) questions.

    If a sample loses all questions it is dropped entirely.
    First occurrence wins; metadata of dropped rows is not merged
    (lineage stays auditable via manifests).
    """
    seen: set[str] = set()
    out: list[Sample] = []
    for s in samples:
        keep_q = []
        for i, q in enumerate(s.questions):
            key = sample_dedup_key(s, i)
            if key not in seen:
                seen.add(key)
                keep_q.append(q)
        if keep_q:
            s.questions = keep_q
            out.append(s)
    return out


def lineage_key(sample: Sample) -> str | None:
    """Benchmark-family lineage key when the manifest provides one."""
    md = sample.metadata
    family = md.get("source_family")
    ex = md.get("source_example_id")
    if family is None or ex is None:
        return None
    root = md.get("translation_of") or md.get("derived_from") or ex
    return f"{family}:{root}"


def group_key(sample: Sample) -> str:
    """Synthetic-template group key (sec 32.3) — falls back to lineage
    or the sample's own dedup key."""
    md = sample.metadata
    parts = [
        md.get("domain", ""),
        md.get("generator_template_id") or md.get("template", ""),
        md.get("scenario_family", ""),
        md.get("generator_seed_family", ""),
    ]
    if any(parts):
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
    return lineage_key(sample) or sample_dedup_key(sample, 0)


def group_split(
    samples: list[Sample], val_frac: float = 0.1, seed: int = 0
) -> tuple[list[Sample], list[Sample]]:
    """Deterministic split by group key — no leakage across splits."""
    train, val = [], []
    for s in samples:
        h = hashlib.sha256(f"{seed}:{group_key(s)}".encode()).digest()
        frac = int.from_bytes(h[:4], "little") / 0xFFFFFFFF
        (val if frac < val_frac else train).append(s)
    return train, val
