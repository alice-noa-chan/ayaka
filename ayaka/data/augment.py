"""Electra-specific counterfactual augmentation (sec 34/44, A5).

Rule-based target semantics accompany every transformation — the
distribution moves with the candidates, and insufficient-evidence
variants never keep a forced one-hot target.
"""

from __future__ import annotations

import copy
import random
import re

from .schema import Candidate, Question, Sample, uniform


def _renorm(dist: dict[str, float]) -> dict[str, float]:
    total = sum(dist.values())
    if total <= 0:
        return {k: 1.0 / len(dist) for k in dist} if dist else {}
    return {k: v / total for k, v in dist.items()}


# ----------------------------------------------------------- candidates


def permute_candidates(q: Question, rng: random.Random) -> Question:
    """Option order is non-semantic; the distribution follows."""
    order = list(range(len(q.candidates)))
    rng.shuffle(order)
    cands = [q.candidates[i] for i in order]
    return Question(
        id=q.id,
        type=q.type,
        instruction=q.instruction,
        candidates=cands,
        target_distribution={c.id: q.target_distribution.get(c.id, 0.0) for c in cands},
    )


def add_distractor(
    q: Question, description: str, rng: random.Random, mass: float = 0.0
) -> Question:
    """Insert a new candidate. Default mass=0 (distractor); nonzero
    mass is drawn proportionally from existing candidates."""
    cid = f"aug_{rng.randrange(1 << 30):x}"
    cand = Candidate(cid, description)
    cands = q.candidates + [cand]
    dist = dict(q.target_distribution)
    if mass > 0:
        dist = {k: v * (1 - mass) for k, v in dist.items()}
    dist[cid] = mass
    return Question(q.id, q.type, q.instruction, cands, _renorm(dist))


def remove_candidate(q: Question, cand_id: str) -> Question | None:
    """Remove a candidate; its mass moves to a NOTA candidate if one
    exists, else the remainder renormalizes (A5)."""
    cands = [c for c in q.candidates if c.id != cand_id]
    if len(cands) < 2:
        return None
    removed = q.target_distribution.get(cand_id, 0.0)
    dist = {c.id: q.target_distribution.get(c.id, 0.0) for c in cands}
    nota = next((c for c in cands if c.is_nota), None)
    if nota is not None:
        dist[nota.id] = dist.get(nota.id, 0.0) + removed
    return Question(q.id, q.type, q.instruction, cands, _renorm(dist))


def add_nota(q: Question, description: str = "none of the above") -> Question:
    """Append explicit none-of-the-above. If a valid answer exists in
    the set, NOTA target is 0 (A5)."""
    nota = Candidate("__nota__", description, is_nota=True)
    cands = q.candidates + [nota]
    dist = dict(q.target_distribution)
    dist["__nota__"] = 0.0
    return Question(q.id, q.type, q.instruction, cands, _renorm(dist))


def paraphrase_candidate(q: Question, cand_id: str, new_description: str) -> Question:
    """Same candidate id, new surface form — target unchanged."""
    cands = [
        Candidate(
            c.id,
            new_description if c.id == cand_id else c.description,
            ordinal=c.ordinal,
            is_nota=c.is_nota,
        )
        for c in q.candidates
    ]
    return Question(q.id, q.type, q.instruction, cands, dict(q.target_distribution))


# ------------------------------------------------------------- evidence

_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


def delete_evidence_text(state: str, rng: random.Random, min_keep: int = 1) -> str:
    """Drop a random subset of sentences (keeps at least min_keep)."""
    sents = _SENT_SPLIT.split(state.strip())
    sents = [s for s in sents if s]
    if len(sents) <= min_keep:
        return " ".join(sents[:1])
    keep = max(min_keep, rng.randrange(min_keep, len(sents)))
    idx = sorted(rng.sample(range(len(sents)), keep))
    return " ".join(sents[i] for i in idx)


def delete_evidence_fields(state: dict, rng: random.Random) -> dict:
    """Drop a random proper subset of top-level fields."""
    keys = list(state.keys())
    if len(keys) <= 1:
        return dict(state)
    drop = rng.sample(keys, rng.randrange(1, len(keys)))
    return {k: v for k, v in state.items() if k not in drop}


def insert_irrelevant(
    state: str, filler: str, rng: random.Random, position: str | None = None
) -> str:
    """Insert irrelevant context at head/middle/tail (sec 44)."""
    pos = position or rng.choice(["head", "middle", "tail"])
    if pos == "head":
        return f"{filler} {state}"
    if pos == "tail":
        return f"{state} {filler}"
    sents = _SENT_SPLIT.split(state.strip())
    mid = len(sents) // 2
    sents.insert(mid, filler)
    return " ".join(sents)


# --------------------------------------------------------------- sample


def evidence_deletion_variant(
    sample: Sample,
    rng: random.Random,
    target: str = "uniform",
) -> Sample:
    """Produce an insufficient-evidence counterfactual.

    The new sample is flagged evidence_state=deleted and retargeted —
    uniform by default, or the caller may relabel with the teacher and
    keep the gold lineage (sec 44/A5).
    """
    s = copy.deepcopy(sample)
    if isinstance(s.state, dict):
        s.state = delete_evidence_fields(s.state, rng)
    elif isinstance(s.state, str):
        s.state = delete_evidence_text(s.state, rng)
    for q in s.questions:
        if target == "uniform":
            q.target_distribution = uniform(q.candidates)
    s.metadata = dict(s.metadata)
    s.metadata["evidence_state"] = "deleted"
    s.metadata["derived_from"] = s.metadata.get("source_example_id", s.metadata.get("id"))
    return s


def group_shared_state(samples: list[Sample]) -> list[Sample]:
    """Merge questions of samples that share the same state object
    into one shared-state sample (multi-question grouping, sec 44)."""
    from ..serialization import canonical_state

    groups: dict[str, Sample] = {}
    order: list[str] = []
    for s in samples:
        key = canonical_state(s.state)
        if key not in groups:
            merged = copy.deepcopy(s)
            merged.questions = []
            groups[key] = merged
            order.append(key)
        groups[key].questions.extend(copy.deepcopy(s.questions))
    return [groups[k] for k in order]
