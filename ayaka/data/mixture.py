"""Task-family quota + language temperature sampling (sec 41/43.1).

Sampling is hierarchical: pick a task family by quota first, then a
language inside the family by temperature, then a dataset row. Raw
row counts never decide batch composition directly.
"""

from __future__ import annotations

import random
from collections import defaultdict

# Family quotas. Jev-distilled decisions are the primary signal on a
# pretrained backbone (docs.md section 41 had 0.30 for a from-scratch
# encoder that also needed broad NLU data); the rest keeps multilingual
# coverage (ko/ja), high-cardinality choice and human soft labels.
TASK_FAMILY_QUOTA: dict[str, float] = {
    "direct_jev": 0.52,
    "choice": 0.10,
    "nli": 0.07,
    "noul": 0.07,
    "score": 0.06,
    "human_soft_label": 0.05,
    "hard_adversarial": 0.07,
    "long_context": 0.06,  # QuALITY articles (~2.8K tokens) vs jev-distill ~190
}

# Within the NLI family, source balance (sec 43.1)
NLI_SOURCE_QUOTA: dict[str, float] = {
    "snli_mnli": 0.35,
    "anli": 0.25,
    "klue_nli": 0.10,
    "kornli": 0.10,
    "jnli": 0.10,
    "chaosnli": 0.10,
}

# Language guardrails (sec 41)
LANGUAGE_GUARDRAILS: dict[str, tuple[float, float]] = {
    "en": (0.0, 0.60),
    "ko": (0.20, 1.0),
    "ja": (0.20, 1.0),
}


def _normalize(w: dict[str, float]) -> dict[str, float]:
    total = sum(w.values())
    if total <= 0:
        n = len(w)
        return dict.fromkeys(w, 1.0 / n)
    return {k: v / total for k, v in w.items()}


def language_weights(counts: dict[str, int], temperature: float = 0.5) -> dict[str, float]:
    """p(language) ∝ n^temperature with en/ko/ja guardrails applied."""
    w = _normalize({lang: c**temperature for lang, c in counts.items()})
    for _ in range(20):  # iterative clamp+renormalize converges fast
        changed = False
        for lang, (lo, hi) in LANGUAGE_GUARDRAILS.items():
            if lang in w and not (lo <= w[lang] <= hi):
                w[lang] = min(max(w[lang], lo), hi)
                changed = True
        w = _normalize(w)
        if not changed:
            break
    return w


class MixtureSampler:
    """Two-level sampler: task family quota, then language temperature."""

    def __init__(
        self,
        quota: dict[str, float] | None = None,
        temperature: float = 0.5,
        seed: int = 0,
    ):
        self.quota = dict(quota or TASK_FAMILY_QUOTA)
        self.temperature = temperature
        self.rng = random.Random(seed)

    def family_weights(self, families: list[str]) -> dict[str, float]:
        """Quota renormalized over families actually present."""
        present = set(families)
        return _normalize({f: w for f, w in self.quota.items() if f in present})

    def plan(
        self, pools: dict[tuple[str, str], list], n_samples: int
    ) -> dict[tuple[str, str], int]:
        """Allocate n_samples across (family, language) cells."""
        families = sorted({f for f, _ in pools})
        fw = self.family_weights(families)
        alloc: dict[tuple[str, str], int] = defaultdict(int)
        langs_by_family: dict[str, list[str]] = defaultdict(list)
        for f, lang in pools:
            if pools[(f, lang)]:
                langs_by_family[f].append(lang)
        for f, fweight in fw.items():
            counts = {lang: len(pools[(f, lang)]) for lang in langs_by_family[f]}
            lw = language_weights(counts, self.temperature)
            family_n = round(fweight * n_samples)
            for lang, w in lw.items():
                alloc[(f, lang)] = round(family_n * w)
        return dict(alloc)

    def sample(self, pools: dict[tuple[str, str], list], n_samples: int) -> list:
        """Draw n_samples rows following the mixture plan."""
        alloc = self.plan(pools, n_samples)
        out = []
        for (f, lang), n in alloc.items():
            pool = pools.get((f, lang), [])
            if not pool:
                continue
            out.extend(self.rng.choice(pool) for _ in range(n))
        self.rng.shuffle(out)
        return out
