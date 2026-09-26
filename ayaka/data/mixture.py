"""Task-family quota + language temperature sampling (sec 41/43.1).

Sampling is hierarchical: pick a task family by quota first, then a
language inside the family by temperature, then a dataset row. Raw
row counts never decide batch composition directly.
"""

from __future__ import annotations

import random
from dataclasses import replace

# Family quotas. Typed decisions (Open-Jev streams) lead; the rest cover
# JevBench's skill families with license-clean human/procedural labels:
# judge (answer adequacy, preference, safety), reasoning (multi-hop,
# numeric, commonsense), fact_check (claim vs evidence incl. "not enough
# info"), long documents, and multilingual NLU. Quotas renormalize over the
# families actually present.
TASK_FAMILY_QUOTA: dict[str, float] = {
    "direct_jev": 0.27,
    "policy": 0.08,  # LegalBench + generated long policies (JevBench long_policy)
    "temporal_numeric": 0.07,  # generated dates/amounts (JevBench temporal_numeric)
    "judge": 0.10,
    "reasoning": 0.09,
    "fact_check": 0.06,
    "choice": 0.05,
    "nli": 0.05,
    "noul": 0.07,
    "score": 0.04,
    "human_soft_label": 0.04,
    "hard_adversarial": 0.03,
    "long_context": 0.05,
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
    if not w:
        return {}
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
        return self._plan({cell: len(rows) for cell, rows in pools.items() if rows}, n_samples)

    def _allocate(self, weights: dict, n: int) -> dict:
        """Largest remainders, with seeded tie breaking and no lost budget."""
        if n < 0:
            raise ValueError("sample budget must be non-negative")
        raw = {key: weight * n for key, weight in weights.items()}
        out = {key: int(value) for key, value in raw.items()}
        order = sorted(raw, key=lambda key: (raw[key] - out[key], self.rng.random()), reverse=True)
        for key in order[: n - sum(out.values())]:
            out[key] += 1
        return out

    def _plan(self, counts: dict[tuple[str, str], int], n: int) -> dict[tuple[str, str], int]:
        families = sorted({f for f, _ in counts})
        fw = self.family_weights(families)
        alloc = {}
        for f, family_n in self._allocate(fw, n).items():
            langs = {lang: count for (fam, lang), count in counts.items() if fam == f}
            lw = language_weights(langs, self.temperature)
            alloc.update({(f, lang): k for lang, k in self._allocate(lw, family_n).items()})
        return alloc

    def question_batches(self, pools: dict, n_questions: int):
        """Exact question quotas while retaining multi-question prefix sharing.

        Prepare the static pools once. Select whole states when they fit a cell's
        budget and a random subset of questions otherwise; a 28-label sample
        cannot turn a 4% question quota into most of an optimizer step.
        """
        if n_questions <= 0:
            raise ValueError("question budget must be positive")
        cells = {key: [s for s in rows if s.questions] for key, rows in pools.items()}
        cells = {key: rows for key, rows in cells.items() if rows}
        counts = {key: sum(len(s.questions) for s in rows) for key, rows in cells.items()}
        while True:
            alloc = self._plan(counts, n_questions)
            if not alloc:
                raise RuntimeError("no question pools match the mixture quotas")
            drawn = []
            for cell, remaining in alloc.items():
                while remaining:
                    sample = self.rng.choice(cells[cell])
                    take = min(remaining, len(sample.questions))
                    if take < len(sample.questions):
                        indices = sorted(self.rng.sample(range(len(sample.questions)), take))
                        sample = replace(sample, questions=[sample.questions[i] for i in indices])
                    drawn.append(sample)
                    remaining -= take
            self.rng.shuffle(drawn)
            yield drawn

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
