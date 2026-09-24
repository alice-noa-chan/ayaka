"""Evaluation decontamination (docs.md section 32, "Evaluation contamination").

Training samples that share a long word n-gram with any JevBench public
item (state, instruction, or option text) are dropped before training.
Short texts that cannot form an n-gram are compared by exact
normalized equality instead.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from importlib import resources

from .schema import Sample

_WORD = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _record_texts(rec: dict) -> list[str]:
    q = rec.get("question", {})
    texts = [
        rec.get("state")
        if isinstance(rec.get("state"), str)
        else json.dumps(rec.get("state"), ensure_ascii=False)
    ]
    texts.append(q.get("instructions") or q.get("instruction") or "")
    crit = q.get("criteria")
    if isinstance(crit, dict):
        texts.extend(str(v) for v in crit.values())
    elif isinstance(crit, list):
        texts.extend(str(v) for v in crit)
    return [t for t in texts if t]


def jevbench_public_dir() -> str:
    return str(resources.files("ayaka.eval") / "data" / "jevbench_public")


class Decontaminator:
    def __init__(self, texts: list[str], n: int = 13, min_exact_words: int = 4):
        self.n = n
        self.grams: set[tuple[str, ...]] = set()
        self.exact: set[str] = set()
        for t in texts:
            w = _words(t)
            if len(w) >= n:
                self.grams |= _ngrams(w, n)
            elif len(w) >= min_exact_words:
                self.exact.add(" ".join(w))

    @classmethod
    def from_jevbench(cls, data_dir: str | None = None, n: int = 13) -> Decontaminator:
        data_dir = data_dir or jevbench_public_dir()
        texts: list[str] = []
        for name in sorted(os.listdir(data_dir)):
            if name.endswith(".jsonl"):
                with open(os.path.join(data_dir, name), encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            texts.extend(_record_texts(json.loads(line)))
        return cls(texts, n=n)

    def text_hit(self, text: str) -> bool:
        w = _words(text)
        if " ".join(w) in self.exact:
            return True
        if len(w) < self.n:
            return False
        return any(tuple(w[i : i + self.n]) in self.grams for i in range(len(w) - self.n + 1))

    def sample_hit(self, s: Sample) -> bool:
        state = s.state if isinstance(s.state, str) else json.dumps(s.state, ensure_ascii=False)
        if self.text_hit(state):
            return True
        return any(self.text_hit(q.instruction) for q in s.questions)

    def filter(self, samples: list[Sample]) -> tuple[list[Sample], int]:
        kept = [s for s in samples if not self.sample_hit(s)]
        return kept, len(samples) - len(kept)
