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
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]")
POLICY = {
    "version": 2,
    "word_ngram": 13,
    "min_exact_words": 4,
    "cjk_character_ngram": 32,
    "min_exact_cjk_characters": 12,
    "min_cjk_characters_per_ngram": 8,
    "scope": "state, instruction and candidate descriptions; complete lineage exclusion",
}


def _cjk_grams(text):
    normalized = "".join(_words(text))
    width = POLICY["cjk_character_ngram"]
    if len(normalized) < width or not _CJK.search(normalized):
        return set()
    return {
        normalized[i : i + width]
        for i in range(len(normalized) - width + 1)
        if len(_CJK.findall(normalized[i : i + width])) >= POLICY["min_cjk_characters_per_ngram"]
    }


def _words(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _record_texts(rec: dict) -> list[str]:
    q = rec.get("question", {})
    texts = _state_texts(rec.get("state"))
    texts.append(q.get("instructions") or q.get("instruction") or "")
    crit = q.get("criteria")
    if isinstance(crit, dict):
        texts.extend(str(v) for v in crit.values())
    elif isinstance(crit, list):
        texts.extend(str(v) for v in crit)
    return [t for t in texts if t]


def _state_texts(state):
    if isinstance(state, str):
        return [state]
    texts = [json.dumps(state, ensure_ascii=False)]
    values = state.values() if isinstance(state, dict) else state if isinstance(state, list) else []
    for value in values:
        texts.extend(_state_texts(value))
    return texts


def jevbench_public_dir() -> str:
    return str(resources.files("ayaka.eval") / "data" / "jevbench_public")


class Decontaminator:
    def __init__(self, texts: list[str], n: int = 13, min_exact_words: int = 4):
        self.n = n
        self.grams: set[tuple[str, ...]] = set()
        self.exact: set[str] = set()
        self.cjk_grams: set[str] = set()
        self.cjk_exact: set[str] = set()
        for t in texts:
            w = _words(t)
            if len(w) >= n:
                self.grams |= _ngrams(w, n)
            elif len(w) >= min_exact_words:
                self.exact.add(" ".join(w))
            normalized = "".join(w)
            self.cjk_grams.update(_cjk_grams(t))
            if (
                len(normalized) < POLICY["cjk_character_ngram"]
                and len(_CJK.findall(normalized)) >= POLICY["min_exact_cjk_characters"]
            ):
                self.cjk_exact.add(normalized)

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
        if (
            any(exact in "".join(w) for exact in self.cjk_exact)
            or _cjk_grams(text) & self.cjk_grams
        ):
            return True
        if len(w) < self.n:
            return False
        return any(tuple(w[i : i + self.n]) in self.grams for i in range(len(w) - self.n + 1))

    def sample_hit(self, s: Sample) -> bool:
        if any(self.text_hit(text) for text in _state_texts(s.state)):
            return True
        return any(
            self.text_hit(q.instruction) or any(self.text_hit(c.description) for c in q.candidates)
            for q in s.questions
        )

    def filter(self, samples: list[Sample]) -> tuple[list[Sample], int]:
        kept = [s for s in samples if not self.sample_hit(s)]
        return kept, len(samples) - len(kept)
