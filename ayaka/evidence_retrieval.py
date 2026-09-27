"""Deterministic query-based source retrieval; all spans remain verbatim."""

import collections
import hashlib
import json
import math
import re

from ayaka.evidence import public_input, source_text

STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "and",
    "or",
    "is",
    "are",
    "be",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "this",
    "that",
    "it",
    "as",
    "not",
    "no",
    "yes",
    "which",
    "what",
    "how",
    "would",
    "should",
    "can",
    "does",
    "do",
    "choose",
    "single",
    "best",
    "answer",
    "true",
    "false",
}


def words(text):
    return [
        w for w in re.findall(r"[a-z0-9]+", str(text).casefold()) if w not in STOP and len(w) > 1
    ]


def retrieve(record, max_chars=5800):
    inp = public_input(record)
    source = source_text(inp["state"])
    if not isinstance(inp["state"], str) or len(source) <= 5000:
        return inp, {"active": False, "chars_before": len(source), "chars_after": len(source)}
    # Split at existing paragraph/sentence boundaries; never paraphrase or invent a source.
    bounds = (
        [0]
        + [m.end() for m in re.finditer(r"\n\s*\n|(?<=[.!?;])\s+(?=[A-Z])", source)]
        + [len(source)]
    )
    spans = []
    for a, b in zip(bounds, bounds[1:], strict=False):
        while b - a > 480:
            end = source.rfind(" ", a + 220, a + 480)
            if end < 0:
                end = a + 480
            spans.append((a, end))
            a = end + 1
        if source[a:b].strip():
            spans.append((a, b))
    tokens = [words(source[a:b]) for a, b in spans]
    df = collections.Counter(w for row in tokens for w in set(row))
    avg = sum(map(len, tokens)) / len(tokens)
    query = collections.Counter(words(json.dumps(inp["question"], ensure_ascii=False)))
    scores = []
    for row in tokens:
        tf = collections.Counter(row)
        score = 0
        for w in query:
            n = tf[w]
            if n:
                idf = math.log(1 + (len(tokens) - df[w] + 0.5) / (df[w] + 0.5))
                score += idf * (n * 2.2) / (n + 1.2 * (0.25 + 0.75 * len(row) / max(avg, 1)))
        scores.append(score)
    selected = set()
    used = 0

    def add(i):
        nonlocal used
        if 0 <= i < len(spans) and i not in selected:
            a, b = spans[i]
            if used + b - a <= max_chars:
                selected.add(i)
                used += b - a

    # Case-file facts commonly appear at the end. Keep exact source tail, plus scored
    # evidence and adjacent clauses to preserve definitions and reference links.
    tail = [i for i, (a, b) in enumerate(spans) if a >= max(0, len(source) - 1100)]
    add(0)
    for i in tail:
        add(i)
    ranked = sorted(range(len(spans)), key=lambda i: (scores[i], -i), reverse=True)
    for i in ranked[:12]:
        add(i)
    for i in ranked[:5]:
        add(i - 1)
        add(i + 1)
    text = "\n\n[separate source span]\n\n".join(
        source[s:e].strip() for i in sorted(selected) for s, e in [spans[i]]
    )
    out = dict(inp, state=text)
    return out, {
        "active": True,
        "chars_before": len(source),
        "chars_after": len(text),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "selected_spans": [
            {"start": spans[i][0], "end": spans[i][1], "score": scores[i]} for i in sorted(selected)
        ],
        "notes": "Lexical retrieval with tail/adjacent spans; omitted facts and references can still matter.",
    }
