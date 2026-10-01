"""Decision prompt rendering for the Gemma 4 chat format.

A decision is split into a shared **prefix** (instructions + state) that
is encoded once per state, and one **suffix** per question (question,
options, answer cue). Suffixes never see each other: at inference they
branch off the prefix KV cache, in training they are independent rows.

Candidate permutation equivariance is structural: choice options are
displayed in a canonical order derived from their content, so any input
permutation renders the identical prompt and the output distribution is
permuted back exactly. Score levels keep their ordinal order (the order
is the semantics there); noul is a fixed Yes/No pair.

Readout labels are single tokens: ``A``..``Z`` after ``The answer is (``
for choice, ordinal digits for score, `` Yes``/`` No`` for noul. Sets
larger than the label alphabet are pointer-only (no label readout).
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass

from .tokenization import Tokenizer

LETTERS = [chr(ord("A") + i) for i in range(26)]

SYSTEM = (
    "You are a decision model. Read the state, then answer the question using only "
    "the state and the answer criteria. If the state does not establish something, "
    "do not assume it."
)

USER_OPEN = "<|turn>user\n"
MODEL_OPEN = "<turn|>\n<|turn>model\n"
CHOICE_CUE = "The answer is ("
NOUL_CUE = "Answer:"
TRIVIAL_NOUL = {"", "true", "false", "yes", "no"}


@dataclass
class QuestionView:
    """What rendering needs from a question (schema- and API-agnostic).

    noul convention: descriptions == [false_description, true_description].
    """

    type: str  # noul | choice | score
    instruction: str
    descriptions: list[str]
    ordinals: list[int] | None = None


@dataclass
class RenderedQuestion:
    suffix_ids: list[int]
    option_spans: list[tuple[int, int]]  # per input candidate: [start, end) in suffix
    label_ids: list[int] | None  # per input candidate readout token; None -> pointer only
    display_order: list[int]  # input indices in rendered order


def render_state(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def prefix_head(tok: Tokenizer) -> list[int]:
    """The constant start of every prefix ([bos] + instructions + <state>).
    Its KV cache does not depend on what follows, so servers encode it once."""
    native = getattr(tok, "decision_chat", None)
    if native is not None:
        return tok.encode(native[0] + SYSTEM + "\n\n<state>\n")
    return ([tok.bos_id] if tok.bos_id is not None else []) + tok.encode(
        USER_OPEN + SYSTEM + "\n\n<state>\n"
    )


def model_open(tok):
    native = getattr(tok, "decision_chat", None)
    return native[1] if native is not None else MODEL_OPEN


def render_prefix(state, tok: Tokenizer, max_state_tokens: int | None = None) -> list[int]:
    """[bos] user-turn open + system text + state block."""
    body = tok.encode(render_state(state))
    if max_state_tokens is not None and len(body) > max_state_tokens:
        # keep both ends: headers and the latest facts are the usual evidence
        keep_head = max_state_tokens // 2
        keep_tail = max_state_tokens - keep_head
        body = body[:keep_head] + tok.encode("\n[...]\n") + body[len(body) - keep_tail :]
    tail = tok.encode("\n</state>\n\n")
    return prefix_head(tok) + body + tail


def _canon(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def canonical_order(descriptions: list[str]) -> list[int]:
    """Content-derived display order: invariant to input permutation."""
    return sorted(
        range(len(descriptions)), key=lambda i: (_canon(descriptions[i]), descriptions[i])
    )


class _Builder:
    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.ids: list[int] = []

    def add(self, text: str) -> tuple[int, int]:
        s = len(self.ids)
        self.ids.extend(self.tok.encode(text))
        return s, len(self.ids)


def render_question(
    q: QuestionView, tok: Tokenizer, max_label_candidates: int = 26
) -> RenderedQuestion:
    n = len(q.descriptions)
    if n < 2:
        raise ValueError("a decision needs at least two candidates")
    b = _Builder(tok)
    b.add("Question: " + q.instruction.strip() + "\n\n")
    spans: list[tuple[int, int]] = [(0, 0)] * n

    if q.type == "noul":
        false_d, true_d = q.descriptions
        yes = true_d if _canon(true_d) not in TRIVIAL_NOUL else "the statement holds"
        no = false_d if _canon(false_d) not in TRIVIAL_NOUL else "the statement does not hold"
        spans[1] = b.add("Answer Yes if: " + yes.strip() + "\n")
        spans[0] = b.add("Answer No if: " + no.strip() + "\n")
        b.add("\nAnswer Yes or No.")
        b.add(model_open(tok) + NOUL_CUE)
        labels = [tok.single_token_id(" No"), tok.single_token_id(" Yes")]
        return RenderedQuestion(b.ids, spans, labels, [1, 0])

    if q.type == "score":
        ords = q.ordinals if q.ordinals is not None else list(range(n))
        order = sorted(range(n), key=lambda i: ords[i])
        if all(0 <= o <= 9 for o in ords):
            names = [str(ords[i]) for i in order]
        elif n <= max_label_candidates:
            names = LETTERS[:n]
        else:
            names = None
        b.add("Levels (ordered from lowest to highest):\n")
    else:
        order = canonical_order(q.descriptions)
        names = LETTERS[:n] if n <= max_label_candidates else None
        b.add("Options:\n")

    for k, i in enumerate(order):
        desc = q.descriptions[i].strip()
        if names is None:
            spans[i] = b.add(f"- {desc}\n")
        elif _canon(desc) == _canon(names[k]):
            spans[i] = b.add(f"({names[k]})\n")
        else:
            spans[i] = b.add(f"({names[k]}) {desc}\n")
    b.add("\nPick the single best " + ("level." if q.type == "score" else "option."))
    b.add(model_open(tok) + (CHOICE_CUE if names is not None else "The answer is:"))

    label_ids = None
    if names is not None:
        label_ids = [0] * n
        for k, i in enumerate(order):
            label_ids[i] = tok.single_token_id(names[k])
    return RenderedQuestion(b.ids, spans, label_ids, order)
