"""Primitive semantics API (docs.md section 15).

One kernel — Decision(state, questions[]) -> probability[K] per
question — read three ways:

- Choice: runtime candidate descriptions -> distribution
- Noul:   proposition -> [P(false), P(true)] -> P(true)
- Score:  ordered semantic levels -> distribution + expected value

Serving path: the state prefix is encoded once, its KV cache is
repeated per question, and all question suffixes run in one batched
forward. Results equal running each question alone (docs.md section 48,
invariant 5). Choice sets larger than the label alphabet use a
pointer shortlist, then label readout inside the shortlist.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

from .collate import EncodedQuestion, encode_decision, suffix_rows
from .model.electra import ElectraDecisionModel
from .model.ragged import ragged_softmax
from .prompt import QuestionView, prefix_head
from .tokenization import Tokenizer


@dataclass
class QuestionSpec:
    type: str  # "choice" | "noul" | "score"
    instruction: str
    candidates: list[str]  # noul: [false_description, true_description]
    ordinals: list[int] | None = None  # score only (addendum A3)

    def __post_init__(self):
        if self.type == "score" and self.ordinals is None:
            self.ordinals = list(range(len(self.candidates)))

    def view(self) -> QuestionView:
        return QuestionView(self.type, self.instruction, list(self.candidates), self.ordinals)


@dataclass
class DecisionResult:
    type: str
    probs: list[float]  # aligned with QuestionSpec.candidates
    distribution: dict[str, float]
    expected: float | None = None
    extras: dict = field(default_factory=dict)


class Decision:
    def __init__(self, model: ElectraDecisionModel, tok: Tokenizer, max_seq_len: int | None = None):
        self.model = model
        self.tok = tok
        cfg = model.cfg
        # inference budget; checkpoints saved before the field default to 8192
        self.max_seq_len = max_seq_len or max(cfg.serve_max_seq_len, cfg.max_seq_len)
        self.max_labels = model.cfg.max_label_candidates
        self.reuse_head = True  # encode the constant prompt head once
        self._head: tuple[list[int], object, torch.device] | None = None

    def _device(self, device):
        return device or next(self.model.parameters()).device

    @torch.no_grad()
    def _run(self, state, views: list[QuestionView], device) -> list[list[float]]:
        prefix, items = encode_decision(state, views, self.tok, self.max_seq_len, self.max_labels)
        return self._run_encoded(prefix, items, device)

    @torch.no_grad()
    def _run_encoded(
        self, prefix: list[int], items: list[EncodedQuestion], device
    ) -> list[list[float]]:
        dev = self._device(device)
        cache = self._prefix_cache(prefix, torch.device(dev))
        cache.batch_repeat_interleave(len(items))
        batch = suffix_rows(items, len(prefix), self.tok.pad_id).to(dev)
        out = self.model(batch, apply_temperature=True, past_key_values=cache)
        p = ragged_softmax(out.logits, out.cand_cu).tolist()
        cu = out.cand_cu.tolist()
        return [p[cu[i] : cu[i + 1]] for i in range(len(items))]

    def _prefix_cache(self, prefix: list[int], dev: torch.device):
        """KV cache for ``prefix``. The constant head is encoded once per
        Decision and deep-copied, so a request only encodes its own state."""
        if self.reuse_head:
            if self._head is None or self._head[2] != dev:
                head = prefix_head(self.tok)
                self._head = (head, self.model.encode_prefix(torch.tensor([head], device=dev)), dev)
            head, head_cache, _ = self._head
            if len(prefix) > len(head) and prefix[: len(head)] == head:
                rest = torch.tensor([prefix[len(head) :]], device=dev)
                return self.model.encode_prefix(rest, cache=copy.deepcopy(head_cache))
        return self.model.encode_prefix(torch.tensor([prefix], device=dev))

    @torch.no_grad()
    def decide(self, state, questions: list[QuestionSpec], device=None) -> list[DecisionResult]:
        self.model.eval()
        views = [q.view() for q in questions]
        probs = self._run(state, views, device)
        for i, v in enumerate(views):
            if v.type == "choice" and len(v.descriptions) > self.max_labels:
                probs[i] = self._shortlist(state, v, probs[i], device)
        results = []
        for q, p in zip(questions, probs, strict=True):
            res = DecisionResult(
                type=q.type, probs=p, distribution=dict(zip(q.candidates, p, strict=True))
            )
            if q.type == "score":
                res.expected = sum(o * pi for o, pi in zip(q.ordinals, p, strict=True))
            if q.type == "noul":
                res.extras["p_true"] = p[1]
            results.append(res)
        return results

    def _shortlist(self, state, v: QuestionView, pointer_p: list[float], device) -> list[float]:
        """Pointer picks the top label-alphabet-many candidates; label
        readout re-ranks inside the shortlist, keeping its total mass."""
        top = sorted(range(len(pointer_p)), key=lambda i: -pointer_p[i])[: self.max_labels]
        mass = sum(pointer_p[i] for i in top)
        sub = QuestionView("choice", v.instruction, [v.descriptions[i] for i in top])
        inner = self._run(state, [sub], device)[0]
        out = list(pointer_p)
        for j, i in enumerate(top):
            out[i] = mass * inner[j]
        return out

    # ------------------------------------------------------ convenience API
    def choice(
        self, state, instruction: str, candidates: list[str], device=None
    ) -> dict[str, float]:
        return self.decide(state, [QuestionSpec("choice", instruction, candidates)], device)[
            0
        ].distribution

    def noul(
        self,
        state,
        proposition: str,
        device=None,
        false_desc: str = "false",
        true_desc: str = "true",
    ) -> float:
        res = self.decide(
            state, [QuestionSpec("noul", proposition, [false_desc, true_desc])], device
        )[0]
        return res.extras["p_true"]

    def score(
        self,
        state,
        instruction: str,
        levels: list[str],
        ordinals: list[int] | None = None,
        device=None,
    ):
        res = self.decide(state, [QuestionSpec("score", instruction, levels, ordinals)], device)[0]
        return res.expected, res.distribution
