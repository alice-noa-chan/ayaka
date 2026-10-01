"""Tokenizer adapters.

The decision prompt is assembled from pieces (state, question, each
option, the answer cue) that are tokenized separately and concatenated,
so option spans are known exactly without offset mapping. Both the
Gemma adapter and the toy test tokenizer expose the same small surface.
"""

from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    name: str
    bos_id: int
    pad_id: int

    def encode(self, text: str) -> list[int]: ...

    def single_token_id(self, text: str) -> int: ...


class HFTokenizer:
    """Wraps a Hugging Face tokenizer (Gemma 4 chat tokenizer)."""

    def __init__(self, hf_tokenizer, name: str):
        self.hf = hf_tokenizer
        self.name = name
        bos = hf_tokenizer.bos_token_id
        self.bos_id = int(bos) if bos is not None else None
        pad = hf_tokenizer.pad_token_id
        self.pad_id = int(pad if pad is not None else 0)
        self._single: dict[str, int] = {}
        self.decision_chat = None
        if (
            getattr(hf_tokenizer, "chat_template", None)
            and "<|turn>" not in hf_tokenizer.get_vocab()
        ):
            marker = "AYAKA_CONTENT_BOUNDARY"
            rendered = hf_tokenizer.apply_chat_template(
                [{"role": "user", "content": marker}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if rendered.count(marker) != 1:
                raise ValueError("chat template must preserve user content exactly once")
            self.decision_chat = tuple(rendered.split(marker))

    @classmethod
    def from_pretrained(cls, repo: str, revision: str | None = None) -> HFTokenizer:
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(repo, revision=revision), repo)

    @classmethod
    def for_config(cls, cfg) -> HFTokenizer:
        """The backbone's tokenizer at the config's pinned revision."""
        return cls.from_pretrained(cfg.backbone, getattr(cfg, "backbone_revision", None))

    def encode(self, text: str) -> list[int]:
        return self.hf.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.hf.decode(ids)

    def single_token_id(self, text: str) -> int:
        if text not in self._single:
            ids = self.encode(text)
            if len(ids) != 1:
                raise ValueError(f"label {text!r} is not a single token: {ids}")
            self._single[text] = ids[0]
        return self._single[text]


class ToyTokenizer:
    """Deterministic char-level tokenizer with a few whole-word tokens.

    Only for CPU tests with the random tiny backbone: registered strings
    (chat markers, answer labels) are matched longest-first, everything
    else falls back to one id per character.
    """

    SPECIAL = ("<|turn>", "<turn|>", " Yes", " No")

    def __init__(self, vocab_size: int = 512):
        self.name = "toy"
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.bos_id = 2
        self._reserved = {s: 3 + i for i, s in enumerate(self.SPECIAL)}
        self._base = 3 + len(self.SPECIAL)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            for s, sid in self._reserved.items():
                if text.startswith(s, i):
                    ids.append(sid)
                    i += len(s)
                    break
            else:
                ids.append(self._base + ord(text[i]) % (self.vocab_size - self._base))
                i += 1
        return ids

    def single_token_id(self, text: str) -> int:
        ids = self.encode(text)
        if len(ids) != 1:
            raise ValueError(f"label {text!r} is not a single token: {ids}")
        return ids[0]
