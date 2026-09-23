"""Tokenizer interface (addendum A7).

The model only depends on this protocol: encode/decode/vocab_size plus
lookup of reserved special-token ids. Production uses a 64K multilingual
BPE trained as part of foundation pretraining; tests use a deterministic
hash tokenizer that needs no assets.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

from .special_tokens import NUM_SPECIAL_TOKENS, SPECIAL_TOKEN_IDS


@runtime_checkable
class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def token_id(self, name: str) -> int: ...


class HashTokenizer:
    """Deterministic tokenizer for tests: special tokens by name,
    everything else hashed into the remaining vocab space."""

    def __init__(self, vocab_size: int = 64_000):
        self._vocab_size = vocab_size
        if vocab_size <= NUM_SPECIAL_TOKENS:
            raise ValueError("vocab_size must exceed special-token count")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def token_id(self, name: str) -> int:
        return SPECIAL_TOKEN_IDS[name]

    def encode(self, text: str) -> list[int]:
        ids = []
        for piece in text.split(" "):
            if not piece:
                continue
            if piece in SPECIAL_TOKEN_IDS:
                ids.append(SPECIAL_TOKEN_IDS[piece])
            else:
                h = int.from_bytes(hashlib.sha256(piece.encode()).digest()[:4], "little")
                ids.append(NUM_SPECIAL_TOKENS + h % (self._vocab_size - NUM_SPECIAL_TOKENS))
        return ids

    def decode(self, ids: list[int]) -> str:
        inv = {v: k for k, v in SPECIAL_TOKEN_IDS.items()}
        return " ".join(inv.get(i, f"<tok:{i}>") for i in ids)


class HFTokenizer:
    """Wraps a tokenizers.Tokenizer, reserving ids for special tokens."""

    def __init__(self, tok):  # tok: tokenizers.Tokenizer
        self._tok = tok
        for name in SPECIAL_TOKEN_IDS:
            if self._tok.token_to_id(name) is None:
                self._tok.add_special_tokens([name])

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def token_id(self, name: str) -> int:
        tid = self._tok.token_to_id(name)
        if tid is None:
            raise KeyError(f"special token {name!r} missing from tokenizer")
        return tid

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


def train_bpe(
    texts,
    vocab_size: int = 64_000,
    save_path: str | None = None,
) -> HFTokenizer:
    """Train a byte-level multilingual BPE (addendum A7).

    ``tokenizers`` is an optional dependency (remote image). Special
    tokens are reserved *inside* the vocab budget so the resulting
    vocab_size never exceeds ``vocab_size``.
    """
    from tokenizers import Tokenizer as _Tk
    from tokenizers import decoders, models, pre_tokenizers, trainers

    tok = _Tk(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKEN_IDS),
        show_progress=False,
    )
    tok.train_from_iterator(iter(texts), trainer=trainer)
    if save_path:
        tok.save(save_path)
    return HFTokenizer(tok)


def load_bpe(path: str) -> HFTokenizer:
    """Load a tokenizers JSON saved by train_bpe."""
    from tokenizers import Tokenizer as _Tk

    return HFTokenizer(_Tk.from_file(path))
