"""One-position letter probabilities from vLLM, local HF models, or fixtures."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any, Protocol

from ayaka.eval.read_artifact import fingerprint

from .policy import normalize
from .prompt import PROMPT_VARIANTS

Messages = list[dict[str, str]]
READOUT = "canonical_letter_raw"
_TOKENIZERS: dict[tuple[str, str], Any] = {}
_TOKENIZER_LOCK = Lock()


def cached_tokenizer(model: str, revision: str):
    if not revision or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ValueError("tokenizer requires a pinned immutable revision")
    with _TOKENIZER_LOCK:
        key = (model, revision)
        if key not in _TOKENIZERS:
            from transformers import AutoTokenizer

            _TOKENIZERS[key] = AutoTokenizer.from_pretrained(
                model, revision=revision, local_files_only=True
            )
        return _TOKENIZERS[key]


def token_input(tokenizer, messages, letters, kwargs):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    actual = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, **kwargs
    )
    if actual != prefix or not prefix:
        raise ValueError("chat-template token ids differ from the canonical answer prefix")
    ids = canonical_letter_ids(tokenizer, prompt, letters)
    return {
        "input_token_ids": prefix,
        "input_token_ids_sha256": fingerprint(prefix),
        "canonical_token_ids": ids,
        "canonical_token_ids_sha256": fingerprint(ids),
    }


@dataclass(frozen=True)
class ReadResult:
    letter_probs: dict[str, float]
    input_tokens: int
    output_tokens: int
    latency_s: float
    letter_log_masses: dict[str, float] | None = None
    input_token_ids: list[int] = field(default_factory=list)
    canonical_token_ids: dict[str, list[int]] = field(default_factory=dict)
    token_logits: dict[int, float] = field(default_factory=dict)


class LetterReader(Protocol):
    def read(self, messages: Messages, letters: list[str]) -> ReadResult: ...


def canonical_logprobs(top_logprobs: list[dict], letters: list[str]) -> dict[str, float]:
    """Only exact canonical token strings count; aliases invalidate the read."""
    masses = {}
    aliases = []
    for entry in top_logprobs:
        token = entry["token"]
        if token not in letters and token.strip() in letters:
            aliases.append(token)
        if token in letters:
            value = entry["logprob"]
            if token in masses or type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"invalid or duplicate canonical logprob: {token!r}")
            masses[token] = value
    if aliases:
        raise ValueError(f"rejected noncanonical letter aliases: {aliases!r}")
    missing = [letter for letter in letters if letter not in masses]
    if missing:
        raise ValueError(
            f"missing canonical letter logprobs: {missing}; require processed_logprobs "
            "and --max-logprobs 26; top-20 must cover every letter"
        )
    return masses


def logmass_probs(masses: dict[str, float]) -> dict[str, float]:
    if not masses or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in masses.values()
    ):
        raise ValueError("complete finite raw log masses required")
    peak = max(masses.values())
    return normalize({letter: math.exp(value - peak) for letter, value in masses.items()})


def top_letter_probs(top_logprobs: list[dict], letters: list[str]) -> dict[str, float]:
    return logmass_probs(canonical_logprobs(top_logprobs, letters))


class VLLMChatReader:
    def __init__(
        self,
        url: str,
        model: str,
        timeout: float = 120.0,
        *,
        chat_template_kwargs: dict | None = None,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        tokenizer: Any = None,
    ):
        root = url.rstrip("/")
        self.url = (
            root
            if root.endswith("/chat/completions")
            else (
                f"{root}/chat/completions"
                if root.endswith("/v1")
                else f"{root}/v1/chat/completions"
            )
        )
        self.model = model
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision or revision
        self.tokenizer = tokenizer
        self.backend = "vllm"
        self.readout = READOUT
        self.logprobs_mode = "raw_logits"
        self.dtype = "bfloat16"
        self.rejected_aliases: list[str] = []
        self.timeout = timeout
        self.chat_template_kwargs = (
            {"enable_thinking": False}
            if chat_template_kwargs is None
            else chat_template_kwargs.copy()
        )

    def describe(self, messages, letters):
        if self.tokenizer is None:
            self.tokenizer = cached_tokenizer(self.model, self.tokenizer_revision)
        return token_input(self.tokenizer, messages, letters, self.chat_template_kwargs)

    def read(self, messages: Messages, letters: list[str]) -> ReadResult:
        start = time.perf_counter()
        description = self.describe(messages, letters)
        ids = description["canonical_token_ids"]
        requested = [ids[letter][0] for letter in letters]
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1,
            "logprobs": True,
            "logprob_token_ids": requested,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "temperature": 0,
            "chat_template_kwargs": self.chat_template_kwargs,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Names/"token_id:N" wire strings verified at v0.30.0 commit ced6857,
        # source receipt .dev/codex-vllm-source-audit-20261004-r2.json.
        # Kernel parity remains unverified until the runner's fixed-cohort gate.
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise ValueError(
                "vLLM exact token gather rejected; require 0.30.0 logprob_token_ids, "
                "return_tokens_as_token_ids and server --logprobs-mode raw_logits"
            ) from exc
        content = result["choices"][0]["logprobs"]["content"]
        if len(content) != 1:
            raise ValueError("exact read requires one assistant position")
        position = content[0]
        selected = gather_token_logits(position["top_logprobs"], requested, position["token"])
        masses = {letter: selected[ids[letter][0]] for letter in letters}
        probs = logmass_probs(masses)
        usage = result["usage"]
        if result.get("prompt_token_ids") != description["input_token_ids"]:
            raise ValueError("server prompt token ids differ from the client-bound input")
        if (
            usage["prompt_tokens"] != len(description["input_token_ids"])
            or usage["completion_tokens"] != 1
        ):
            raise ValueError("server token usage differs from the bound one-position input")
        return ReadResult(
            probs,
            int(usage["prompt_tokens"]),
            int(usage["completion_tokens"]),
            time.perf_counter() - start,
            masses,
            description["input_token_ids"],
            ids,
            selected,
        )


def wire_token_id(token):
    if not isinstance(token, str) or not re.fullmatch(r"token_id:\d+", token):
        raise ValueError("expected vLLM return_tokens_as_token_ids token_id:N")
    return int(token.split(":", 1)[1])


def gather_token_logits(entries, requested, sampled_token):
    """Wire 'logprob' means raw logit; reject losses rather than invent mass."""
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("requested canonical token ids must be distinct")
    sampled = wire_token_id(sampled_token)
    selected, seen = {}, set()
    for entry in entries:
        token_id = wire_token_id(entry["token"])
        if token_id in seen:
            raise ValueError(f"duplicate gathered token id: {token_id}")
        seen.add(token_id)
        if token_id not in requested:
            if token_id != sampled:
                raise ValueError("unexpected token id; only extra sampled token may be ignored")
            continue
        value = entry["logprob"]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("gathered raw logits must be finite")
        if value <= -9999:
            raise ValueError("clipped raw logit at vLLM wire floor -9999")
        selected[token_id] = value
    if set(selected) != set(requested):
        raise ValueError("missing requested canonical token ids")
    return selected


def letter_token_ids(tokenizer: Any) -> dict[str, list[int]]:
    """Diagnostic alias_sum only: enumerate vocabulary ids by decoded text."""
    ids = {chr(ord("A") + i): [] for i in range(26)}
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id]).strip()
        if text in ids:
            ids[text].append(token_id)
    return ids


def canonical_letter_ids(tokenizer: Any, prompt: str, letters: list[str]) -> dict[str, list[int]]:
    """Find the one appended token at the actual assistant answer position."""
    if not letters or len(letters) > 26 or len(set(letters)) != len(letters):
        raise ValueError("canonical read needs 1..26 distinct letters")
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    ids = {}
    for letter in letters:
        completed = tokenizer.encode(prompt + letter, add_special_tokens=False)
        if completed[:-1] != prefix or len(completed) != len(prefix) + 1:
            raise ValueError(f"letter {letter} is not one canonical token at the answer position")
        ids[letter] = [completed[-1]]
    if len({values[0] for values in ids.values()}) != len(letters):
        raise ValueError("canonical letters must have distinct token ids")
    return ids


def aggregate_letter_logits(
    logits: Sequence[float], ids: dict[str, list[int]], letters: list[str]
) -> dict[str, float]:
    """Logsumexp across matching ids, then softmax across letters."""
    sums = {}
    for letter in letters:
        values = [float(logits[i]) for i in ids.get(letter, [])]
        if not values:
            raise ValueError(f"tokenizer has no token for letter {letter}")
        peak = max(values)
        sums[letter] = peak + math.log(math.fsum(math.exp(value - peak) for value in values))
    peak = max(sums.values())
    return normalize({letter: math.exp(value - peak) for letter, value in sums.items()})


class HFReader:
    def __init__(
        self,
        model_id_or_path: str,
        device: str = "cpu",
        dtype: str = "float32",
        *,
        readout: str = READOUT,
        revision: str | None = None,
        chat_template_kwargs: dict | None = None,
    ):
        if readout not in (READOUT, "alias_sum"):
            raise ValueError("HF readout must be canonical_letter_raw or diagnostic alias_sum")
        self.model_id_or_path = model_id_or_path
        self.backend = "hf"
        self.readout = readout
        self.revision = revision
        self.tokenizer_revision = revision
        self.logprobs_mode = "raw_logits"
        self.chat_template_kwargs = (
            {"enable_thinking": False}
            if chat_template_kwargs is None
            else chat_template_kwargs.copy()
        )
        self.device = device
        self.dtype = dtype
        self.tokenizer: Any = None
        self.model: Any = None
        self._letter_ids: dict[str, list[int]] | None = None
        self._lock = Lock()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id_or_path, revision=self.revision, local_files_only=True
            )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_id_or_path,
                torch_dtype=getattr(torch, self.dtype) if self.dtype != "auto" else "auto",
                local_files_only=True,
                revision=self.revision,
            )
            .to(self.device)
            .eval()
        )

    def describe(self, messages, letters):
        if self.tokenizer is None:
            self.tokenizer = cached_tokenizer(self.model_id_or_path, self.tokenizer_revision)
        return token_input(self.tokenizer, messages, letters, self.chat_template_kwargs)

    def read(self, messages: Messages, letters: list[str]) -> ReadResult:
        import torch

        start = time.perf_counter()
        with self._lock, torch.inference_mode():
            if self.model is None:
                self._load()
            if self.readout == "alias_sum" and self._letter_ids is None:
                self._letter_ids = letter_token_ids(self.tokenizer)
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs
            )
            ids = (
                canonical_letter_ids(self.tokenizer, prompt, letters)
                if self.readout == READOUT
                else self._letter_ids
            )
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                **self.chat_template_kwargs,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            if encoded["input_ids"][0].tolist() != self.tokenizer.encode(
                prompt, add_special_tokens=False
            ):
                raise ValueError("chat-template token ids differ from the canonical answer prefix")
            logits = self.model(**encoded).logits[0, -1]
            required = [token_id for letter in letters for token_id in ids[letter]]
            gathered = logits[required].float().cpu().tolist()
            selected = dict(zip(required, gathered, strict=True))
            masses = {}
            for letter in letters:
                values = [selected[i] for i in ids[letter]]
                if not values or any(not math.isfinite(value) for value in values):
                    raise ValueError(f"missing or nonfinite logits for {letter}")
                peak = max(values)
                masses[letter] = peak + math.log(
                    math.fsum(math.exp(value - peak) for value in values)
                )
            probs = logmass_probs(masses)
            return ReadResult(
                probs,
                int(encoded["input_ids"].shape[-1]),
                1,
                time.perf_counter() - start,
                masses,
                encoded["input_ids"][0].tolist(),
                ids,
                selected,
            )


class FakeReader:
    """Thread-safe scripted probabilities, or a per-call fixture function."""

    backend = "fixture"
    readout = READOUT
    logprobs_mode = "fixture_probabilities"
    tokenizer_revision = "fixture-tokenizer-v1"

    def describe(self, messages, letters):
        prefix = [int(fingerprint(messages)[:8], 16)]
        ids = {letter: [ord(letter)] for letter in letters}
        return {
            "input_token_ids": prefix,
            "input_token_ids_sha256": fingerprint(prefix),
            "canonical_token_ids": ids,
            "canonical_token_ids_sha256": fingerprint(ids),
        }

    def __init__(
        self,
        results: Sequence[dict[str, float] | ReadResult]
        | Callable[[Messages, list[str]], dict[str, float] | ReadResult]
        | None = None,
    ):
        self.results = results
        self.calls: list[tuple[Messages, list[str]]] = []
        self._lock = Lock()

    def read(self, messages: Messages, letters: list[str]) -> ReadResult:
        with self._lock:
            index = len(self.calls)
            self.calls.append((messages, letters.copy()))
            if callable(self.results):
                result = self.results(messages, letters)
            elif self.results is None:
                result = dict.fromkeys(letters, 1.0)
            else:
                result = self.results[index]
        if not isinstance(result, ReadResult):
            result = ReadResult(
                normalize({letter: result.get(letter, 0.0) for letter in letters}), 10, 1, 0.01
            )
        description = self.describe(messages, letters)
        masses = result.letter_log_masses or {
            letter: math.log(p) if p > 0 else -1e6 for letter, p in result.letter_probs.items()
        }
        return replace(
            result,
            letter_log_masses=masses,
            input_token_ids=result.input_token_ids or description["input_token_ids"],
            canonical_token_ids=result.canonical_token_ids or description["canonical_token_ids"],
            token_logits=result.token_logits or {ord(k): v for k, v in masses.items()},
        )


def template_kwargs(value: str) -> dict:
    try:
        result = json.loads(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("chat template kwargs must be a JSON object") from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("chat template kwargs must be a JSON object")
    return result


def add_reader_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--chat-template-kwargs", type=template_kwargs)
    parser.add_argument("--hf-model")
    parser.add_argument("--revision", help="resolved model/tokenizer commit")
    parser.add_argument("--readout", choices=[READOUT, "alias_sum"], default=READOUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--state-format", choices=["pretty", "compact"], default="pretty")
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="min")


def reader_from_args(args: argparse.Namespace) -> LetterReader:
    if args.backend == "hf":
        if not args.hf_model:
            raise ValueError("--backend hf requires --hf-model")
        return HFReader(
            args.hf_model,
            args.device,
            args.dtype,
            readout=args.readout,
            revision=args.revision,
            chat_template_kwargs=args.chat_template_kwargs,
        )
    if args.readout != READOUT:
        raise ValueError("alias_sum is an HF-only diagnostic")
    return VLLMChatReader(
        args.vllm_url,
        args.model,
        args.timeout,
        chat_template_kwargs=getattr(args, "chat_template_kwargs", None),
        revision=args.revision,
    )
