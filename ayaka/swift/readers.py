"""One-position letter probabilities from vLLM, local HF models, or fixtures."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

from .policy import normalize
from .prompt import PROMPT_VARIANTS

Messages = list[dict[str, str]]


@dataclass(frozen=True)
class ReadResult:
    letter_probs: dict[str, float]
    input_tokens: int
    output_tokens: int
    latency_s: float
    letter_log_masses: dict[str, float] | None = None


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
        self.backend = "vllm"
        self.readout = "canonical_letter"
        self.logprobs_mode = "processed_logprobs"
        self.rejected_aliases: list[str] = []
        self.timeout = timeout
        self.chat_template_kwargs = (
            {"enable_thinking": False}
            if chat_template_kwargs is None
            else chat_template_kwargs.copy()
        )

    def read(self, messages: Messages, letters: list[str]) -> ReadResult:
        start = time.perf_counter()
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": min(20, len(letters)),
            "temperature": 0,
            "chat_template_kwargs": self.chat_template_kwargs,
            "structured_outputs": {"choice": letters},
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        entries = result["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        self.rejected_aliases.extend(
            entry["token"]
            for entry in entries
            if entry["token"] not in letters and entry["token"].strip() in letters
        )
        masses = canonical_logprobs(entries, letters)
        probs = logmass_probs(masses)
        usage = result["usage"]
        return ReadResult(
            probs,
            int(usage["prompt_tokens"]),
            int(usage["completion_tokens"]),
            time.perf_counter() - start,
            masses,
        )


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
        readout: str = "canonical_letter",
        revision: str | None = None,
        chat_template_kwargs: dict | None = None,
    ):
        if readout not in ("canonical_letter", "alias_sum"):
            raise ValueError("HF readout must be canonical_letter or diagnostic alias_sum")
        self.model_id_or_path = model_id_or_path
        self.backend = "hf"
        self.readout = readout
        self.revision = revision
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
                if self.readout == "canonical_letter"
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
                probs, int(encoded["input_ids"].shape[-1]), 1, time.perf_counter() - start, masses
            )


class FakeReader:
    """Thread-safe scripted probabilities, or a per-call fixture function."""

    backend = "fixture"
    readout = "canonical_letter"
    logprobs_mode = "fixture_probabilities"

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
        if isinstance(result, ReadResult):
            return result
        return ReadResult(
            normalize({letter: result.get(letter, 0.0) for letter in letters}), 10, 1, 0.01
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
    parser.add_argument(
        "--readout", choices=["canonical_letter", "alias_sum"], default="canonical_letter"
    )
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
    if args.readout != "canonical_letter":
        raise ValueError("alias_sum is an HF-only diagnostic")
    return VLLMChatReader(
        args.vllm_url,
        args.model,
        args.timeout,
        chat_template_kwargs=getattr(args, "chat_template_kwargs", None),
        revision=args.revision,
    )
