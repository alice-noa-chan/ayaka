"""Bounded frozen-model worked steps followed by the canonical raw letter read.

Trace text stays internal to the read (and its offline provenance), never in the
TypeSafe answer. Routing first needs a direct read, so a routed decision costs
that read plus both calls here.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.request
from dataclasses import dataclass, field, fields
from pathlib import Path

from ayaka.eval.read_artifact import fingerprint
from ayaka.reasoning import ReasoningSettings

from .grouping import QuestionRead, read_question
from .prompt import render_question
from .readers import READOUT, HFReader, VLLMChatReader

TRACE_MAX_TOKENS = 384
TRACE_INSTRUCTION = (
    "Work out the relevant facts, rules, dates and arithmetic step by step. "
    "Keep the worked steps brief. Do NOT state a final option letter."
)
FINAL_INSTRUCTION = "Answer with only the option letter."


def reasoning_recipe(max_tokens=TRACE_MAX_TOKENS):
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("trace max_tokens must be a positive integer")
    return {
        "contract": "swift_reasoned_read_v1",
        "instruction": TRACE_INSTRUCTION,
        "final_instruction": FINAL_INSTRUCTION,
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
        "stop": "eos",
        "readout": READOUT,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


@dataclass(frozen=True)
class TraceResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    latency_s: float


class ReasoningFailure(ValueError):
    """Keep observed usage when offline reasoning fails; never conceal a failed read."""

    def __init__(self, diagnostics):
        super().__init__(diagnostics["finish_reason"])
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class ReasoningRead(QuestionRead):
    reasoning: dict = field(default_factory=dict)


def _served_read(read, diagnostics):
    if diagnostics["route"] == "direct" and read.readout == "grouped_approx":
        diagnostics = {**diagnostics, "route": "grouped"}
    values = {item.name: getattr(read, item.name) for item in fields(QuestionRead)}
    values["input_tokens"] += diagnostics["trace_input_tokens"]
    values["output_tokens"] += diagnostics["trace_tokens"]
    values["latency_s"] += diagnostics["trace_latency_s"] + diagnostics["read_latency_s"]
    return ReasoningRead(**values, reasoning=diagnostics)


def generate_trace(reader, messages, max_tokens):
    """Use the same frozen backend; fakes may supply generate_trace directly."""
    if hasattr(reader, "generate_trace"):
        return reader.generate_trace(messages, max_tokens)
    start = time.perf_counter()
    if isinstance(reader, VLLMChatReader):
        # describe binds the generation prefix as well as the later read prefix.
        description = reader.describe(messages, ["A"])
        body = {
            "model": reader.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "ignore_eos": False,
            "return_token_ids": True,
            "chat_template_kwargs": {**reader.chat_template_kwargs, "enable_thinking": False},
        }
        request = urllib.request.Request(
            reader.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=reader.timeout) as response:
            result = json.load(response)
        usage, choice = result["usage"], result["choices"][0]
        if result.get("prompt_token_ids") != description["input_token_ids"] or usage[
            "prompt_tokens"
        ] != len(description["input_token_ids"]):
            raise ValueError("trace generation prompt differs from bound input")
        return TraceResult(
            choice["message"]["content"],
            usage["prompt_tokens"],
            usage["completion_tokens"],
            "eos" if choice["finish_reason"] == "stop" else choice["finish_reason"],
            time.perf_counter() - start,
        )
    if isinstance(reader, HFReader):
        import torch

        with reader._lock, torch.inference_mode():
            if reader.model is None:
                reader._load()
            encoded = reader.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **{**reader.chat_template_kwargs, "enable_thinking": False},
            )
            encoded = {k: v.to(reader.device) for k, v in encoded.items()}
            output = reader.model.generate(**encoded, do_sample=False, max_new_tokens=max_tokens)[
                0, encoded["input_ids"].shape[-1] :
            ].tolist()
            eos = reader.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            return TraceResult(
                reader.tokenizer.decode(output, skip_special_tokens=True),
                encoded["input_ids"].shape[-1],
                len(output),
                "eos" if output and output[-1] in eos else "length",
                time.perf_counter() - start,
            )
    raise ValueError("reasoned read requires a frozen trace generator")


def reasoned_read(
    reader,
    state,
    question,
    *,
    prompt_variant="min",
    state_format="pretty",
    max_tokens=TRACE_MAX_TOKENS,
):
    recipe = reasoning_recipe(max_tokens)
    if getattr(reader, "readout", None) != READOUT or len(question.labels) > 26:
        raise ValueError("reasoning requires a single canonical-letter read")
    if getattr(reader, "chat_template_kwargs", {}).get("enable_thinking", False):
        raise ValueError("reasoning requires enable_thinking=false")
    original, mapping = render_question(
        state, question, prompt_variant=prompt_variant, state_format=state_format
    )
    # Only the original user turn is needed. Its option rendering is unchanged;
    # the direct system instruction would contradict the worked-steps request.
    messages = [{"role": "user", "content": original[-1]["content"] + "\n\n" + TRACE_INSTRUCTION}]
    diagnostics = {
        "trace_input_tokens": 0,
        "trace_tokens": 0,
        "trace_latency_s": 0.0,
        "read_latency_s": 0.0,
        "finish_reason": "generation_error",
        "error": None,
        "usage_complete": False,
    }
    start = time.perf_counter()
    try:
        generation_input = reader.describe(messages, list(mapping))
        trace = generate_trace(reader, messages, max_tokens)
        diagnostics["finish_reason"] = "invalid_trace"
        if (
            not isinstance(trace.text, str)
            or type(trace.input_tokens) is not int
            or trace.input_tokens < 1
            or type(trace.output_tokens) is not int
            or not 0 <= trace.output_tokens <= max_tokens
            or trace.finish_reason not in ("eos", "length")
            or not math.isfinite(trace.latency_s)
            or trace.latency_s <= 0
        ):
            raise ValueError("invalid trace generation result")
        diagnostics.update(
            trace_input_tokens=trace.input_tokens,
            trace_tokens=trace.output_tokens,
            trace_latency_s=trace.latency_s,
            finish_reason="empty_trace",
            usage_complete=True,
        )
        if not trace.text.strip():
            raise ValueError("empty worked steps")
        final_messages = [
            *messages,
            {"role": "assistant", "content": trace.text},
            {"role": "user", "content": FINAL_INSTRUCTION},
        ]
        diagnostics.update(finish_reason="readout_error", usage_complete=False)
        read_start = time.perf_counter()
        result = reader.read(final_messages, list(mapping))
        diagnostics.update(finish_reason=trace.finish_reason, usage_complete=True)
    except Exception as exc:
        # Backend messages can contain private trace text. Expose only the error type.
        diagnostics["error"] = type(exc).__name__
        if diagnostics["finish_reason"] == "readout_error":
            diagnostics["read_latency_s"] = time.perf_counter() - read_start
        else:
            diagnostics["trace_latency_s"] = (
                diagnostics["trace_latency_s"] or time.perf_counter() - start
            )
        raise ReasoningFailure(diagnostics) from exc
    read = QuestionRead(
        {mapping[k]: v for k, v in result.letter_probs.items()},
        trace.input_tokens + result.input_tokens,
        trace.output_tokens + result.output_tokens,
        trace.latency_s + result.latency_s,
        READOUT,
        2,
        {mapping[k]: v for k, v in result.letter_log_masses.items()},
        [
            {
                "messages": final_messages,
                "input_token_ids": result.input_token_ids,
                "canonical_token_ids": result.canonical_token_ids,
                "token_logits": {str(k): v for k, v in result.token_logits.items()},
            }
        ],
    )
    metadata = {
        "recipe": recipe,
        "recipe_sha256": fingerprint(recipe),
        "generation_messages": messages,
        "generation_input": generation_input,
        "trace_tokens": trace.output_tokens,
        "trace_input_tokens": trace.input_tokens,
        "finish_reason": trace.finish_reason,
        "length_capped": trace.finish_reason == "length",
        "trace_latency_s": trace.latency_s,
        "read_latency_s": result.latency_s,
        "read_input_tokens": result.input_tokens,
        "read_output_tokens": result.output_tokens,
        "usage_complete": True,
    }
    return read, metadata


def validate_reasoning(reader, question, settings):
    """Reject unsupported positive budgets before any request starts inference."""
    if settings.budget == 0:
        return
    if (
        question is None
        or len(question.labels) > 26
        or getattr(reader, "readout", None) != READOUT
        or getattr(reader, "chat_template_kwargs", {}).get("enable_thinking", False)
        or not (
            isinstance(reader, (HFReader, VLLMChatReader))
            or callable(getattr(reader, "generate_trace", None))
        )
    ):
        raise ValueError("reasoning not supported by this readout")


def system_read(
    reader, state, question, *, policy, reasoning: ReasoningSettings | None = None, **kwargs
):
    """Honor explicit controls; omitted controls retain the saved routing policy."""
    if reasoning is None and (not policy.reasoning_route or len(question.labels) > 26):
        return read_question(reader, state, question, **kwargs)
    settings = reasoning or ReasoningSettings()
    validate_reasoning(reader, question, settings)
    diagnostics = {
        "settings": settings.as_dict(),
        "budget": settings.budget,
        "route": "direct",
        "finish_reason": "disabled" if settings.budget == 0 else "not_selected",
        "error": None,
        "trace_input_tokens": 0,
        "trace_tokens": 0,
        "trace_latency_s": 0.0,
        "read_latency_s": 0.0,
        "usage_complete": True,
    }
    direct = None
    use_reasoning = settings.mode == "on" and settings.budget > 0
    if not use_reasoning:
        direct = read_question(reader, state, question, **kwargs)
    budget = settings.budget
    if settings.mode == "auto" and budget > 0 and policy.reasoning_route:
        from .router import should_route

        use_reasoning = should_route(
            policy.reasoning_route, question.type, direct.raw_probs, state, question.instruction
        )
        budget = (
            min(budget, policy.reasoning_route["max_tokens"])
            if reasoning is not None
            else policy.reasoning_route["max_tokens"]
        )
    diagnostics["effective_budget"] = budget if use_reasoning else 0
    if not use_reasoning:
        return _served_read(direct, diagnostics)
    try:
        result, metadata = reasoned_read(
            reader,
            state,
            question,
            prompt_variant=kwargs.get("prompt_variant", "min"),
            state_format=kwargs.get("state_format", "pretty"),
            max_tokens=budget,
        )
    except ReasoningFailure as exc:
        diagnostics.update(exc.diagnostics, route="fallback")
        if direct is None:
            direct = read_question(reader, state, question, **kwargs)
        return _served_read(direct, diagnostics)
    diagnostics.update(
        route="reasoned",
        finish_reason=metadata["finish_reason"],
        trace_tokens=metadata["trace_tokens"],
        trace_input_tokens=metadata["trace_input_tokens"],
        trace_latency_s=metadata["trace_latency_s"],
        read_latency_s=metadata["read_latency_s"],
    )
    # Successful reasoned reads already include trace usage; add only the auto baseline.
    values = {item.name: getattr(result, item.name) for item in fields(QuestionRead)}
    if direct is not None:
        for key in ("input_tokens", "output_tokens", "latency_s"):
            values[key] += getattr(direct, key)
    return ReasoningRead(**values, reasoning=diagnostics)
