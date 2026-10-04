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
from dataclasses import dataclass, replace
from pathlib import Path

from ayaka.eval.read_artifact import fingerprint

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
    generation_input = reader.describe(messages, list(mapping))
    trace = generate_trace(reader, messages, max_tokens)
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
    final_messages = [
        *messages,
        {"role": "assistant", "content": trace.text},
        {"role": "user", "content": FINAL_INSTRUCTION},
    ]
    result = reader.read(final_messages, list(mapping))
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
    }
    return read, metadata


def system_read(reader, state, question, *, policy, **kwargs):
    direct = read_question(reader, state, question, **kwargs)
    if policy.reasoning_route:
        from .router import should_route

        if should_route(
            policy.reasoning_route, question.type, direct.raw_probs, state, question.instruction
        ):
            result, _ = reasoned_read(
                reader,
                state,
                question,
                prompt_variant=kwargs.get("prompt_variant", "min"),
                state_format=kwargs.get("state_format", "pretty"),
                max_tokens=policy.reasoning_route["max_tokens"],
            )
            return replace(
                result,
                input_tokens=direct.input_tokens + result.input_tokens,
                output_tokens=direct.output_tokens + result.output_tokens,
                latency_s=direct.latency_s + result.latency_s,
            )
    return direct
