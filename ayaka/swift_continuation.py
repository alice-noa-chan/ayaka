"""Recipe-bound worked steps and canonical typed reads on the same cache.

The complete three-turn read is tokenized by the native chat template. Its
common token prefix is reused from generation, never from another question.
Some native templates remove generation-only channel markers when rendering
an assistant history turn. Preserve the complete original conversation and
explicitly reread that canonical final chat when its cache prefix changes.
"""

from __future__ import annotations

import copy

import torch

from .collate import EncodedQuestion, suffix_rows
from .input_contract import swift_wire_question, validate_tokenizer
from .model.electra import PRIMITIVE_INDEX
from .model.ragged import ragged_softmax
from .prompt import RenderedQuestion
from .reasoning_pipeline import TraceFailure, TraceGenerator
from .swift.prompt import render_question
from .swift.readers import token_input
from .swift.reasoning import FINAL_INSTRUCTION, TRACE_INSTRUCTION


class SwiftTraceGenerator(TraceGenerator):
    # EOS is counted but the final template, rather than generation, feeds it
    # into the cache. Sliding caches then need no rollback merely to remove EOS.
    cache_eos = False

    def __init__(self, model, tok, contract, **kwargs):
        super().__init__(model, tok, **kwargs)
        self.contract = copy.deepcopy(contract)
        self.encoding = self.contract["input_encoding"]
        validate_tokenizer(self.contract, tok)

    def _tokens(self, messages, letters):
        validate_tokenizer(self.contract, self.tok)
        return token_input(
            self.tok.hf,
            messages,
            letters,
            {**self.encoding["chat_template_kwargs"], "return_dict": False},
        )

    def messages_for(self, state, spec):
        _, wire, _ = swift_wire_question(spec.view())
        original, _ = render_question(
            state,
            wire,
            prompt_variant=self.encoding["prompt_variant"],
            state_format=self.encoding["state_format"],
        )
        return [{"role": "user", "content": original[-1]["content"] + "\n\n" + TRACE_INSTRUCTION}]

    def prepare(self, messages):
        return self._tokens(messages, ["A"])["input_token_ids"], None

    def reserve_tokens(self, messages, spec):
        final = self._final_messages(messages, "")
        return max(1, len(self.prepare(final)[0]) - len(self.prepare(messages)[0]))

    def conversation_tokens(self, messages):
        """Bind the supplied conversation before the generation-only cue."""
        validate_tokenizer(self.contract, self.tok)
        ids = self.tok.hf.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            **{**self.encoding["chat_template_kwargs"], "return_dict": False},
        )
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return list(ids)

    @staticmethod
    def _final_messages(messages, text):
        return [
            *messages,
            {"role": "assistant", "content": text},
            {"role": "user", "content": FINAL_INSTRUCTION},
        ]

    def generate_trace(self, messages, budget, reserve=0):
        trace = super().generate_trace(messages, budget, reserve)
        trace.messages = copy.deepcopy(messages)
        try:
            trace.text = self.tok.hf.decode(
                [token for token in trace.token_ids if token not in self.eos],
                skip_special_tokens=True,
            )
        except (RuntimeError, ValueError) as exc:
            trace.finish_reason = "generation_error"
            trace.error = f"{type(exc).__name__}: {str(exc)[:250]}"
            raise TraceFailure(trace) from exc
        return trace

    @torch.inference_mode()
    def readout(self, trace, spec):
        question, wire, semantic = swift_wire_question(spec.view())
        displayed = [semantic[label] for label in wire.labels]
        letters = [chr(65 + i) for i in range(len(displayed))]
        bound = self._tokens(self._final_messages(trace.messages, trace.text), letters)
        ids = bound["input_token_ids"]
        if len(ids) > self.max_context:
            raise ValueError("complete Swift reasoning readout exceeds context; refuse truncation")
        cached_tokens = trace.token_ids[:-1] if trace.finish_reason == "eos" else trace.token_ids
        generated = trace.input_ids + cached_tokens
        common = 0
        while common < min(len(ids), len(generated)) and ids[common] == generated[common]:
            common += 1
        original = self.conversation_tokens(trace.messages)
        if (
            not original
            or ids[: len(original)] != original
            or trace.input_ids[: len(original)] != original
        ):
            raise ValueError("Swift final template rewrites the original conversation")
        cache = trace.cache
        if common < len(trace.input_ids):
            # Gemma 4's empty thought channel belongs to the generation cue;
            # its history renderer removes it. A full canonical read avoids
            # feeding final-chat tokens into a cache from a different prefix.
            common, cache = 0, None
            trace.cache = None
            trace.readout_execution = "canonical_full_read"
        else:
            trace.readout_execution = "cache_reuse"
            common = min(common, len(ids) - 1)  # retain the final answer position
        if cache is not None and common != len(generated):
            crop = getattr(trace.cache, "crop", None)
            if not callable(crop):
                raise ValueError("Swift trace cache cannot roll back the final template boundary")
            crop(common - len(generated))
            if int(trace.cache.get_seq_length()) != common:
                raise ValueError("Swift trace cache rollback length differs from token prefix")
        original_ids = [candidate.id for candidate in question.candidates]
        label_ids = [
            bound["canonical_token_ids"][letters[displayed.index(key)]][0] for key in original_ids
        ]
        rendered = RenderedQuestion(
            ids[common:],
            [(0, 1)] * len(original_ids),  # native LM-only readout never consumes option features
            label_ids,
            [original_ids.index(key) for key in displayed],
        )
        item = EncodedQuestion(ids[:common], rendered, PRIMITIVE_INDEX[spec.type])
        batch = suffix_rows([item], common, self.tok.pad_id).to(self.model.embed_weight().device)
        trace.readout_tokens += len(rendered.suffix_ids)
        trace.readout_input_ids = ids
        trace.readout_label_ids = label_ids
        out = self.model(batch, past_key_values=cache, apply_temperature=self.apply_temperature)
        return ragged_softmax(out.logits, out.cand_cu).tolist()
