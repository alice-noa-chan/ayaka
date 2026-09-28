"""Greedy extraction-plan generation with the backbone's own LM head.

The decision checkpoint's LoRA adapter is disabled while generating, so plans
come from the instruction-tuned base model; decision readouts keep the adapter.
Gemma ties input embeddings and the LM head, and final-logit softcapping is
monotonic, so the greedy token is argmax(h @ E^T). Only batched greedy decoding
is implemented: no sampling, no external calls.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from .evidence import EvidenceError
from .evidence_ids import parse_id_plan
from .model.fastpath import prefill_last
from .prompt import MODEL_OPEN, USER_OPEN


def chat_ids(tok, messages) -> list[int]:
    """Chat-template token ids ending in the model's generation cue."""
    hf = getattr(tok, "hf", None)
    if hf is not None and getattr(hf, "chat_template", None):
        chat = hf.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        ids = chat["input_ids"] if isinstance(chat, dict) or hasattr(chat, "input_ids") else chat
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
    # tokenizers without a chat template (tests): the prompt module's markers
    text = "\n\n".join(m["content"] for m in messages)
    return tok.encode(USER_OPEN + text + MODEL_OPEN)


def plan_complete(text: str) -> bool:
    try:
        parse_id_plan(text)
    except EvidenceError:
        return False
    return True


class PlanGenerator:
    """``generate(list_of_messages) -> list[str]``; also callable on one message list."""

    def __init__(
        self,
        model,
        tok,
        max_new_tokens: int = 768,
        max_context: int = 12_288,
        stop_when=plan_complete,
        check_every: int = 4,
        adapter: str = "off",
    ):
        if adapter not in ("off", "on"):
            raise ValueError("adapter must be 'off' (base model) or 'on' (decision LoRA)")
        self.adapter = adapter
        self.model = model
        self.tok = tok
        self.max_new_tokens = max_new_tokens
        self.max_context = max_context
        self.stop_when = stop_when
        self.check_every = check_every
        hf = getattr(tok, "hf", None)
        self.eos = {tok.eos_id} if getattr(tok, "eos_id", None) is not None else set()
        if hf is not None:
            self.eos |= {hf.eos_token_id, hf.convert_tokens_to_ids("<turn|>")}
        self.eos.discard(None)

    def __call__(self, messages) -> str:
        return self.generate([messages])[0]

    def _adapter_off(self):
        """ "off": generate with the base model; "on": keep the decision LoRA
        (identical to a merged export). A merged backbone has no switch."""
        disable = getattr(self.model.backbone, "disable_adapter", None)
        if self.adapter == "on" or disable is None:
            return nullcontext()
        return disable()

    @torch.inference_mode()
    def generate(self, batch_messages) -> list[str]:
        seqs = [chat_ids(self.tok, m) for m in batch_messages]
        longest = max(map(len, seqs))
        if longest + self.max_new_tokens > self.max_context:
            # never truncate evidence silently: the caller falls back to baseline
            raise EvidenceError(f"extraction context {longest} tokens exceeds limit")
        device = self.model.embed_weight().device
        pad = self.tok.pad_id
        n = len(seqs)
        ids = torch.tensor([[pad] * (longest - len(s)) + s for s in seqs], device=device)
        mask = torch.tensor([[0] * (longest - len(s)) + [1] * len(s) for s in seqs], device=device)
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        text = self.model.text_model()
        embed = self.model.embed_weight()
        generated: list[list[int]] = [[] for _ in range(n)]
        done = [False] * n
        with self._adapter_off():
            hidden, cache = prefill_last(text, ids, mask, positions)
            for step in range(self.max_new_tokens):
                logits = torch.nn.functional.linear(hidden.to(embed.dtype), embed)
                tokens = logits.argmax(-1).tolist()
                for j, token in enumerate(tokens):
                    if done[j]:
                        continue
                    if token in self.eos:
                        done[j] = True
                        continue
                    generated[j].append(token)
                    if (
                        self.stop_when is not None
                        and step % self.check_every == self.check_every - 1
                        and self.stop_when(self.tok.decode(generated[j]))
                    ):
                        done[j] = True
                if all(done):
                    break
                step_ids = torch.tensor(
                    [[pad if done[j] else t] for j, t in enumerate(tokens)], device=device
                )
                live = torch.tensor([[0 if d else 1] for d in done], device=device)
                mask = torch.cat([mask, live.to(mask.dtype)], dim=1)
                positions = (mask.sum(-1) - 1).clamp(min=0)[:, None]
                out = text(
                    input_ids=step_ids,
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache, hidden = out.past_key_values, out.last_hidden_state[:, -1, :]
        return [self.tok.decode(g) for g in generated]
