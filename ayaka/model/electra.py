"""ElectraDecisionModel — decision head over a Gemma 4 text backbone.

    prefix (instructions + state)          encoded once per state
      └─ suffix_q (question + options + cue)  isolated branch per question
           ├─ answer-position state h_q  ──► label readout  softcap(h_q · E[label_i])
           └─ option-span states r_i ─┐
                                      Set Mixer (permutation-equivariant)
                                      └─► pointer logits  <W_q h_q, W_k R_i>
    logit_i = label_i + g[primitive, length] · pointer_i (sets with a label alphabet)
    logit_i = pointer_i                              (larger sets: pointer only)

``g`` starts at 0, so an untrained head reproduces the backbone's
zero-shot answer distribution exactly; training moves it only as far as
the pointer helps. Per-primitive temperatures are fitted post-hoc.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
import torch.nn as nn

from ..backbone import embedding_rows, label_logits, load_text_backbone
from ..config import ElectraConfig
from .attention import enable_windowed_attention
from .fastpath import forward_kept, kv_shared_start, prefix_cache
from .heads import PointerHead
from .ragged import ragged_log_softmax

NOUL, CHOICE, SCORE = 0, 1, 2
LENGTH_BUCKETS = ("short", "long")


def length_bucket(seq_len, threshold: int, n: int, device) -> torch.Tensor:
    """0 = short, 1 = long prompt; unknown lengths count as short."""
    if seq_len is None:
        return torch.zeros(n, dtype=torch.long, device=device)
    return (seq_len.to(device) >= threshold).long()


PRIMITIVE_INDEX = {"noul": NOUL, "choice": CHOICE, "score": SCORE}


@dataclass
class DecisionBatch:
    """One row per question. Positions are relative to the encoded rows
    (full prefix+suffix rows in training, suffix-only on the cache path)."""

    input_ids: torch.Tensor  # [B, L]
    attention_mask: torch.Tensor  # [B, L] (or [B, P+L] with a cached prefix)
    answer_pos: torch.Tensor  # [B]
    cand_cu: torch.Tensor  # [B + 1]
    cand_question: torch.Tensor  # [n_c]
    cand_spans: torch.Tensor  # [n_c, 2] [start, end)
    label_ids: torch.Tensor  # [n_c] readout token (0 when pointer-only)
    has_label: torch.Tensor  # [B] bool
    primitive: torch.Tensor  # [B]
    position_ids: torch.Tensor | None = None  # set on the cache path
    seq_len: torch.Tensor | None = None  # [B] full prompt length (prefix + suffix)

    def to(self, device) -> DecisionBatch:
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kw[f.name] = v.to(device) if isinstance(v, torch.Tensor) else v
        return DecisionBatch(**kw)


@dataclass
class DecisionOutput:
    logits: torch.Tensor  # [n_c]
    label_logits: torch.Tensor  # [n_c] (0 where pointer-only)
    pointer_logits: torch.Tensor  # [n_c]; zero when unused pointer branch is skipped in eval
    cand_cu: torch.Tensor
    cand_question: torch.Tensor
    primitive: torch.Tensor  # [n_q]

    def log_probs(self) -> torch.Tensor:
        return ragged_log_softmax(self.logits, self.cand_cu)

    def probs(self) -> torch.Tensor:
        return self.log_probs().exp()

    def pointer_log_probs(self) -> torch.Tensor:
        return ragged_log_softmax(self.pointer_logits, self.cand_cu)


def at_least_fp32(t: torch.Tensor) -> torch.Tensor:
    """bf16/fp16 -> fp32 for the head; fp32/fp64 pass through unchanged."""
    return t.to(torch.promote_types(t.dtype, torch.float32))


def span_means(
    hidden: torch.Tensor, rows: torch.Tensor, spans: torch.Tensor, norm: nn.Module | None = None
) -> torch.Tensor:
    """Gather only option tokens, optionally normalize, then mean each span.

    Work scales with option tokens rather than the entire state/padded rows.
    Local sums also avoid subtracting nearly equal large prefix sums. Empty
    spans produce zero, and repeated/overlapping spans retain their gradients.
    """
    if spans.shape[0] == 0:
        return at_least_fp32(hidden.new_zeros(0, hidden.shape[-1]))
    lengths = (spans[:, 1] - spans[:, 0]).clamp(min=0)
    offsets = torch.nn.functional.pad(lengths.cumsum(0), (1, 0))
    candidates = torch.repeat_interleave(torch.arange(spans.shape[0], device=spans.device), lengths)
    positions = (
        spans[candidates, 0]
        + torch.arange(candidates.numel(), device=spans.device)
        - offsets[candidates]
    )
    selected = hidden[rows[candidates], positions]
    if norm is not None:
        selected = norm(selected)
    selected = at_least_fp32(selected)
    total = torch.segment_reduce(selected, "sum", lengths=lengths)
    return total / lengths.clamp(min=1).unsqueeze(-1).to(total.dtype)


class ElectraDecisionModel(nn.Module):
    def __init__(self, cfg: ElectraConfig, backbone: nn.Module, text_config):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.text_config = text_config
        self.softcap = getattr(text_config, "final_logit_softcapping", None)
        for c in {id(text_config): text_config, id(backbone.config): backbone.config}.values():
            enable_windowed_attention(c)  # exact; skips masked-out key blocks
        self.head = PointerHead(
            text_config.hidden_size, cfg.pointer_dim, cfg.set_mixer_layers, cfg.set_mixer_heads
        )
        # Separate short/long mixing strengths so pointer corrections learned
        # from compositional prompts need not disturb short typed decisions.
        self.gate = nn.Parameter(torch.zeros(3, len(LENGTH_BUCKETS)))
        # scalar temperature per (primitive, prompt-length bucket): long prompts
        # were measurably overconfident with a single per-primitive scalar
        self.register_buffer("temperature", torch.ones(3, len(LENGTH_BUCKETS)))
        # exact speed-up on KV-shared backbones (E2B/E4B); see model/fastpath.py
        # option spans are read below the KV-shared layers (all of them on
        # backbones without KV sharing); the answer position from the top
        self.span_layer = kv_shared_start(text_config)
        self.prune_shared_positions = self.span_layer is not None

    @classmethod
    def from_config(
        cls, cfg: ElectraConfig, dtype: torch.dtype = torch.bfloat16, device="cpu"
    ) -> ElectraDecisionModel:
        backbone, text_config = load_text_backbone(cfg.backbone, dtype=dtype, device=device)
        model = cls(cfg, backbone, text_config)
        model.head.to(device)
        model.gate.data = model.gate.data.to(device)
        model.temperature = model.temperature.to(device)
        return model

    # ------------------------------------------------------------ backbone
    def text_model(self) -> nn.Module:
        """The bare Gemma4TextModel (unwraps a PEFT wrapper if present)."""
        m = self.backbone
        if hasattr(m, "get_base_model"):
            m = m.get_base_model()
        return m

    def embed_weight(self) -> torch.Tensor:
        return self.text_model().embed_tokens.weight

    def label_rows(self, ids: torch.Tensor) -> torch.Tensor:
        return embedding_rows(self.text_model().embed_tokens, ids)

    def encode(
        self, batch: DecisionBatch, past_key_values=None, *, sparse_spans: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, DecisionBatch]:
        """-> (answer states, span states, batch indexed into them).

        On KV-shared backbones (E2B/E4B) only the answer positions run
        through the shared layers and the returned batch points answer_pos
        into that compact [B, 1, D] view; option spans index the states
        below the shared layers. Both paths give identical results
        (tests/test_model.py).
        sparse_spans=True returns raw states below the shared layers;
        forward normalizes only gathered option tokens in that case.
        """
        common = {
            "attention_mask": batch.attention_mask,
            "position_ids": batch.position_ids,
            "past_key_values": past_key_values,
        }
        if self.prune_shared_positions:
            keep = batch.answer_pos.unsqueeze(1)
            top, spans_h = forward_kept(
                self.text_model(), batch.input_ids, keep, normalize_spans=not sparse_spans, **common
            )
            return top, spans_h, replace(batch, answer_pos=torch.zeros_like(batch.answer_pos))
        out = self.backbone(
            input_ids=batch.input_ids,
            use_cache=past_key_values is not None,
            output_hidden_states=self.span_layer is not None,
            **common,
        )
        top = out.last_hidden_state
        if self.span_layer is None:
            return top, top, batch
        spans_h = out.hidden_states[self.span_layer]
        return top, spans_h if sparse_spans else self.text_model().norm(spans_h), batch

    def encode_prefix(self, prefix_ids: torch.Tensor, cache=None):
        """Run the shared prefix once; returns its KV cache (on KV-shared
        backbones the shared layers are skipped for the prefix). ``cache``
        continues an already-encoded head."""
        if self.prune_shared_positions:
            return prefix_cache(self.text_model(), prefix_ids, cache)
        out = self.backbone(input_ids=prefix_ids, past_key_values=cache, use_cache=True)
        return out.past_key_values

    # ---------------------------------------------------------------- head
    def decide(
        self,
        hidden: torch.Tensor,
        batch: DecisionBatch,
        apply_temperature: bool = False,
        span_hidden: torch.Tensor | None = None,
        span_norm: nn.Module | None = None,
    ) -> DecisionOutput:
        n_q = batch.answer_pos.numel()
        q = at_least_fp32(hidden[torch.arange(n_q, device=hidden.device), batch.answer_pos])
        cq = batch.cand_question
        has_label_c = batch.has_label[cq]
        prim_c = batch.primitive[cq]
        bucket = length_bucket(batch.seq_len, self.cfg.long_prompt_tokens, n_q, q.device)
        gate = self.gate[prim_c, bucket[cq]]
        lab = label_logits(q[cq], self.label_rows(batch.label_ids), self.softcap)
        lab = torch.where(has_label_c, lab, torch.zeros_like(lab))
        # Training must retain pointer auxiliary losses and gate gradients,
        # even at initialization. In no-grad eval, an all-zero active gate
        # makes this entire branch irrelevant to the returned distribution.
        skip_pointer = (
            not self.training
            and not torch.is_grad_enabled()
            and bool((has_label_c & (gate == 0)).all())
        )
        if skip_pointer:
            ptr = torch.zeros_like(lab)
        else:
            spans_h = hidden if span_hidden is None else span_hidden
            r = span_means(spans_h, cq, batch.cand_spans, norm=span_norm)
            ptr = at_least_fp32(self.head(r, q, batch.cand_cu, cq))
        logits = torch.where(has_label_c, lab + gate * ptr, ptr)
        if apply_temperature:
            logits = logits / self.temperature[prim_c, bucket[cq]]
        return DecisionOutput(logits, lab, ptr, batch.cand_cu, cq, batch.primitive)

    def forward(
        self, batch: DecisionBatch, apply_temperature: bool = False, past_key_values=None
    ) -> DecisionOutput:
        top, spans_h, view = self.encode(batch, past_key_values, sparse_spans=True)
        norm = self.text_model().norm if self.span_layer is not None else None
        return self.decide(top, view, apply_temperature, span_hidden=spans_h, span_norm=norm)

    # ------------------------------------------------------- checkpointing
    def head_state_dict(self) -> dict:
        """Everything that is not frozen backbone weight: pointer head,
        gate, temperatures (LoRA adapters are saved separately)."""
        return {
            "head": self.head.state_dict(),
            "gate": self.gate.detach().cpu(),
            "temperature": self.temperature.detach().cpu(),
        }

    def load_head_state_dict(self, sd: dict) -> None:
        self.head.load_state_dict(sd["head"])
        with torch.no_grad():
            g = sd["gate"]
            if g.dim() == 1:  # legacy per-primitive gates preserve both length regimes
                g = g.unsqueeze(1).expand_as(self.gate)
            self.gate.copy_(g)
            t = sd["temperature"]
            if t.dim() == 1:  # checkpoints from before length buckets
                t = t.unsqueeze(1).expand_as(self.temperature)
            self.temperature.copy_(t)
