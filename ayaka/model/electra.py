"""ElectraDecisionModel — decision head over a Gemma 4 text backbone.

    prefix (instructions + state)          encoded once per state
      └─ suffix_q (question + options + cue)  isolated branch per question
           ├─ answer-position state h_q  ──► label readout  softcap(h_q · E[label_i])
           └─ option-span states r_i ─┐
                                      Set Mixer (permutation-equivariant)
                                      └─► pointer logits  <W_q h_q, W_k R_i>
    logit_i = label_i + g[primitive] · pointer_i     (sets with a label alphabet)
    logit_i = pointer_i                              (larger sets: pointer only)

``g`` starts at 0, so an untrained head reproduces the backbone's
zero-shot answer distribution exactly; training moves it only as far as
the pointer helps. Per-primitive temperatures are fitted post-hoc.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn

from ..backbone import embedding_rows, label_logits, load_text_backbone
from ..config import ElectraConfig
from .heads import PointerHead
from .ragged import ragged_log_softmax

NOUL, CHOICE, SCORE = 0, 1, 2
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
    pointer_logits: torch.Tensor  # [n_c]
    cand_cu: torch.Tensor
    cand_question: torch.Tensor
    primitive: torch.Tensor  # [n_q]

    def log_probs(self) -> torch.Tensor:
        return ragged_log_softmax(self.logits, self.cand_cu)

    def probs(self) -> torch.Tensor:
        return self.log_probs().exp()

    def pointer_log_probs(self) -> torch.Tensor:
        return ragged_log_softmax(self.pointer_logits, self.cand_cu)


def span_means(hidden: torch.Tensor, rows: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
    """Mean hidden state over [start, end) per candidate (fp32)."""
    cs = torch.nn.functional.pad(hidden.float().cumsum(1), (0, 0, 1, 0))  # [B, L+1, D]
    s, e = spans[:, 0], spans[:, 1]
    total = cs[rows, e] - cs[rows, s]
    return total / (e - s).clamp(min=1).unsqueeze(-1).to(total.dtype)


class ElectraDecisionModel(nn.Module):
    def __init__(self, cfg: ElectraConfig, backbone: nn.Module, text_config):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.text_config = text_config
        self.softcap = getattr(text_config, "final_logit_softcapping", None)
        self.head = PointerHead(
            text_config.hidden_size, cfg.pointer_dim, cfg.set_mixer_layers, cfg.set_mixer_heads
        )
        self.gate = nn.Parameter(torch.zeros(3))
        self.register_buffer("temperature", torch.ones(3))

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

    def encode(self, batch: DecisionBatch, past_key_values=None) -> torch.Tensor:
        out = self.backbone(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
        )
        return out.last_hidden_state

    def encode_prefix(self, prefix_ids: torch.Tensor):
        """Run the shared prefix once; returns its KV cache."""
        out = self.backbone(input_ids=prefix_ids, use_cache=True)
        return out.past_key_values

    # ---------------------------------------------------------------- head
    def decide(
        self, hidden: torch.Tensor, batch: DecisionBatch, apply_temperature: bool = False
    ) -> DecisionOutput:
        n_q = batch.answer_pos.numel()
        q = hidden[torch.arange(n_q, device=hidden.device), batch.answer_pos].float()
        r = span_means(hidden, batch.cand_question, batch.cand_spans)
        cq = batch.cand_question
        has_label_c = batch.has_label[cq]
        lab = label_logits(q[cq], self.label_rows(batch.label_ids), self.softcap)
        lab = torch.where(has_label_c, lab, torch.zeros_like(lab))
        ptr = self.head(r, q, batch.cand_cu, cq).float()
        prim_c = batch.primitive[cq]
        logits = torch.where(has_label_c, lab + self.gate[prim_c] * ptr, ptr)
        if apply_temperature:
            logits = logits / self.temperature[prim_c]
        return DecisionOutput(logits, lab, ptr, batch.cand_cu, cq, batch.primitive)

    def forward(self, batch: DecisionBatch, apply_temperature: bool = False) -> DecisionOutput:
        return self.decide(self.encode(batch), batch, apply_temperature)

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
            self.gate.copy_(sd["gate"])
            self.temperature.copy_(sd["temperature"])
