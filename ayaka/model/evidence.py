"""Opt-in, text-feature evidence readout; not a production backbone adapter.

The caller supplies native output-head logits and contextual features. Memory
must contain only shared record evidence, or have explicit question scopes.
The last correction projection starts at zero, preserving the native prior.
Set equivariance here does not imply invariance of a causal backbone prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class EvidenceOutput:
    logits: torch.Tensor  # [record, question, candidate], padding is -inf
    correction: torch.Tensor
    candidate_mask: torch.Tensor
    question_mask: torch.Tensor

    def log_probs(self) -> torch.Tensor:
        # A fully padded question has no distribution. Avoid softmax(-inf,...).
        dtype = torch.float64 if self.logits.dtype == torch.float64 else torch.float32
        safe = torch.where(self.question_mask[..., None], self.logits, 0).to(dtype)
        return F.log_softmax(safe, dim=-1).masked_fill(~self.candidate_mask, -torch.inf)

    def probs(self) -> torch.Tensor:
        return self.log_probs().exp()


class _AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x: torch.Tensor, memory: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        s = memory.shape[1]
        q = self.q(self.query_norm(x)).view(b, n, self.heads, d // self.heads)
        k, v = (
            self.kv(self.memory_norm(memory)).view(b, s, 2, self.heads, d // self.heads).unbind(2)
        )
        routed = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=allowed[:, None],
            dropout_p=self.dropout if self.training else 0.0,
        )
        x = x + self.out(routed.transpose(1, 2).reshape(b, n, d))
        return x + self.ffn(self.ffn_norm(x))


class EvidenceResidualHead(nn.Module):
    """Small, zero-init residual over actual native candidate logits.

    ``group_ids=None`` isolates questions. Equal nonnegative group ids explicitly
    permit field mixing within one record, never across records. Optional lexical
    features must come from the actual output head; input embeddings are not an
    assumed substitute. ``memory_scope`` can isolate each question's trace bank.
    Caps fail explicitly, without truncation or an additional backbone forward.
    The default detaches native logits for head-only training; backbone training
    requires an explicit ``detach_prior=False`` and a separately frozen reference.
    """

    def __init__(
        self,
        hidden: int,
        dim: int = 128,
        heads: int = 4,
        routing_layers: int = 1,
        field_layers: int = 1,
        *,
        lexical: bool = False,
        dropout: float = 0.0,
        detach_prior: bool = True,
        max_memory_tokens: int = 8192,
        max_questions: int = 64,
        max_candidates: int = 128,
        max_total_candidates: int = 2048,
    ):
        super().__init__()
        integers = (
            hidden,
            dim,
            heads,
            max_memory_tokens,
            max_questions,
            max_candidates,
            max_total_candidates,
        )
        if any(type(v) is not int or v <= 0 for v in integers):
            raise ValueError("dimensions and caps must be positive integers")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if any(type(v) is not int or v < 0 for v in (routing_layers, field_layers)):
            raise ValueError("layer counts must be nonnegative integers")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if type(lexical) is not bool or type(detach_prior) is not bool:
            raise ValueError("lexical and detach_prior must be booleans")
        self.hidden = hidden
        self.dim = dim
        self.detach_prior = detach_prior
        self.max_memory_tokens = max_memory_tokens
        self.max_questions = max_questions
        self.max_candidates = max_candidates
        self.max_total_candidates = max_total_candidates
        self.memory_projection = nn.Linear(hidden, dim, bias=False)
        self.candidate_projection = nn.Linear(hidden, dim, bias=False)
        self.query_projection = nn.Linear(hidden, dim, bias=False)
        self.lexical_projection = nn.Linear(hidden, dim, bias=False) if lexical else None
        self.primitive_embedding = nn.Embedding(3, dim)
        self.routing = nn.ModuleList(
            _AttentionBlock(dim, heads, dropout) for _ in range(routing_layers)
        )
        self.fields = nn.ModuleList(
            _AttentionBlock(dim, heads, dropout) for _ in range(field_layers)
        )
        self.correction = nn.Sequential(
            nn.LayerNorm(4 * dim), nn.Linear(4 * dim, dim), nn.GELU(), nn.Linear(dim, 1, bias=False)
        )
        nn.init.zeros_(self.correction[-1].weight)

    def forward(
        self,
        native_logits: torch.Tensor,
        candidates: torch.Tensor,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        question_mask: torch.Tensor,
        primitive: torch.Tensor,
        *,
        group_ids: torch.Tensor | None = None,
        lexical: torch.Tensor | None = None,
        memory_scope: torch.Tensor | None = None,
    ) -> EvidenceOutput:
        if native_logits.ndim != 3 or not native_logits.is_floating_point():
            raise ValueError("native_logits must be floating [record, question, candidate]")
        b, q, k = native_logits.shape
        if min(b, q, k) <= 0 or q > self.max_questions or k > self.max_candidates:
            raise ValueError("question/candidate dimensions exceed caps or are empty")
        if b * q * k > self.max_total_candidates:
            raise ValueError("total padded candidates exceed cap")
        if memory.ndim != 3 or memory.shape[0] != b or memory.shape[2] != self.hidden:
            raise ValueError("memory must be [record, token, hidden]")
        s = memory.shape[1]
        if not 0 < s <= self.max_memory_tokens:
            raise ValueError("memory length exceeds cap or is empty")
        expected = {
            "candidates": (candidates, (b, q, k, self.hidden)),
            "queries": (queries, (b, q, self.hidden)),
            "memory_mask": (memory_mask, (b, s)),
            "candidate_mask": (candidate_mask, (b, q, k)),
            "question_mask": (question_mask, (b, q)),
            "primitive": (primitive, (b, q)),
        }
        if group_ids is not None:
            expected["group_ids"] = (group_ids, (b, q))
        if memory_scope is not None:
            expected["memory_scope"] = (memory_scope, (b, q, s))
        if self.lexical_projection is not None:
            if lexical is None:
                raise ValueError("lexical head requires actual output-head features")
            expected["lexical"] = (lexical, (b, q, k, self.hidden))
        elif lexical is not None:
            raise ValueError("lexical features require lexical=True")
        for name, (tensor, shape) in expected.items():
            if tensor.shape != shape or tensor.device != native_logits.device:
                raise ValueError(f"{name} shape/device mismatch")
        if (
            memory.device != native_logits.device
            or self.candidate_projection.weight.device != memory.device
        ):
            raise ValueError("head/features must share a device")
        masks = [memory_mask, candidate_mask, question_mask]
        if memory_scope is not None:
            masks.append(memory_scope)
        if any(t.dtype != torch.bool for t in masks):
            raise ValueError("masks must be boolean")
        if not torch.equal(candidate_mask.any(-1), question_mask):
            raise ValueError("active questions must have candidates; padding must have none")
        if (
            primitive.dtype not in (torch.int32, torch.int64)
            or ((primitive[question_mask] < 0) | (primitive[question_mask] > 2)).any()
        ):
            raise ValueError("primitive must contain integer types 0, 1, 2")
        if group_ids is not None and (
            group_ids.dtype not in (torch.int32, torch.int64)
            or (group_ids[question_mask] < 0).any()
        ):
            raise ValueError("active group_ids must be nonnegative integers")
        scope = memory_mask[:, None].expand(b, q, s)
        if memory_scope is not None:
            scope = scope & memory_scope
        if (question_mask & ~scope.any(-1)).any():
            raise ValueError("active question has no allowed memory")
        features = [(candidates, candidate_mask), (queries, question_mask), (memory, memory_mask)]
        if lexical is not None:
            features.append((lexical, candidate_mask))
        for tensor, valid in features:
            if not tensor.is_floating_point() or not torch.isfinite(tensor[valid]).all():
                raise ValueError("valid features must be finite floating tensors")
        if not torch.isfinite(native_logits[candidate_mask]).all():
            raise ValueError("valid native logits must be finite")

        dtype = self.candidate_projection.weight.dtype

        def clean(tensor: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            cleaned = torch.where(valid[..., None], tensor, 0).to(dtype)
            if not torch.isfinite(cleaned).all():
                raise ValueError("features overflow head dtype")
            return cleaned

        mem = self.memory_projection(clean(memory, memory_mask))
        query = self.query_projection(clean(queries, question_mask))
        types = self.primitive_embedding(primitive.masked_fill(~question_mask, 0).long())
        query = (query + types).masked_fill(~question_mask[..., None], 0)
        x = self.candidate_projection(clean(candidates, candidate_mask)) + query[:, :, None]
        if lexical is not None:
            x = x + self.lexical_projection(clean(lexical, candidate_mask))
        allowed = (
            (scope[:, :, None] & candidate_mask[..., None]).expand(b, q, k, s).reshape(b, q * k, s)
        )
        x = x.reshape(b, q * k, self.dim)
        for block in self.routing:
            x = block(x, mem, allowed)
        x = x.reshape(b, q, k, self.dim).masked_fill(~candidate_mask[..., None], 0)
        field = x.sum(2) / candidate_mask.sum(-1, keepdim=True).clamp_min(1)
        field = field + query
        groups = (
            torch.arange(q, device=memory.device)[None].expand(b, q)
            if group_ids is None
            else group_ids
        )
        field_allowed = (
            (groups[:, :, None] == groups[:, None, :])
            & question_mask[:, :, None]
            & question_mask[:, None, :]
        )
        for block in self.fields:
            field = block(field, field, field_allowed).masked_fill(~question_mask[..., None], 0)
        field = field[:, :, None].expand_as(x)
        score_features = torch.cat((x, field, x * field, (x - field).abs()), -1)
        delta = self.correction(score_features).squeeze(-1).to(native_logits.dtype)
        delta = delta.masked_fill(~candidate_mask, 0)
        prior = native_logits.detach() if self.detach_prior else native_logits
        logits = (prior.masked_fill(~candidate_mask, 0) + delta).masked_fill(
            ~candidate_mask, -torch.inf
        )
        if not torch.isfinite(logits[candidate_mask]).all():
            raise ValueError("evidence correction produced nonfinite logits")
        return EvidenceOutput(logits, delta, candidate_mask, question_mask)
