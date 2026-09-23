"""ElectraDecisionModel — the full decision kernel (docs.md section 8/22/38).

    state (encoded once) -> StateMemory
    isolated question branches -> question latents
    dynamic candidates -> candidate latents
    candidate <-> state evidence cross-attention
    permutation-equivariant set mixer
    pointer distribution head -> calibrated distribution

The forward contract is fully ragged (padding-free): every axis is
described by cu_seqlens/index tensors, never by padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import ElectraConfig
from .evidence import EvidenceCrossAttention
from .pointer import PointerHead, ragged_log_softmax, ragged_softmax
from .pool import LatentPool
from .ragged import gather_block_tokens, gather_segments
from .router import BlockRouter, flatten_selection
from .set_mixer import SetMixer
from .state_encoder import StateEncoder, StateMemory
from .text_encoder import TextEncoder

# primitive indices for the temperature buffer / API layer
NOUL, CHOICE, SCORE = 0, 1, 2


@dataclass
class DecisionOutput:
    logits: torch.Tensor  # [n_c] per-candidate pointer logits
    cand_cu: torch.Tensor  # [n_q + 1] candidate-set boundaries
    cand_question_index: torch.Tensor  # [n_c]
    question_repr: torch.Tensor  # [n_q, D] pointer-query source
    candidate_repr: torch.Tensor  # [n_c, D] post-mixer R
    pre_mixer_repr: torch.Tensor  # [n_c, D] pre-mixer r
    route_probs: torch.Tensor  # [n_c, n_blocks]
    memory: StateMemory
    question_latents: torch.Tensor  # [n_q, Mq, D]
    candidate_latents: torch.Tensor  # [n_c, Mc, D]
    evidence_latents: torch.Tensor  # [n_c, Mc, D]

    def log_probs(self) -> torch.Tensor:
        return ragged_log_softmax(self.logits, self.cand_cu)

    def probs(self) -> torch.Tensor:
        return ragged_softmax(self.logits, self.cand_cu)


class ElectraDecisionModel(nn.Module):
    def __init__(self, cfg: ElectraConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden
        self.state_encoder = StateEncoder(cfg)
        self.text_encoder = TextEncoder(cfg, self.state_encoder.embed)
        self.evidence = EvidenceCrossAttention(cfg)
        self.router = BlockRouter(cfg)
        self.set_mixer = SetMixer(cfg)
        self.pointer = PointerHead(cfg)
        self.cand_pool = LatentPool(d, cfg.heads)
        self.q_pool = LatentPool(d, cfg.heads)
        # per-primitive scalar temperature (fitted post-hoc, addendum A8)
        self.register_buffer("temperature", torch.ones(3))

    def _lat_index(self, cand_idx: torch.Tensor, m: int) -> torch.Tensor:
        """Expand candidate indices to their Mc latent offsets."""
        return (
            cand_idx.unsqueeze(1) * m + torch.arange(m, device=cand_idx.device).unsqueeze(0)
        ).reshape(-1)

    def _evidence_for_full(
        self,
        flat: torch.Tensor,
        mem: StateMemory,
        cand_state: torch.Tensor,
        full_idx: torch.Tensor,
        mc: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared-memory path: all of a state's candidates in one segment."""
        st = cand_state[full_idx]
        order = torch.argsort(st, stable=True)
        gidx = full_idx[order]
        needed, inv = torch.unique_consecutive(st[order], return_inverse=True)
        counts = torch.bincount(inv)
        item_cu = torch.zeros(counts.numel() + 1, dtype=mem.state_cu.dtype, device=flat.device)
        item_cu[1:] = torch.cumsum(counts, 0) * mc
        kv, kv_cu = gather_segments(mem.hs, mem.state_cu, needed)
        lat_sel = self._lat_index(gidx, mc)
        out = self.evidence(flat[lat_sel], item_cu, kv, kv_cu)
        return lat_sel, out

    def _evidence_for_routed(
        self,
        flat: torch.Tensor,
        mem: StateMemory,
        route: dict[str, torch.Tensor],
        routed_idx: torch.Tensor,
        mc: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Long-context path: per-candidate top-k block token memory."""
        flat_sel, counts = flatten_selection(
            route["selected"][routed_idx], route["valid"][routed_idx]
        )
        kv, kv_cu = gather_block_tokens(mem.hs, mem.blk_cu, flat_sel, counts)
        n_r = routed_idx.numel()
        item_cu = torch.arange(0, n_r * mc + 1, mc, dtype=kv_cu.dtype, device=flat.device)
        lat_sel = self._lat_index(routed_idx, mc)
        out = self.evidence(flat[lat_sel], item_cu, kv, kv_cu)
        return lat_sel, out

    def forward(
        self,
        state_ids: torch.Tensor,
        state_cu: torch.Tensor,
        question_ids: torch.Tensor,
        question_cu: torch.Tensor,
        question_state_index: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_cu: torch.Tensor,
        candidate_question_index: torch.Tensor,
        primitive_index: torch.Tensor | None = None,
        apply_temperature: bool = True,
    ) -> DecisionOutput:
        mem = self.state_encoder(state_ids, state_cu)
        rq = self.text_encoder(question_ids, question_cu, "q")  # [n_q, Mq, D]
        rc = self.text_encoder(candidate_ids, candidate_cu, "c")  # [n_c, Mc, D]
        n_c, mc, d = rc.shape
        n_q = rq.shape[0]

        counts = torch.bincount(candidate_question_index, minlength=n_q)
        cand_cu = torch.zeros(n_q + 1, dtype=candidate_cu.dtype, device=rc.device)
        cand_cu[1:] = torch.cumsum(counts, 0)

        cand_state = question_state_index[candidate_question_index]
        routed = mem.state_lens[cand_state] > self.cfg.short_context_threshold

        # router runs for every candidate: route_probs feed the router
        # loss even when the token gather only serves long states
        route = self.router(rc.mean(1), mem.bs, cand_state, mem.block_state_index)

        flat = rc.reshape(n_c * mc, d)
        ec_flat = torch.empty_like(flat)

        full_idx = torch.nonzero(~routed).squeeze(1)
        if full_idx.numel() > 0:
            lat_sel, out = self._evidence_for_full(flat, mem, cand_state, full_idx, mc)
            ec_flat[lat_sel] = out

        routed_idx = torch.nonzero(routed).squeeze(1)
        if routed_idx.numel() > 0:
            lat_sel, out = self._evidence_for_routed(flat, mem, route, routed_idx, mc)
            ec_flat[lat_sel] = out

        ec = ec_flat.view(n_c, mc, d)
        r = self.cand_pool(ec)  # [n_c, D]
        r_mixed = self.set_mixer(r, cand_cu)  # [n_c, D]
        dq = self.q_pool(rq)  # [n_q, D]
        logits = self.pointer(dq, r_mixed, candidate_question_index)

        if apply_temperature and primitive_index is not None:
            temp = self.temperature.clamp(min=1e-2)
            logits = logits / temp[primitive_index[candidate_question_index]]

        return DecisionOutput(
            logits=logits,
            cand_cu=cand_cu,
            cand_question_index=candidate_question_index,
            question_repr=dq,
            candidate_repr=r_mixed,
            pre_mixer_repr=r,
            route_probs=route["route_probs"],
            memory=mem,
            question_latents=rq,
            candidate_latents=rc,
            evidence_latents=ec,
        )
