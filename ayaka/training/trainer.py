"""Token-budget trainer (docs.md section 45).

BF16 compute + FP32 master params/optimizer state, fused AdamW,
cosine decay to a floor, global-norm clip 1.0, zero_grad(set_to_none),
optional torch.compile of the inner model, optional FSDP2 wrapping.
Padding-free batches arrive pre-packed from data.packing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from ..losses import DecisionLossWeights, decision_loss
from ..metrics import compute_metrics
from ..model.model import ElectraDecisionModel
from .schedule import build_optimizer, cosine_warmup_schedule


@dataclass
class TrainConfig:
    lr: float = 3e-5
    steps: int = 1000
    warmup_frac: float = 0.02
    min_lr_frac: float = 0.1
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    bf16: bool = True
    compile: bool = False
    fused: bool = True
    missing_tau: float = 0.8
    log_every: int = 20
    eval_every: int = 0
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    loss_weights: DecisionLossWeights = field(default_factory=DecisionLossWeights)


class Trainer:
    def __init__(self, model: ElectraDecisionModel, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        self.opt = build_optimizer(
            model,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.eps,
            fused=cfg.fused,
        )
        self.sched = cosine_warmup_schedule(self.opt, cfg.steps, cfg.warmup_frac, cfg.min_lr_frac)
        self._forward = model
        if cfg.compile:
            self._forward = torch.compile(model, backend="inductor")
        self.step_i = 0
        self.history: list[dict] = []

    def _autocast(self):
        if self.cfg.bf16 and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast(self.device.type, enabled=False)

    def step(self, batch) -> dict[str, float]:
        cfg = self.cfg
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        with self._autocast():
            out = self._forward(**batch.inputs)
            parts = decision_loss(
                out,
                batch.targets.float(),
                cand_ordinals=batch.cand_ordinals,
                score_question_mask=batch.score_question_mask,
                missing_mask=batch.missing_mask,
                block_target=batch.block_target,
                block_mask=batch.block_mask,
                weights=cfg.loss_weights,
                missing_tau=cfg.missing_tau,
            )
        parts["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.opt.step()
        self.sched.step()
        self.step_i += 1
        rec = {
            "step": self.step_i,
            "loss": float(parts["total"].detach()),
            "grad_norm": float(grad_norm),
            "lr": self.sched.get_last_lr()[0],
        }
        for k, v in parts.items():
            if k != "total":
                rec[f"loss_{k}"] = float(v)
        self.history.append(rec)
        return rec

    @torch.no_grad()
    def evaluate(self, batches) -> dict[str, float]:
        self.model.eval()
        agg: dict[str, list[float]] = {}
        n = 0
        for batch in batches:
            out = self._forward(**batch.inputs)
            m = compute_metrics(out, batch.targets.float(), batch.missing_mask)
            for k, v in m.items():
                agg.setdefault(k, []).append(v * batch.n_questions)
            n += batch.n_questions
        return {k: sum(v) / max(n, 1) for k, v in agg.items()}

    def train(self, train_batches, eval_batches=None) -> list[dict]:
        """train_batches: iterable (re-iterable) of packed TrainBatch —
        sampled fresh per step for infinite-stream semantics."""
        it = iter(train_batches)
        while self.step_i < self.cfg.steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_batches)
                break
            t0 = time.time()
            rec = self.step(batch)
            rec["sec"] = time.time() - t0
            if self.cfg.log_every and self.step_i % self.cfg.log_every == 0:
                pass  # caller reads history; no printing inside trainer
            if eval_batches and self.cfg.eval_every and self.step_i % self.cfg.eval_every == 0:
                rec.update({f"eval_{k}": v for k, v in self.evaluate(eval_batches).items()})
        return self.history

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict(),
            "step": self.step_i,
            "config": self.cfg,
            "temperature": self.model.temperature.tolist(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.model.load_state_dict(sd["model"])
        self.opt.load_state_dict(sd["optimizer"])
        self.sched.load_state_dict(sd["scheduler"])
        self.step_i = sd["step"]
