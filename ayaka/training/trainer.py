"""Decision trainer for the LoRA-adapted Gemma 4 backbone.

Each optimizer step consumes a fixed number of questions, split into
token-budgeted micro-batches with gradient accumulation (loss weighted
by questions per micro-batch). BF16 autocast on CUDA, activation
checkpointing on the backbone, AdamW with separate LR for LoRA weights
and the freshly initialized decision head.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from ..losses import LossWeights, decision_loss
from ..metrics import compute_metrics
from ..model.electra import ElectraDecisionModel
from ..model.ragged import ragged_log_softmax
from ..tokenization import Tokenizer
from .batching import TrainItem, budget_batches, collate_items, plan_chunks
from .reasoning import trace_ce
from .schedule import cosine_warmup_schedule


@dataclass
class TrainConfig:
    steps: int = 1000
    questions_per_step: int = 64
    micro_batch_tokens: int = 8_192  # halves automatically on CUDA OOM
    lr: float = 1e-4  # LoRA weights
    head_lr: float = 5e-4  # pointer head + gate
    warmup_frac: float = 0.03
    min_lr_frac: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    bf16: bool = True
    grad_checkpointing: bool = False  # off: ~30% faster; OOM backoff turns it on
    min_micro_batch_tokens: int = 1_024
    # When OOM forces activation checkpointing, apply it only to forward chunks
    # holding a question at least this long (0 = every chunk, the old global
    # mode). Short chunks keep full speed and shared-prefix encoding.
    selective_checkpoint_tokens: int = 1_024
    missing_tau: float = 0.8
    log_every: int = 20
    eval_every: int = 0
    seed: int = 0
    # wall-clock budget for the optimizer loop (0 = none). The LR schedule still
    # follows `steps`, so size `steps` to fit; this is a billing guard.
    max_train_seconds: float = 0.0
    loss_weights: LossWeights = field(default_factory=LossWeights)
    reasoning_ce_weight: float = 0.3
    proposal_ce_weight: float = 0.2
    ce_chunk_tokens: int = 128
    image_batch_rows: int = 4
    image_feature_cache_bytes: int = 128 * 1024 * 1024


class Trainer:
    def __init__(
        self,
        model: ElectraDecisionModel,
        tok: Tokenizer,
        cfg: TrainConfig,
        device,
        image_backend=None,
    ):
        self.model = model
        self.tok = tok
        self.cfg = cfg
        self.device = torch.device(device)
        self.image_backend = image_backend
        self.image_features = None
        if cfg.image_batch_rows < 1 or cfg.image_feature_cache_bytes < 0:
            raise ValueError("image batch rows must be positive and cache bytes nonnegative")
        if image_backend is not None:
            if image_backend.model is not model:
                raise ValueError("training and image backend must share the exact decision model")
            from .multimodal import freeze_image_components

            freeze_image_components(image_backend)
            if cfg.image_feature_cache_bytes:
                from .image_features import FrozenImageFeatures

                self.image_features = FrozenImageFeatures(
                    image_backend, cfg.image_feature_cache_bytes
                )
        torch.manual_seed(cfg.seed)
        self.micro_tokens = cfg.micro_batch_tokens  # chunks without checkpointing
        self.micro_ckpt_tokens = cfg.micro_batch_tokens  # checkpointed chunks
        # None: never checkpoint; N: checkpoint chunks with a question >= N tokens
        self.ckpt_threshold: int | None = 0 if cfg.grad_checkpointing else None
        self._ckpt_active = False
        self._chunk_ckpt = False  # kind of the chunk that is running (OOM backoff)
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = list(model.head.parameters()) + [model.gate]
        self.opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
                {"params": head_params, "lr": cfg.head_lr, "weight_decay": 0.0},
            ],
            betas=(0.9, 0.98),
            fused=self.device.type == "cuda",
        )
        self.sched = cosine_warmup_schedule(self.opt, cfg.steps, cfg.warmup_frac, cfg.min_lr_frac)
        self.step_i = 0
        self.stopped_early = False
        self.use_amp = cfg.bf16 and self.device.type == "cuda"
        self.can_share = "linear_attention" not in getattr(model.text_config, "layer_types", [])

    def n_trainable(self) -> int:
        return sum(p.numel() for g in self.opt.param_groups for p in g["params"])

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp)

    @property
    def checkpointing(self) -> bool:
        return self.ckpt_threshold is not None

    def _set_checkpointing(self, on: bool) -> None:
        """Toggle backbone activation checkpointing for the next forward."""
        backbone = self.model.backbone
        if on == self._ckpt_active or not hasattr(backbone, "gradient_checkpointing_enable"):
            return
        if on:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            backbone.gradient_checkpointing_disable()
        self._ckpt_active = on

    def _plan(self, items: list[TrainItem]) -> list[tuple[str, list[TrainItem], bool]]:
        """Forward chunks as (kind, items, checkpointed)."""
        images = [it for it in items if it.native_inputs is not None]
        items = [it for it in items if it.native_inputs is None]
        thr = self.ckpt_threshold
        plain = items if thr is None else [it for it in items if it.length < thr]
        heavy = [] if thr is None else [it for it in items if it.length >= thr]
        chunks = [
            (k, mb, False) for k, mb in plan_chunks(plain, self.micro_tokens, share=self.can_share)
        ]
        # activation checkpointing drops KV caches inside HF layers: no sharing then
        chunks += [
            (k, mb, True) for k, mb in plan_chunks(heavy, self.micro_ckpt_tokens, share=False)
        ]
        for checkpointed in (False, True):
            budget = self.micro_ckpt_tokens if checkpointed else self.micro_tokens
            # RPS and missing-evidence penalties average their eligible subset.
            # Homogeneous strata preserve the previous singleton loss weights.
            strata = {}
            for item in images:
                if (thr is not None and item.length >= thr) == checkpointed:
                    strata.setdefault((item.type, item.flagged), []).append(item)
            for group in strata.values():
                chunks += [
                    ("image", rows, checkpointed)
                    for rows in budget_batches(
                        group, budget, max_rows=self.cfg.image_batch_rows, shuffle_seed=None
                    )
                ]
        return chunks

    def train_step(self, items: list[TrainItem]) -> dict:
        """One optimizer step, retried on CUDA OOM so one config fits any GPU.

        An OOM in an uncheckpointed chunk first turns on checkpointing for
        long chunks only (``selective_checkpoint_tokens``), then halves the
        uncheckpointed micro-batch, then checkpoints shorter questions too.
        An OOM in a checkpointed chunk halves that chunk kind's micro-batch.
        """
        floor = self.cfg.min_micro_batch_tokens
        while True:
            try:
                return self._train_step(items)
            except torch.cuda.OutOfMemoryError:
                self.opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if self._chunk_ckpt:
                    if self.micro_ckpt_tokens // 2 < floor:
                        raise
                    self.micro_ckpt_tokens //= 2
                elif self.ckpt_threshold is None:
                    self.ckpt_threshold = self.cfg.selective_checkpoint_tokens
                elif self.micro_tokens // 2 >= floor:
                    self.micro_tokens //= 2
                elif self.ckpt_threshold > floor:
                    self.ckpt_threshold //= 2
                elif self.ckpt_threshold > 0:
                    self.ckpt_threshold = 0
                else:
                    raise
                print(
                    f"[train] CUDA OOM -> micro_batch_tokens={self.micro_tokens} "
                    f"checkpointed>={self.ckpt_threshold} tokens "
                    f"(micro {self.micro_ckpt_tokens})",
                    flush=True,
                )

    def _forward(self, kind: str, mb: list[TrainItem], apply_temperature: bool = False):
        """One planned chunk -> (DecisionOutput, TrainTensors)."""
        out, tensors = self._decision_forward(kind, mb, apply_temperature)
        if any(it.proposal_labels for it in mb):
            from .candidates import proposal_ce

            with self._autocast():
                tensors.proposal_ce = proposal_ce(
                    self.model, mb, self.tok.pad_id, self.cfg.ce_chunk_tokens
                )
        return out, tensors

    def _decision_forward(self, kind, mb, apply_temperature=False):
        if any(it.native_inputs is not None for it in mb):
            if self.image_backend is None or kind != "image":
                raise ValueError(
                    "native image training requires an image backend and independent rows"
                )
            from .multimodal import image_forward

            t = collate_items(mb, self.tok.pad_id).to(self.device)
            with self._autocast():
                hidden, spans, norm = image_forward(
                    self.image_backend, mb, t.batch, self.image_features
                )
                out = self.model.decide(
                    hidden, t.batch, apply_temperature, span_hidden=spans, span_norm=norm
                )
                t.reasoning_ce = trace_ce(self.model, hidden, mb, self.cfg.ce_chunk_tokens)
            return out, t
        if any(it.reasoning_labels for it in mb):
            t = collate_items(mb, self.tok.pad_id).to(self.device)
            with self._autocast():
                native = self.model.backbone(
                    input_ids=t.batch.input_ids,
                    attention_mask=t.batch.attention_mask,
                    use_cache=False,
                    output_hidden_states=self.model.span_layer is not None,
                )
                hidden = native.last_hidden_state
                spans = (
                    native.hidden_states[self.model.span_layer]
                    if self.model.span_layer is not None
                    else hidden
                )
                norm = self.model.text_model().norm if self.model.span_layer is not None else None
                out = self.model.decide(
                    hidden, t.batch, apply_temperature, span_hidden=spans, span_norm=norm
                )
                t.reasoning_ce = trace_ce(self.model, hidden, mb, self.cfg.ce_chunk_tokens)
            return out, t
        if kind == "shared":
            prefix = mb[0].enc.prefix_ids
            t = collate_items(mb, self.tok.pad_id, prefix_len=len(prefix)).to(self.device)
            with self._autocast():
                cache = self.model.encode_prefix(torch.tensor([prefix], device=self.device))
                cache.batch_repeat_interleave(len(mb))
                out = self.model(
                    t.batch, apply_temperature=apply_temperature, past_key_values=cache
                )
            return out, t
        t = collate_items(mb, self.tok.pad_id).to(self.device)
        with self._autocast():
            out = self.model(t.batch, apply_temperature=apply_temperature)
        return out, t

    def _train_step(self, items: list[TrainItem]) -> dict:
        self.model.train()
        n_q = len(items)
        agg: dict[str, torch.Tensor] = {}
        for kind, mb, ckpt in self._plan(items):
            self._chunk_ckpt = ckpt
            self._set_checkpointing(ckpt)
            out, t = self._forward(kind, mb)
            parts = decision_loss(
                out,
                t.targets,
                ordinals=t.ordinals,
                missing_mask=t.flagged,
                teacher_probs=t.teacher,
                teacher_mask=t.teacher_mask,
                weights=self.cfg.loss_weights,
                missing_tau=self.cfg.missing_tau,
            )
            if hasattr(t, "reasoning_ce"):
                parts["reasoning_ce"] = t.reasoning_ce
                parts["total"] = parts["total"] + self.cfg.reasoning_ce_weight * t.reasoning_ce
            if hasattr(t, "proposal_ce"):
                parts["proposal_ce"] = t.proposal_ce
                parts["total"] = parts["total"] + self.cfg.proposal_ce_weight * t.proposal_ce
            frac = len(mb) / n_q
            (parts["total"] * frac).backward()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v.detach() * frac
        params = [p for g in self.opt.param_groups for p in g["params"]]
        gn = torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
        self.opt.step()
        self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        self.step_i += 1
        # One host transfer per step instead of a CUDA synchronization for every
        # loss component in every micro-batch. Public history remains floats.
        values = torch.stack(list(agg.values())).float().tolist()
        agg = dict(zip(agg, values, strict=True))
        agg.update(
            step=self.step_i,
            grad_norm=float(gn),
            lr=self.sched.get_last_lr()[0],
            gate=self.model.gate.detach().tolist(),
            micro_tokens=self.micro_tokens,
            checkpoint_threshold=self.ckpt_threshold,
            family_questions=dict(Counter(it.family for it in items)),
            source_questions=dict(Counter(it.source for it in items)),
            long_questions=sum(it.length >= self.model.cfg.long_prompt_tokens for it in items),
        )
        return agg

    def train(
        self,
        stream: Iterator[list[TrainItem]],
        eval_items: list[TrainItem] | None = None,
        on_step=None,
        verbose: bool = True,
    ) -> list[dict]:
        history = []
        t0 = time.time()
        while self.step_i < self.cfg.steps:
            rec = self.train_step(next(stream))
            rec["elapsed"] = time.time() - t0
            if eval_items and self.cfg.eval_every and self.step_i % self.cfg.eval_every == 0:
                rec.update({f"eval_{k}": v for k, v in self.evaluate(eval_items).items()})
            history.append(rec)
            if (
                verbose
                and self.cfg.log_every
                and (self.step_i % self.cfg.log_every == 0 or self.step_i == 1)
            ):
                ev = " ".join(
                    f"{k}={v:.4f}"
                    for k, v in rec.items()
                    if k.startswith("eval_") and isinstance(v, float)
                )
                print(
                    f"[train] step {self.step_i}/{self.cfg.steps} loss={rec['total']:.4f} nll={rec['nll']:.4f} "
                    f"brier={rec['brier']:.4f} gn={rec['grad_norm']:.2f} lr={rec['lr']:.2e} "
                    f"gate={[[round(g, 3) for g in row] for row in rec['gate']]} {rec['elapsed']:.0f}s {ev}",
                    flush=True,
                )
            if on_step is not None:
                on_step(self.step_i, rec)
            if self.cfg.max_train_seconds and rec["elapsed"] >= self.cfg.max_train_seconds:
                self.stopped_early = True
                if verbose:
                    print(
                        f"[train] time budget {self.cfg.max_train_seconds:.0f}s reached at step "
                        f"{self.step_i}/{self.cfg.steps}; stopping",
                        flush=True,
                    )
                break
        return history

    @torch.no_grad()
    def predict(
        self, items: list[TrainItem], apply_temperature: bool = True, return_logits: bool = False
    ):
        """Per-question probability vectors (input candidate order)."""
        self.model.eval()
        probs: list[list[float] | None] = [None] * len(items)
        logits: list[list[float] | None] = [None] * len(items)
        index = {id(it): i for i, it in enumerate(items)}
        for kind, mb, _ in self._plan(items):
            out, _ = self._forward(kind, mb, apply_temperature=apply_temperature)
            lp = ragged_log_softmax(out.logits.float(), out.cand_cu).exp().tolist()
            lg = out.logits.float().tolist()
            cu = out.cand_cu.tolist()
            for j, it in enumerate(mb):
                probs[index[id(it)]] = lp[cu[j] : cu[j + 1]]
                logits[index[id(it)]] = lg[cu[j] : cu[j + 1]]
        return (probs, logits) if return_logits else probs

    def evaluate(self, items: list[TrainItem]) -> dict:
        p = self.predict(items)
        y = [it.target for it in items]
        # RPS needs ordinal order
        p_o, y_o = [], []
        for pi, yi, it in zip(p, y, items, strict=True):
            order = sorted(range(len(it.ordinals)), key=lambda k: it.ordinals[k])
            p_o.append([pi[k] for k in order])
            y_o.append([yi[k] for k in order])
        return compute_metrics(
            p_o, y_o, types=[it.type for it in items], flagged=[it.flagged for it in items]
        )
