"""LR schedule: linear warmup + cosine decay to a floor (sec 45.7).

warmup = 2% of total steps; minimum LR = 10% of peak.
"""

from __future__ import annotations

import math

import torch


def cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_frac: float = 0.02,
    min_lr_frac: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_frac + (1 - min_lr_frac) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: bool = True,
) -> torch.optim.AdamW:
    """Fused AdamW; bias/norm/embedding-row params skip weight decay
    (sec 45.7)."""
    decay, no_decay = [], []
    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 1-dim params are norms and biases -> no weight decay
        (no_decay if p.ndim <= 1 else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    use_fused = fused and torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=use_fused)
