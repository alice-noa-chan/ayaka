"""Teacher labeling for Large -> Base/Small distillation (docs.md 28.4, Stage 5).

The trained Large model scores a pool of training samples; each sample
is written back as canonical-schema JSON with the teacher's calibrated
distribution per question in ``metadata.teacher_probs``. A student run
with ``RunConfig.teacher_labels`` trains on KL-to-teacher for these.

Gold lineage is kept: ``target_distribution`` stays the original label,
so the student still sees Brier/RPS against gold (sec 44).
"""

from __future__ import annotations

import json
import os
import random

import torch

from ..checkpoint import load_checkpoint
from ..data.decontam import Decontaminator
from ..data.loaders import load_pools
from ..tokenization import HFTokenizer, ToyTokenizer
from .batching import sample_to_items
from .trainer import TrainConfig, Trainer


def label_with_teacher(
    teacher_ckpt: str,
    out_path: str,
    specs: list[str],
    n_samples: int,
    limit_per_spec: int | None = 20_000,
    spec_limits: dict | None = None,
    micro_batch_tokens: int = 32_768,
    seed: int = 1,
    pools=None,
    verbose: bool = True,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = load_checkpoint(teacher_ckpt, device=device, dtype=dtype).requires_grad_(False)
    cfg = model.cfg
    tok = ToyTokenizer() if cfg.backbone == "tiny" else HFTokenizer.from_pretrained(cfg.backbone)
    if pools is None:
        pools, _ = load_pools(
            specs,
            limit_per_spec,
            dedup=True,
            limits=spec_limits,
            seed=seed,
            decontaminator=Decontaminator.from_jevbench(),
        )
    samples = [s for cell in pools.values() for s in cell]
    random.Random(seed).shuffle(samples)
    samples = samples[:n_samples]

    trainer = Trainer(
        model,
        tok,
        TrainConfig(micro_batch_tokens=micro_batch_tokens, grad_checkpointing=False, steps=1),
        device,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n_q = 0
    chunk = 2048
    with open(out_path, "w", encoding="utf-8") as f:
        for start in range(0, len(samples), chunk):
            part = samples[start : start + chunk]
            items_per = [sample_to_items(s, tok, cfg) for s in part]
            flat = [it for its in items_per for it in its]
            probs = trainer.predict(flat)
            k = 0
            for s, its in zip(part, items_per, strict=True):
                s.metadata = dict(s.metadata)
                s.metadata["teacher_probs"] = {
                    q.id: probs[k + j] for j, q in enumerate(s.questions)
                }
                s.metadata["teacher"] = teacher_ckpt
                k += len(its)
                f.write(json.dumps(s.to_json(), ensure_ascii=False) + "\n")
            n_q += len(flat)
            if verbose:
                print(
                    f"[distill] labeled {min(start + chunk, len(samples))}/{len(samples)} samples",
                    flush=True,
                )
    return {"out": out_path, "samples": len(samples), "questions": n_q}
