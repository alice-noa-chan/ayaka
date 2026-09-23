"""End-to-end fine-tune run driver (sec 41/45/49).

Pools -> MixtureSampler -> pack_by_token_budget -> build_train_batch
-> Trainer, with checkpoints, metrics history, and dataset manifests
written under ``artifacts_dir``. Device-agnostic: the same function
runs locally (smoke) and inside the beam GPU container.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field

import torch

from ..config import MODEL_FAMILY, ElectraConfig, tiny_config
from ..data.loaders import load_pools
from ..data.manifest import write_manifest
from ..data.mixture import MixtureSampler
from ..data.packing import pack_by_token_budget
from ..data.schema import Sample
from ..model.model import ElectraDecisionModel
from ..tokenizer import HashTokenizer, load_bpe
from .batch import build_train_batch
from .trainer import TrainConfig, Trainer

DEFAULT_SPECS = [
    "snli",
    "multi_nli",
    "anli_r1",
    "boolq",
    "banking77",
    "clinc_oos",
    "super_glue_multirc",
    "klue_nli",
    "klue_sts",
    "klue_ynat",
    "kor_nli_multi",
    "massive_ja",
    "massive_ko",
    "jglue_jnli",
    "jglue_jsts",
    "jglue_commonsense",
    "go_emotions",
    "amazon_reviews",
]

SMOKE_SPECS = ["snli", "boolq", "banking77"]


@dataclass
class RunConfig:
    model_size: str = "electra-small"  # electra-small|base|large|tiny
    specs: list[str] = field(default_factory=lambda: list(DEFAULT_SPECS))
    limit_per_spec: int | None = 20_000
    steps: int = 1_000
    samples_per_step: int = 64
    token_budget: int = 65_536
    lr: float = 3e-5
    warmup_frac: float = 0.02
    temperature: float = 0.5
    tokenizer_path: str = ""  # tokenizers JSON; "" -> HashTokenizer
    artifacts_dir: str = "artifacts"
    run_name: str = "run"
    seed: int = 0
    compile: bool = False
    bf16: bool = True
    log_every: int = 20
    save_every: int = 0  # 0 -> only final checkpoint
    dedup: bool = True


def model_config(name: str) -> ElectraConfig:
    return tiny_config() if name == "tiny" else MODEL_FAMILY[name]


def build_tokenizer(vocab_size: int, tokenizer_path: str):
    if tokenizer_path and os.path.exists(tokenizer_path):
        return load_bpe(tokenizer_path)
    return HashTokenizer(vocab_size)


def synthetic_pools(n_per_cell: int = 32, seed: int = 0) -> dict[tuple[str, str], list[Sample]]:
    """Offline pools for smoke runs — no datasets dependency needed."""
    from ..data.schema import Candidate, Question, one_hot

    rng = random.Random(seed)
    cells = [("nli", "en"), ("choice", "en"), ("noul", "ko"), ("score", "ja")]
    pools: dict[tuple[str, str], list[Sample]] = {}
    for fam, lang in cells:
        cell = []
        for i in range(n_per_cell):
            if fam == "noul":
                q = Question.noul("q0", f"proposition {i}?", float(rng.random() > 0.5))
            else:
                k = 3 if fam != "score" else 5
                cands = [
                    Candidate(f"c{j}", f"option {j}", ordinal=j if fam == "score" else None)
                    for j in range(k)
                ]
                q = Question(
                    id="q0",
                    type="score" if fam == "score" else "choice",
                    instruction=f"question {i}?",
                    candidates=cands,
                    target_distribution=one_hot(cands, f"c{rng.randrange(k)}"),
                )
            cell.append(
                Sample(
                    state={"text": f"{lang} synthetic state {i}"},
                    questions=[q],
                    metadata={"language": lang, "task_family": fam, "source": "synthetic"},
                )
            )
        pools[(fam, lang)] = cell
    return pools


def packed_stream(
    pools: dict[tuple[str, str], list[Sample]],
    sampler: MixtureSampler,
    tokenizer,
    cfg: RunConfig,
    block_size: int,
    device,
):
    """Infinite stream of packed TrainBatches under the mixture quotas."""
    while True:
        drawn = sampler.sample(pools, cfg.samples_per_step)
        for ss, _desc in pack_by_token_budget(drawn, tokenizer, cfg.token_budget):
            yield build_train_batch(ss, tokenizer, block_size, device)


def run_training(cfg: RunConfig, pools=None, verbose: bool = True) -> dict:
    """Load data, build the model + trainer, run, checkpoint. Returns
    a summary dict (history tail, artifacts written, param count)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = model_config(cfg.model_size)
    tok = build_tokenizer(mcfg.vocab_size, cfg.tokenizer_path)
    art = os.path.join(cfg.artifacts_dir, cfg.run_name)
    os.makedirs(art, exist_ok=True)

    manifests = []
    if pools is None:
        pools, manifests = load_pools(cfg.specs, cfg.limit_per_spec, cfg.dedup)
        write_manifest(manifests, os.path.join(art, "dataset_manifest.jsonl"))
    if verbose:
        total = sum(len(v) for v in pools.values())
        print(f"[run] pools: {total} samples across {len(pools)} cells", flush=True)

    sampler = MixtureSampler(temperature=cfg.temperature, seed=cfg.seed)
    model = ElectraDecisionModel(mcfg)
    tcfg = TrainConfig(
        lr=cfg.lr,
        steps=cfg.steps,
        warmup_frac=cfg.warmup_frac,
        bf16=cfg.bf16,
        compile=cfg.compile,
        log_every=cfg.log_every,
        seed=cfg.seed,
        device=str(device),
    )
    trainer = Trainer(model, tcfg)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(
            f"[run] {mcfg.name} ~{n_params / 1e6:.0f}M params on {device}",
            flush=True,
        )

    t0 = time.time()
    history = trainer.train(packed_stream(pools, sampler, tok, cfg, mcfg.block_size, device))
    elapsed = time.time() - t0
    for rec in history:
        if verbose and cfg.log_every and rec["step"] % cfg.log_every == 0:
            print(
                f"[run] step {rec['step']}/{cfg.steps} loss={rec['loss']:.4f} "
                f"lr={rec['lr']:.2e} gn={rec['grad_norm']:.2f} {elapsed:.0f}s",
                flush=True,
            )

    ckpt = os.path.join(art, "checkpoint_final.pt")
    torch.save(trainer.state_dict(), ckpt)
    with open(os.path.join(art, "history.json"), "w") as f:
        json.dump(history, f)
    with open(os.path.join(art, "run_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    return {
        "checkpoint": ckpt,
        "artifacts": art,
        "params": n_params,
        "steps": trainer.step_i,
        "elapsed_sec": elapsed,
        "final_loss": history[-1]["loss"] if history else None,
        "manifests": len(manifests),
    }
