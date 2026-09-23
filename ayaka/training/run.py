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
    train_tokenizer: bool = False  # train BPE on pooled text when path missing
    pretrain_steps: int = 0  # RTD + structured-corruption stage first
    pretrain_lr: float = 3e-4
    augment_p: float = 0.0  # per-sample candidate permutation prob
    evidence_aug_p: float = 0.0  # insufficient-evidence counterfactual prob
    eval_samples: int = 0  # held-out rows for eval + temperature fit
    calibrate: bool = False
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


def _corpus_texts(pools: dict[tuple[str, str], list[Sample]], limit: int = 50_000):
    """Raw text for BPE training: serialized states + instructions +
    candidate descriptions across all pools."""
    from ..serialization import serialize_typed

    n = 0
    for cell in pools.values():
        for s in cell:
            yield serialize_typed(s.state)
            for q in s.questions:
                yield q.instruction
                for c in q.candidates:
                    yield c.description
            n += 1
            if n >= limit:
                return


def build_tokenizer(
    vocab_size: int,
    tokenizer_path: str,
    train: bool = False,
    corpus=None,
    verbose: bool = True,
):
    """Load a trained BPE if the path exists; optionally train one on
    the pooled corpus; else fall back to HashTokenizer."""
    from ..tokenizer import train_bpe

    if tokenizer_path and os.path.exists(tokenizer_path):
        if verbose:
            print(f"[run] tokenizer: load {tokenizer_path}", flush=True)
        return load_bpe(tokenizer_path)
    if train and corpus is not None:
        if verbose:
            print("[run] tokenizer: training BPE on pooled corpus", flush=True)
        tok = train_bpe(corpus, vocab_size=vocab_size, save_path=tokenizer_path or None)
        return tok
    if verbose:
        print("[run] tokenizer: HashTokenizer fallback", flush=True)
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


def _augment_sample(sample: Sample, cfg: RunConfig, rng: random.Random) -> Sample:
    """Counterfactual augmentation (sec 34/44): candidate permutation
    keeps the target attached to its candidate; evidence deletion
    produces a flagged insufficient-evidence variant with a uniform
    target (never a forced one-hot, A5)."""
    import copy

    from ..data.augment import evidence_deletion_variant, permute_candidates

    s = sample
    if cfg.augment_p > 0 and rng.random() < cfg.augment_p:
        s = copy.deepcopy(s)
        s.questions = [permute_candidates(q, rng) for q in s.questions]
    if cfg.evidence_aug_p > 0 and rng.random() < cfg.evidence_aug_p:
        s = evidence_deletion_variant(s, rng, target="uniform")
    return s


def packed_stream(
    pools: dict[tuple[str, str], list[Sample]],
    sampler: MixtureSampler,
    tokenizer,
    cfg: RunConfig,
    block_size: int,
    device,
    rng: random.Random | None = None,
):
    """Infinite stream of packed TrainBatches under the mixture quotas."""
    rng = rng or random.Random(cfg.seed + 1)
    while True:
        drawn = sampler.sample(pools, cfg.samples_per_step)
        drawn = [_augment_sample(s, cfg, rng) for s in drawn]
        for ss, _desc in pack_by_token_budget(drawn, tokenizer, cfg.token_budget):
            yield build_train_batch(ss, tokenizer, block_size, device)


def pretrain_stage(
    model: ElectraDecisionModel,
    pools: dict[tuple[str, str], list[Sample]],
    tokenizer,
    cfg: RunConfig,
    device,
    verbose: bool = True,
) -> list[dict]:
    """Foundation stage (sec 40.1): RTD + structured-corruption
    discriminator losses over encoded states from all pools.

    v1 generator policy (A10): uniform corruption, no learned MLM
    generator. Span-relation / multilingual-alignment / evidence-
    retrieval heads exist in pretrain.py but need paired data that
    this corpus does not provide — they are not wired here.
    """
    from ..collate import encode_state
    from ..special_tokens import NUM_SPECIAL_TOKENS
    from .pretrain import (
        RTDHead,
        rtd_corrupt,
        rtd_loss,
        structured_corrupt,
        structured_corrupt_loss,
    )

    head = RTDHead(model.cfg.hidden).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=cfg.pretrain_lr)
    rng = random.Random(cfg.seed + 2)
    all_samples = [s for cell in pools.values() for s in cell]
    history = []
    model.train()
    for step in range(1, cfg.pretrain_steps + 1):
        drawn = rng.sample(all_samples, min(cfg.samples_per_step, len(all_samples)))
        ids: list[int] = []
        cu = [0]
        for s in drawn:
            sids = encode_state(s.state, tokenizer)
            ids += sids
            cu.append(len(ids))
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        cu_t = torch.tensor(cu, dtype=torch.long, device=device)
        gen = torch.Generator(device=device)
        gen.manual_seed(cfg.seed + step)
        corrupt_rtd, rtd_labels = rtd_corrupt(
            ids_t, cu_t, NUM_SPECIAL_TOKENS, tokenizer.vocab_size, generator=gen
        )
        corrupt_struct, struct_labels = structured_corrupt(ids_t, generator=gen)
        opt.zero_grad(set_to_none=True)
        loss = rtd_loss(model, head, ids_t, corrupt_rtd, rtd_labels, cu_t)
        loss = loss + 0.3 * structured_corrupt_loss(
            model, head, ids_t, corrupt_struct, struct_labels, cu_t
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
        opt.step()
        rec = {"step": step, "pretrain_loss": float(loss.detach())}
        history.append(rec)
        if verbose and cfg.log_every and step % cfg.log_every == 0:
            print(
                f"[pretrain] step {step}/{cfg.pretrain_steps} loss={rec['pretrain_loss']:.4f}",
                flush=True,
            )
    return history


def _split_eval(pools: dict[tuple[str, str], list[Sample]], n_eval: int, seed: int) -> list[Sample]:
    """Pop a proportional held-out eval set out of the pools."""
    rng = random.Random(seed + 3)
    total = sum(len(v) for v in pools.values())
    eval_set: list[Sample] = []
    for cell in pools.values():
        if not cell or not total:
            continue
        n = max(1, round(n_eval * len(cell) / total)) if cell else 0
        n = min(n, len(cell) // 10, len(cell) - 1) if len(cell) > 1 else 0
        if n > 0:
            eval_set.extend(rng.sample(cell, n))
            del_ids = {id(s) for s in eval_set[-n:]}
            cell[:] = [s for s in cell if id(s) not in del_ids]
    return eval_set


def _calibrate(
    model: ElectraDecisionModel,
    eval_samples: list[Sample],
    tokenizer,
    cfg: RunConfig,
    block_size: int,
    device,
) -> dict[int, float]:
    """Fit per-primitive scalar temperatures on held-out data (A8)."""
    from .calibrate import apply_temperatures, collect_logits_by_primitive, fit_temperatures

    if not eval_samples:
        return {}
    batches = []
    for ss, _ in pack_by_token_budget(eval_samples, tokenizer, cfg.token_budget):
        batches.append(build_train_batch(ss, tokenizer, block_size, device))
    collected = collect_logits_by_primitive(model, batches)
    temps = fit_temperatures(
        {p: v["logits"] for p, v in collected.items()},
        {p: v["targets"] for p, v in collected.items()},
        {p: v["cu"] for p, v in collected.items()},
    )
    apply_temperatures(model, temps)
    return temps


def run_training(cfg: RunConfig, pools=None, verbose: bool = True) -> dict:
    """Load data, build the model + trainer, run, checkpoint. Returns
    a summary dict (history tail, artifacts written, param count)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = model_config(cfg.model_size)
    art = os.path.join(cfg.artifacts_dir, cfg.run_name)
    os.makedirs(art, exist_ok=True)

    manifests = []
    if pools is None:
        pools, manifests = load_pools(cfg.specs, cfg.limit_per_spec, cfg.dedup)
        write_manifest(manifests, os.path.join(art, "dataset_manifest.jsonl"))
    if verbose:
        total = sum(len(v) for v in pools.values())
        print(f"[run] pools: {total} samples across {len(pools)} cells", flush=True)
    if not any(pools.values()):
        raise RuntimeError("no training samples: every dataset spec failed to load")

    eval_samples = _split_eval(pools, cfg.eval_samples, cfg.seed) if cfg.eval_samples else []
    tok_path = cfg.tokenizer_path or os.path.join(art, "tokenizer.json")
    tok = build_tokenizer(
        mcfg.vocab_size,
        tok_path,
        train=cfg.train_tokenizer,
        corpus=_corpus_texts(pools) if cfg.train_tokenizer else None,
        verbose=verbose,
    )

    sampler = MixtureSampler(temperature=cfg.temperature, seed=cfg.seed)
    model = ElectraDecisionModel(mcfg)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(
            f"[run] {mcfg.name} ~{n_params / 1e6:.0f}M params on {device}",
            flush=True,
        )

    pretrain_hist = []
    if cfg.pretrain_steps:
        t0 = time.time()
        pretrain_hist = pretrain_stage(model, pools, tok, cfg, device, verbose)
        if verbose:
            print(f"[pretrain] done in {time.time() - t0:.0f}s", flush=True)

    eval_batches = None
    if eval_samples:
        eval_batches = [
            build_train_batch(ss, tok, mcfg.block_size, device)
            for ss, _ in pack_by_token_budget(eval_samples, tok, cfg.token_budget)
        ]

    tcfg = TrainConfig(
        lr=cfg.lr,
        steps=cfg.steps,
        warmup_frac=cfg.warmup_frac,
        bf16=cfg.bf16,
        compile=cfg.compile,
        log_every=cfg.log_every,
        eval_every=cfg.log_every if eval_batches else 0,
        seed=cfg.seed,
        device=str(device),
    )
    trainer = Trainer(model, tcfg)

    t0 = time.time()
    history = trainer.train(
        packed_stream(pools, sampler, tok, cfg, mcfg.block_size, device),
        eval_batches=eval_batches,
    )
    elapsed = time.time() - t0
    for rec in history:
        if verbose and cfg.log_every and rec["step"] % cfg.log_every == 0:
            ev = f" eval_nll={rec.get('eval_nll', float('nan')):.4f}" if eval_batches else ""
            print(
                f"[run] step {rec['step']}/{cfg.steps} loss={rec['loss']:.4f} "
                f"lr={rec['lr']:.2e} gn={rec['grad_norm']:.2f} {elapsed:.0f}s{ev}",
                flush=True,
            )

    temps = {}
    eval_metrics = {}
    if eval_batches:
        eval_metrics = trainer.evaluate(eval_batches)
        if verbose:
            print(f"[eval] {eval_metrics}", flush=True)
    if cfg.calibrate:
        temps = _calibrate(model, eval_samples, tok, cfg, mcfg.block_size, device)
        if verbose:
            print(f"[calibrate] temperatures: {temps}", flush=True)

    ckpt = os.path.join(art, "checkpoint_final.pt")
    torch.save(trainer.state_dict(), ckpt)
    with open(os.path.join(art, "history.json"), "w") as f:
        json.dump(history, f)
    with open(os.path.join(art, "pretrain_history.json"), "w") as f:
        json.dump(pretrain_hist, f)
    with open(os.path.join(art, "run_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    return {
        "checkpoint": ckpt,
        "artifacts": art,
        "params": n_params,
        "steps": trainer.step_i,
        "elapsed_sec": elapsed,
        "final_loss": history[-1]["loss"] if history else None,
        "eval_metrics": eval_metrics,
        "temperatures": temps,
        "manifests": len(manifests),
        "tokenizer": tok_path if os.path.exists(tok_path) else "hash",
    }
