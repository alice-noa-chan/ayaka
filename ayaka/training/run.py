"""End-to-end decision training run.

    pools (decontaminated) ─► LoRA Electra ─► train ─► checkpoint
        ─► temperature calibration (jev-distill calibration split)
        ─► evals: held-out mix, Jev fidelity (test_set_30k), JevBench public

Distillation (Large -> Base/Small) is the same run with
``teacher_labels`` pointing at a jsonl written by training.distill:
those samples carry the teacher's distributions and are trained with
KL-to-teacher instead of the gold NLL term.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import torch

from ..checkpoint import apply_lora, save_checkpoint
from ..config import model_config
from ..data.decontam import Decontaminator
from ..data.dedup import lineage_key
from ..data.loaders import load_pools, load_spec_samples
from ..data.manifest import write_manifest
from ..data.mixture import TASK_FAMILY_QUOTA, MixtureSampler
from ..data.schema import Candidate, Question, Sample
from ..model.electra import ElectraDecisionModel
from ..serialization import canonical_state
from ..tokenization import HFTokenizer, ToyTokenizer
from .batching import TrainItem, sample_to_items
from .calibrate import apply_temperatures, fit_temperatures
from .trainer import TrainConfig, Trainer

# Default training data: only sources whose licenses/terms allow training
# a model that may be published (see the provenance notes in
# data/loaders.py). jev_open is the CC0 Open-Jev stream of jev-distill.
DEFAULT_SPECS = [
    "jev_open",
    "open_jev_bde",
    "vitaminc",
    "helpsteer2",
    "hh_rlhf",
    "aegis_safety",
    "aqua_rat",
    "hotpot_decisions",
    "squad_v2_answerable",
    "strategyqa",
    "arc_challenge",
    "commonsense_qa",
    "legalbench",
    "synth_temporal",
    "synth_numeric",
    "synth_policy",
    "synth_long_rules",
    "synth_calendar",
    "synth_probability",
    "snli",
    "multi_nli",
    "boolq",
    "banking77",
    "clinc_oos",
    "klue_nli",
    "klue_ynat",
    "kor_nli_multi",
    "massive_ja",
    "massive_ko",
    "jglue_jnli",
    "jglue_jsts",
    "jglue_commonsense",
    "go_emotions",
    "quality",
    "quality_dev",
]

# Opt-in only, each with a restriction: jev_distill (Jev API Output under
# TypeSafe MCA s2.3(b) + an unidentified 32B teacher), anli_r1 (CC BY-NC),
# super_glue_multirc (unclear terms), amazon_reviews (Amazon's terms).
RESTRICTED_SPECS = ["jev_distill", "anli_r1", "super_glue_multirc", "amazon_reviews"]
RELEASE_EXCLUDED = set(RESTRICTED_SPECS)
RELEASE_SPECS = list(DEFAULT_SPECS)

SMOKE_SPECS = ["jev_open", "boolq", "banking77"]


@dataclass
class RunConfig:
    model_size: str = "electra-small"
    specs: list[str] = field(default_factory=lambda: list(DEFAULT_SPECS))
    limit_per_spec: int | None = 20_000
    spec_limits: dict = field(
        default_factory=lambda: {
            "jev_open": 100_000,
            "open_jev_bde": 60_000,  # groups (several questions each)
            "vitaminc": 40_000,
            "hh_rlhf": 30_000,
            "helpsteer2": 21_000,
            "aegis_safety": 30_000,
            "aqua_rat": 30_000,
            "squad_v2_answerable": 30_000,
            "synth_temporal": 20_000,
            "synth_numeric": 20_000,
            "synth_policy": 15_000,
            "synth_long_rules": 10_000,
            "synth_calendar": 20_000,
            "synth_probability": 10_000,
            "jev_distill": 200_000,
        }
    )
    steps: int = 2000
    questions_per_step: int = 64
    micro_batch_tokens: int = 8_192  # auto-halved on CUDA OOM
    lr: float = 1e-4
    head_lr: float = 5e-4
    max_seq_len: int | None = None  # override the size default
    eval_questions: int = 1000
    eval_every: int = 250
    calibrate: bool = True
    calibration_questions: int = 4000
    calibration_spec: str = "jev_open_calibration"  # held-out split for temperatures
    calibration_per_bucket: int = 200  # reserved top-up per (primitive, length) bucket
    fidelity_questions: int = 3000  # reference-labelled held-out evaluation
    fidelity_spec: str = "jev_open_test"
    jevbench: bool = True
    teacher_labels: str = ""  # jsonl from training.distill -> distillation run
    teacher_quota: float = 0.6
    evidence_aug_p: float = 0.0
    decontaminate: bool = True
    include_restricted: bool = False  # add RESTRICTED_SPECS; otherwise they are dropped
    artifacts_dir: str = "artifacts"
    run_name: str = "run"
    seed: int = 0
    bf16: bool = True
    grad_checkpointing: bool = False  # enabled automatically if OOM persists
    liger: bool = False  # fused RMSNorm/GeGLU (CUDA+Triton), parity-checked
    compile: bool = False  # torch.compile per decoder layer (experimental)
    log_every: int = 20
    save_every: int = 0


def apply_release_policy(cfg: RunConfig, verbose: bool = True) -> None:
    """Default: drop restricted training specs and keep calibration and
    evaluation off Jev API Output (the jev_distill_* splits contain it).
    ``include_restricted`` adds the restricted specs instead."""
    if cfg.include_restricted:
        cfg.specs = list(cfg.specs) + [s for s in RESTRICTED_SPECS if s not in cfg.specs]
        if verbose:
            print(f"[run] include_restricted: training also on {RESTRICTED_SPECS}", flush=True)
        return
    dropped = [s for s in cfg.specs if s in RELEASE_EXCLUDED]
    cfg.specs = [s for s in cfg.specs if s not in RELEASE_EXCLUDED]
    if cfg.calibration_spec.startswith("jev_distill"):
        dropped.append(cfg.calibration_spec)
        cfg.calibration_spec = "jev_open_calibration"
    if cfg.fidelity_spec.startswith("jev_distill"):
        dropped.append(cfg.fidelity_spec)
        cfg.fidelity_spec = "jev_open_test"
    if verbose and dropped:
        print(f"[run] release policy: not using {dropped}", flush=True)


def build_tokenizer(backbone: str):
    return ToyTokenizer() if backbone == "tiny" else HFTokenizer.from_pretrained(backbone)


def build_model(cfg: RunConfig, device) -> ElectraDecisionModel:
    mcfg = model_config(cfg.model_size)
    if cfg.max_seq_len:
        from dataclasses import replace

        mcfg = replace(mcfg, max_seq_len=cfg.max_seq_len)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = ElectraDecisionModel.from_config(mcfg, dtype=dtype, device=device)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    if cfg.liger and device.type == "cuda":
        apply_liger(model, device)
    model = apply_lora(model)
    if cfg.compile:
        for layer in model.text_model().layers:
            layer.compile(dynamic=True)
    return model


def apply_liger(model: ElectraDecisionModel, device) -> bool:
    """Swap in Liger's fused RMSNorm/GeGLU kernels, keeping them only if
    the backbone output is unchanged on a probe input. Liger targets the
    plain Gemma 4 31B layer stack; E2B/E4B add per-layer embeddings and
    KV sharing, so parity is checked instead of assumed."""
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma4_text
    except ImportError:
        print("[liger] liger-kernel not installed; skipped", flush=True)
        return False
    text = model.text_model()
    probe = torch.randint(10, 1000, (2, 64), device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = text(input_ids=probe).last_hidden_state.float()
    try:
        apply_liger_kernel_to_gemma4_text(
            rope=False,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
            rms_norm=True,
            geglu=True,
            model=text,
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            got = text(input_ids=probe).last_hidden_state.float()
    except Exception as e:
        raise RuntimeError(
            f"liger patch failed on this backbone ({e}); rerun without --liger"
        ) from e
    rel = float((got - ref).abs().max() / ref.abs().max().clamp(min=1e-6))
    print(f"[liger] applied, parity rel-err={rel:.2e}", flush=True)
    if rel > 2e-2:
        raise RuntimeError(
            f"liger changed backbone outputs (rel-err {rel:.2e}); rerun without --liger"
        )
    return True


def synthetic_pools(n_per_cell: int = 16, seed: int = 0) -> dict[tuple[str, str], list[Sample]]:
    """Tiny rule-labeled pools for smoke/CPU tests (no downloads)."""
    rng = random.Random(seed)
    pools: dict[tuple[str, str], list[Sample]] = {}
    colors = ["red", "green", "blue", "amber"]
    for fam in ("direct_jev", "choice", "noul", "score"):
        cell = []
        for i in range(n_per_cell):
            c = rng.choice(colors)
            state = {"light": c, "ticket": i}
            if fam == "noul":
                q = Question.noul("q0", f"Is the light {c}?", 1.0 if i % 2 == 0 else 0.0)
                if i % 2:
                    state["light"] = rng.choice([x for x in colors if x != c])
            elif fam == "score":
                cands = [Candidate(f"s{k}", f"level {k}", ordinal=k) for k in range(4)]
                lvl = colors.index(c)
                q = Question(
                    "q0",
                    "score",
                    "How warm is the light color?",
                    cands,
                    {f"s{k}": float(k == lvl) for k in range(4)},
                )
            else:
                cands = [Candidate(x, f"the light is {x}") for x in colors]
                q = Question(
                    "q0",
                    "choice",
                    "Which color is the light?",
                    cands,
                    {x: float(x == c) for x in colors},
                )
            cell.append(
                Sample(
                    state=state,
                    questions=[q],
                    metadata={
                        "language": "en",
                        "task_family": fam,
                        "source_example_id": f"{fam}-{i}",
                    },
                )
            )
        pools[(fam, "en")] = cell
    return pools


def _state_groups(pools: dict[tuple[str, str], list[Sample]]):
    """Union state, lineage and generator-scenario aliases into indivisible groups.

    Returns ``(groups, cell_groups)``: group key -> [(cell, sample)], and
    cell -> set of group keys (so callers never empty a training cell).
    """
    parent = {}

    def root(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    entries = []
    for cell, samples in pools.items():
        for sample in samples:
            digest = hashlib.sha256(canonical_state(sample.state).encode()).hexdigest()
            key = f"state:{digest}"
            aliases = [key]
            if lineage := lineage_key(sample):
                aliases.append(f"lineage:{lineage}")
            if split_group := sample.metadata.get("split_group"):
                aliases.append(f"generator:{split_group}")
            for alias in aliases:
                parent[root(alias)] = root(key)
            entries.append((cell, sample, key))
    groups = defaultdict(list)
    cell_groups = defaultdict(set)
    for cell, sample, key in entries:
        key = root(key)
        groups[key].append((cell, sample))
        cell_groups[cell].add(key)
    return groups, cell_groups


def _remove_held(pools, held) -> None:
    held_ids = {id(sample) for sample in held}
    for samples in pools.values():
        samples[:] = [sample for sample in samples if id(sample) not in held_ids]


def split_eval(pools: dict[tuple[str, str], list[Sample]], n_eval: int, seed: int) -> list[Sample]:
    """Hold out whole state/lineage groups, including cross-source duplicates.

    ``n_eval`` is a question budget; an indivisible group may overshoot it.
    Always leave at least one group in every affected training cell.
    """
    if n_eval <= 0:
        return []
    rng = random.Random(seed + 3)
    groups, cell_groups = _state_groups(pools)
    keys = sorted(groups)
    rng.shuffle(keys)
    held, n_questions = [], 0
    for key in keys:
        affected = {cell for cell, _ in groups[key]}
        if any(len(cell_groups[cell]) <= 1 for cell in affected):
            continue
        for cell in affected:
            cell_groups[cell].remove(key)
        held.extend(sample for _, sample in groups[key])
        n_questions += sum(len(sample.questions) for _, sample in groups[key])
        if n_questions >= n_eval:
            break
    _remove_held(pools, held)
    return held


def is_generated_source(source: str | None) -> bool:
    return str(source or "").startswith("synth_")


def reserve_calibration(
    pools: dict[tuple[str, str], list[Sample]],
    tok,
    mcfg,
    per_bucket: int = 200,
    seed: int = 0,
    max_groups: int = 50_000,
) -> list[TrainItem]:
    """Hold out whole groups until every (primitive, length) bucket has
    ``per_bucket`` calibration questions, and return their items.

    A flat random draw rarely yields long prompts: the 2026-09-26 Small run
    got 1 long noul and 10 long score questions, so those buckets silently
    fell back to short-prompt temperatures (noul T=0.88 sharpened long, hard
    prompts; JevBench hard ECE 0.29). A group is taken only if it adds a
    question to a bucket that is still short.

    Natural (non-generated) sources are scanned first. The model trains on
    the procedural generators' own templates, where it is far more accurate
    than on unseen long documents, so fitting temperatures there would
    understate uncertainty. Generated groups only fill what remains.
    """
    if per_bucket <= 0:
        return []
    rng = random.Random(seed + 17)
    groups, cell_groups = _state_groups(pools)
    keys = sorted(groups)
    rng.shuffle(keys)
    keys.sort(
        key=lambda k: any(is_generated_source(s.metadata.get("source")) for _, s in groups[k])
    )
    long_n = mcfg.long_prompt_tokens
    need = {(t, b): per_bucket for t in ("noul", "choice", "score") for b in (False, True)}
    present = {q.type for cell in pools.values() for s in cell for q in s.questions}
    need = {k: v for k, v in need.items() if k[0] in present}
    held, items = [], []
    for key in keys[:max_groups]:
        if not any(need.values()):
            break
        affected = {cell for cell, _ in groups[key]}
        if any(len(cell_groups[cell]) <= 1 for cell in affected):
            continue
        group_items = [it for _, s in groups[key] for it in sample_to_items(s, tok, mcfg)]
        if not any(need.get((it.type, it.length >= long_n), 0) for it in group_items):
            continue
        for cell in affected:
            cell_groups[cell].remove(key)
        held.extend(sample for _, sample in groups[key])
        for it in group_items:
            bucket = it.type, it.length >= long_n
            if need.get(bucket, 0):
                need[bucket] -= 1
        items.extend(group_items)
    _remove_held(pools, held)
    return items


def pool_summary(pools) -> dict:
    """Audit the question population without tokenizing the entire corpus."""
    summary = {}
    for (family, language), samples in sorted(pools.items()):
        primitives, sources = Counter(), Counter()
        true_mass, n_noul = 0.0, 0
        for sample in samples:
            sources[sample.metadata.get("source", "synthetic")] += len(sample.questions)
            for question in sample.questions:
                primitives[question.type] += 1
                if question.type == "noul":
                    n_noul += 1
                    true_mass += question.target_distribution.get("true", 0.0)
        summary[f"{family}/{language}"] = {
            "states": len(samples),
            "questions": sum(primitives.values()),
            "primitives": dict(primitives),
            "source_questions": dict(sources),
            "noul_true_mass": true_mass / n_noul if n_noul else None,
        }
    return summary


def item_stream(pools, sampler: MixtureSampler, tok, mcfg, cfg: RunConfig, rng: random.Random):
    from ..data.augment import evidence_deletion_variant

    for drawn in sampler.question_batches(pools, cfg.questions_per_step):
        items: list[TrainItem] = []
        for s in drawn:
            if (
                cfg.evidence_aug_p
                and rng.random() < cfg.evidence_aug_p
                and "teacher_probs" not in s.metadata
            ):
                s = evidence_deletion_variant(s, rng, target="uniform")
            items.extend(sample_to_items(s, tok, mcfg))
        yield items


def load_teacher_samples(path: str) -> list[Sample]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Sample.from_json(json.loads(line)))
    return out


def items_from_spec(
    spec: str, n: int, tok, mcfg, seed: int, decon: Decontaminator | None
) -> list[TrainItem]:
    samples, _ = load_spec_samples(spec, limit=n, dedup=False, seed=seed)
    if decon is not None:
        samples, _ = decon.filter(samples)
    return [it for s in samples for it in sample_to_items(s, tok, mcfg)][:n]


def run_training(cfg: RunConfig, pools=None, verbose: bool = True) -> dict:
    # Seed head and adapter initialization too, not just the later Trainer.
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    art = os.path.join(cfg.artifacts_dir, cfg.run_name)
    os.makedirs(art, exist_ok=True)
    apply_release_policy(cfg, verbose)
    with open(os.path.join(art, "run_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    model = build_model(cfg, device)
    mcfg = model.cfg
    tok = build_tokenizer(mcfg.backbone)
    decon = Decontaminator.from_jevbench() if cfg.decontaminate else None

    manifests = []
    if pools is None:
        pools, manifests = load_pools(
            cfg.specs,
            cfg.limit_per_spec,
            dedup=True,
            limits=cfg.spec_limits,
            seed=cfg.seed,
            decontaminator=decon,
        )
        write_manifest(manifests, os.path.join(art, "dataset_manifest.jsonl"))
    quota = dict(TASK_FAMILY_QUOTA)
    if cfg.teacher_labels:
        pools[("teacher_distill", "en")] = load_teacher_samples(cfg.teacher_labels)
        quota = {k: v * (1 - cfg.teacher_quota) for k, v in quota.items()}
        quota["teacher_distill"] = cfg.teacher_quota
    if not any(pools.values()):
        raise RuntimeError("no training samples: every dataset spec failed to load")
    held = split_eval(pools, cfg.eval_questions, cfg.seed) if cfg.eval_questions else []
    # separate from `held`: tops up (primitive, length) buckets the calibration
    # spec lacks (Open-Jev has no score and few long questions) without
    # touching evaluation data
    cal_reserve = (
        reserve_calibration(pools, tok, mcfg, cfg.calibration_per_bucket, cfg.seed + 11)
        if cfg.calibrate
        else []
    )
    with open(os.path.join(art, "data_summary.json"), "w") as f:
        json.dump(pool_summary(pools), f, indent=2)
    eval_items = [it for s in held for it in sample_to_items(s, tok, mcfg)]
    if verbose:
        sizes = {f"{f}/{lang}": len(v) for (f, lang), v in pools.items()}
        print(f"[run] pools: {sizes} | held-out questions: {len(eval_items)}", flush=True)

    tcfg = TrainConfig(
        steps=cfg.steps,
        questions_per_step=cfg.questions_per_step,
        micro_batch_tokens=cfg.micro_batch_tokens,
        lr=cfg.lr,
        head_lr=cfg.head_lr,
        bf16=cfg.bf16,
        grad_checkpointing=cfg.grad_checkpointing and device.type == "cuda",
        log_every=cfg.log_every,
        eval_every=cfg.eval_every,
        seed=cfg.seed,
    )
    trainer = Trainer(model, tok, tcfg, device)
    if verbose:
        print(
            f"[run] {mcfg.name} ({mcfg.backbone}) trainable={trainer.n_trainable() / 1e6:.1f}M on {device}",
            flush=True,
        )

    ckpt_dir = os.path.join(art, "checkpoint")

    def on_step(step, _rec):
        if cfg.save_every and step % cfg.save_every == 0:
            save_checkpoint(model, os.path.join(art, f"checkpoint-{step}"), {"step": step})

    sampler = MixtureSampler(quota=quota, seed=cfg.seed)
    t0 = time.time()
    history = trainer.train(
        item_stream(pools, sampler, tok, mcfg, cfg, random.Random(cfg.seed + 7)),
        eval_items,
        on_step,
        verbose,
    )
    train_sec = time.time() - t0
    with open(os.path.join(art, "history.json"), "w") as f:
        json.dump(history, f)
    meta = {"steps": trainer.step_i, "train_sec": train_sec, "run": asdict(cfg)}
    save_checkpoint(model, ckpt_dir, meta)  # before calibration: a late failure can't lose the run

    report: dict = {"train_sec": train_sec, "steps": trainer.step_i}
    if cfg.calibrate:
        if _is_synthetic(pools):
            cal = eval_items
        else:
            cal = items_from_spec(
                cfg.calibration_spec, cfg.calibration_questions, tok, mcfg, cfg.seed, decon
            )
            extra = cal_reserve
            long_n = mcfg.long_prompt_tokens
            have = {
                (t, b): sum(it.type == t and (it.length >= long_n) == b for it in cal)
                for t in ("noul", "choice", "score")
                for b in (False, True)
            }
            # top up every (primitive, length bucket) the calibration split lacks,
            # long prompts especially: they were overconfident with one scalar
            for it in extra:
                key = it.type, it.length >= long_n
                if have[key] < cfg.calibration_per_bucket:
                    cal.append(it)
                    have[key] += 1
            report["calibration_counts"] = {
                f"{prim}@{'long' if long else 'short'}": count
                for (prim, long), count in have.items()
            }
        _, logits = trainer.predict(cal, apply_temperature=False, return_logits=True)
        temps = fit_temperatures(
            logits,
            [it.target for it in cal],
            [it.type for it in cal],
            [it.length for it in cal],
            mcfg.long_prompt_tokens,
        )
        apply_temperatures(model, temps)
        report["temperatures"] = temps
        if verbose:
            print(f"[calibrate] {temps} on {len(cal)} questions", flush=True)
    if eval_items:
        report["heldout"] = trainer.evaluate(eval_items)
        if verbose:
            print(f"[eval] held-out {_fmt(report['heldout'])}", flush=True)
    if cfg.fidelity_questions and not _is_synthetic(pools):
        fid = items_from_spec(cfg.fidelity_spec, cfg.fidelity_questions, tok, mcfg, cfg.seed, decon)
        report["reference_eval"] = {"spec": cfg.fidelity_spec, **trainer.evaluate(fid)}
        if verbose:
            print(f"[eval] {cfg.fidelity_spec} {_fmt(report['reference_eval'])}", flush=True)
    meta.update(report)
    save_checkpoint(model, ckpt_dir, meta)
    if cfg.jevbench:
        from ..eval.jevbench import run_jevbench

        report["jevbench"] = run_jevbench(
            model, tok, out_path=os.path.join(art, "jevbench_report.json"), verbose=verbose
        )["summary"]
    with open(os.path.join(art, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return {"checkpoint": ckpt_dir, "artifacts": art, **report}


def _is_synthetic(pools) -> bool:
    return all(s.metadata.get("source") is None for cell in pools.values() for s in cell[:1])


def _fmt(m: dict) -> str:
    keys = ("n", "accuracy", "nll", "kl", "brier", "ece")
    return " ".join(
        f"{k}={m[k]:.4f}" if isinstance(m.get(k), float) else f"{k}={m.get(k)}"
        for k in keys
        if k in m
    )
