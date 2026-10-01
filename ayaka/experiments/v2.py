"""Single-GPU exploration with process deadlines and persistent budget accounting."""

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch

from ..checkpoint import apply_lora, load_checkpoint, save_checkpoint
from ..config import ElectraConfig
from ..data.decontam import Decontaminator
from ..data.reasoning_v2 import CURRICULUM_VERSION, SPLITS, curriculum
from ..eval.reasoning_v2 import dataset_signature
from ..eval.reasoning_v2 import evaluate_efforts as run_efforts
from ..eval.v2 import assert_isolated, select_candidates
from ..losses import LossWeights
from ..model.electra import ElectraDecisionModel
from ..reasoning_pipeline import controlled_decision
from ..routing import fit_router, paired_training_rows
from ..tokenization import HFTokenizer
from ..training.path_calibration import PathCalibration
from ..training.reasoning import reasoning_items
from ..training.trainer import TrainConfig, Trainer

STAGE_SECONDS = {
    "screen": 7200,
    "recover_screen": 3600,
    "heads": 3600,
    "sft": 7200,
    "evaluate": 7200,
    "reserve": 3600,
}
TOTAL_SECONDS = 28800


def training_stream(items, seed=15):
    rng = random.Random(seed)
    while True:
        yield rng.sample(items, min(8, len(items)))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in manifest["candidates"]:
        if len(item["revision"]) != 40 or any(
            c not in "0123456789abcdef" for c in item["revision"]
        ):
            raise ValueError("each model must have an exact Hub revision")
        if (
            item["license"] not in ("mit", "apache-2.0")
            or not 0 < item["total_parameters"] <= 14_000_000_000
        ):
            raise ValueError("candidate violates license or total-parameter policy")
    return manifest


def candidate_config(candidate, **overrides):
    return ElectraConfig(
        name=candidate["name"],
        backbone=candidate["repo"],
        backbone_revision=candidate["revision"],
        pointer_dim=256,
        set_mixer_layers=2,
        set_mixer_heads=4,
        lora_r=16,
        lora_alpha=32,
        lora_targets=tuple(candidate["lora_targets"]),
        version=2,
        **overrides,
    )


def prepare(manifest_path, out, download_weights=False):
    from huggingface_hub import snapshot_download

    manifest = load_manifest(manifest_path)
    archive_stale_curriculum(out)
    records = {split: curriculum(split, 128 if split == "train" else 32) for split in SPLITS}
    assert_isolated({s: [sample for sample, _ in rows] for s, rows in records.items()})
    decon = Decontaminator.from_jevbench()
    for sample, traces in records["train"]:
        if decon.sample_hit(sample) or any(decon.text_hit(t) for t in traces.values()):
            raise ValueError("verified curriculum overlaps public JevBench; revise the generator")
    for split, rows in records.items():
        write_json(
            Path(out) / f"{split}.json",
            [{"sample": sample.to_json(), "traces": traces} for sample, traces in rows],
        )
    # Additional natural-language diagnostics are evaluation-only. They never
    # enter the curriculum, router fitting, calibration, or candidate ranking.
    from dataclasses import asdict

    from ..data.loaders import load_spec_samples

    natural_status = []
    natural = []
    for source in ("jev_open_test", "helpsteer2", "klue_nli", "jglue_jnli"):
        try:
            samples, provenance = load_spec_samples(source, limit=32, seed=15)
            kept, removed = decon.filter(samples)
            for sample in kept:
                sample.questions = sample.questions[:1]
                sample.metadata.update(
                    split="test",
                    source_example_id=f"natural/{source}/{sample.metadata.get('source_example_id')}",
                )
            natural.extend(kept[:32])
            natural_status.append(
                {
                    "source": source,
                    "status": "ready",
                    "n": min(32, len(kept)),
                    "removed_overlap": removed,
                    "provenance": asdict(provenance),
                }
            )
        except Exception as exc:
            natural_status.append(
                {
                    "source": source,
                    "status": "unavailable",
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                }
            )
    write_json(Path(out) / "natural_test.json", [s.to_json() for s in natural])
    write_json(Path(out) / "natural_provenance.json", natural_status)
    statuses = []
    for candidate in manifest["candidates"]:
        entry = {"name": candidate["name"], "status": candidate["support"]}
        if candidate["support"] == "builtin":
            try:
                patterns = [
                    "*.json",
                    "*.jinja",
                    "*.model",
                    "*.txt",
                    "tokenizer*",
                    "vocab*",
                    "merges*",
                ]
                if download_weights:
                    patterns.append("*.safetensors")
                snapshot_download(
                    candidate["repo"], revision=candidate["revision"], allow_patterns=patterns
                )
                entry["status"] = "weights_ready" if download_weights else "metadata_ready"
            except Exception as exc:
                entry.update(
                    status="download_failed", error=f"{type(exc).__name__}: {str(exc)[:300]}"
                )
        statuses.append(entry)
    write_json(
        Path(out) / "preparation.json",
        {
            "candidates": statuses,
            "cuda_available": torch.cuda.is_available(),
            "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
            "splits_isolated": True,
            "curriculum_version": CURRICULUM_VERSION,
        },
    )
    return statuses


def archive_stale_curriculum(out):
    folder = Path(out)
    previous = folder / "preparation.json"
    if not previous.exists():
        return
    version = json.loads(previous.read_text()).get("curriculum_version", 1)
    if version == CURRICULUM_VERSION:
        return
    archive = folder / f"interrupted-curriculum-v{version}"
    archive.mkdir(exist_ok=True)
    for path in folder.glob("*.json"):
        shutil.copy2(path, archive / path.name)
    if (folder / "screen").exists():
        shutil.copytree(folder / "screen", archive / "screen", dirs_exist_ok=True)
    write_json(folder / "screen_summary.json", [])
    write_json(folder / "selection.json", {"ranked": [], "status": "stale_curriculum"})


def gpu_info():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exploration requires exactly one CUDA GPU")
    prop = torch.cuda.get_device_properties(0)
    if "H100" not in prop.name or prop.total_memory < 75 * 1024**3:
        raise RuntimeError("exploration requires H100 80GB")
    return {"name": prop.name, "bytes": prop.total_memory, "torch": torch.__version__}


def load_candidate(candidate):
    torch.manual_seed(15)
    cfg = candidate_config(candidate)
    tok = HFTokenizer.for_config(cfg)
    model = ElectraDecisionModel.from_config(cfg, device="cuda", dtype=torch.bfloat16).eval()
    return model, tok


def worker(manifest_path, out, stage, seconds):
    evaluate_efforts = partial(run_efforts, verbose=True)
    gpu = gpu_info()
    start, deadline = time.monotonic(), time.monotonic() + seconds
    manifest = load_manifest(manifest_path)
    folder = Path(out)
    code_hashes = {
        name: hashlib.sha256((Path(__file__).parents[1] / name).read_bytes()).hexdigest()
        for name in (
            "experiments/v2.py",
            "routing.py",
            "data/reasoning_v2.py",
            "reasoning_pipeline.py",
            "eval/reasoning_v2.py",
            "training/batching.py",
        )
    }
    preparation = json.loads((folder / "preparation.json").read_text())
    if (
        preparation["manifest_sha256"]
        != hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    ):
        raise ValueError("manifest changed since CPU preparation")
    if preparation.get("curriculum_version") != CURRICULUM_VERSION:
        raise ValueError("curriculum changed since CPU preparation; prepare the run again")
    failed, unfinished = [], []
    if stage in ("screen", "recover_screen"):
        summary_path = folder / "screen_summary.json"
        reports = (
            json.loads(summary_path.read_text())
            if stage == "recover_screen" and summary_path.exists()
            else []
        )
        previous = {r["name"]: r for r in reports}
        completed = {r["name"] for r in select_candidates(reports)}
        samples = [s for s, _ in curriculum("dev")]
        signature = dataset_signature(samples)
        candidates = manifest["candidates"]
        ready = sorted(
            [c for c in candidates if c["support"] == "builtin"],
            key=lambda c: c["total_parameters"],
        )
        ready = [c for c in ready if c["name"] not in completed]
        write_json(
            folder / "selection.json", {"split": "dev", "ranked": [], "status": "incomplete"}
        )
        for index, candidate in enumerate(ready):
            print(f"[screen] candidate={candidate['name']} {index + 1}/{len(ready)}", flush=True)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Equal per-candidate slices prevent the first slow model from
            # consuming the entire screen. Loading counts against its slice.
            candidate_deadline = min(deadline, time.monotonic() + remaining / (len(ready) - index))
            result = {
                "name": candidate["name"],
                "status": "incomplete",
                "gpu": gpu,
                "code_sha256": code_hashes,
            }
            model = tok = decision = None
            try:
                report_path = folder / "screen" / f"{candidate['name']}.json"
                saved = (
                    json.loads(report_path.read_text())
                    if stage == "recover_screen" and report_path.exists()
                    else {}
                )
                cached = (
                    saved.get("rows")
                    if saved.get("dataset_signature") == signature
                    or (saved.get("dataset_signature") is None and candidate["name"] in previous)
                    else None
                )
                model, tok = load_candidate(candidate)
                decision = controlled_decision(model, tok)
                evaluated = evaluate_efforts(
                    decision,
                    samples,
                    efforts=("low",),
                    deadline=candidate_deadline,
                    resume_rows=cached,
                    on_progress=partial(write_json, report_path),
                )
                write_json(folder / "screen" / f"{candidate['name']}.json", evaluated)
                result.update(evaluated["reports"].get("low", {}))
                result.update(name=candidate["name"], status=evaluated["status"])
                result["elapsed_s"] = time.monotonic() - start
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            print(
                f"[screen] candidate={candidate['name']} status={result['status']} n={result.get('n', 0)}",
                flush=True,
            )
            reports = [r for r in reports if r["name"] != candidate["name"]] + [result]
            if result["status"] != "complete":
                unfinished.append(candidate["name"])
            write_json(folder / "screen_summary.json", reports)
            write_json(
                folder / "selection.json",
                {
                    "split": "dev",
                    "ranked": [r["name"] for r in select_candidates(reports)],
                    "status": "incomplete",
                },
            )
            del decision, model, tok
            gc.collect()
            torch.cuda.empty_cache()
        ranked = select_candidates(reports)
        write_json(
            folder / "selection.json",
            {
                "split": "dev",
                "ranked": [r["name"] for r in ranked],
                "incomplete": [r["name"] for r in reports if r["status"] != "complete"],
                "survey_only": [c["name"] for c in candidates if c["support"] != "builtin"],
            },
        )
    else:
        selected = json.loads((folder / "selection.json").read_text())["ranked"]
        if not selected:
            write_json(
                folder / f"{stage}.json",
                {"status": "blocked", "reason": "no complete candidate screen"},
            )
            return
        by_name = {c["name"]: c for c in manifest["candidates"]}
        names = selected[:2] if stage in ("sft", "evaluate") else selected[:1]
        for index, name in enumerate(names):
            sub_deadline = min(
                deadline,
                time.monotonic() + max(0, deadline - time.monotonic()) / (len(names) - index),
            )
            model = tok = trainer = probe = decision = None
            try:
                model, tok = load_candidate(by_name[name])
                if stage == "heads":
                    comparisons = []
                    for readout, layers, aux in [
                        ("lm", 0, 0),
                        ("pointer", 0, 0),
                        ("hybrid", 2, 0),
                        ("hybrid", 2, 0.3),
                    ]:
                        if time.monotonic() >= sub_deadline:
                            break
                        cfg = replace(model.cfg, readout=readout, set_mixer_layers=layers)
                        torch.manual_seed(15)
                        probe = ElectraDecisionModel(cfg, model.backbone, model.text_config).cuda()
                        probe.backbone.requires_grad_(False)
                        train_items = [
                            it
                            for sample, _ in curriculum("train", 8)
                            for it in reasoning_items(sample, tok, cfg, {}, include_direct=True)
                        ]
                        trainer = Trainer(
                            probe,
                            tok,
                            TrainConfig(
                                steps=30,
                                bf16=True,
                                max_train_seconds=max(1, (sub_deadline - time.monotonic()) / 4),
                                loss_weights=LossWeights(pointer_aux=aux),
                            ),
                            "cuda",
                        )
                        history = (
                            trainer.train(training_stream(train_items), verbose=False)
                            if readout != "lm"
                            else []
                        )
                        report = evaluate_efforts(
                            controlled_decision(probe.eval(), tok),
                            [s for s, _ in curriculum("dev")],
                            efforts=(),
                            deadline=sub_deadline,
                        )
                        comparisons.append(
                            {
                                "readout": readout,
                                "set_mixer_layers": layers,
                                "pointer_aux": aux,
                                "last_step": history[-1] if history else None,
                                "evaluation": report,
                            }
                        )
                        write_json(folder / "heads.json", comparisons)
                        if report["status"] != "complete" or trainer.stopped_early:
                            unfinished.append(name)
                        trainer = probe = None
                    head_reports = [
                        {
                            **entry["evaluation"]["reports"].get("off", {}),
                            "name": str(i),
                            "status": entry["evaluation"]["status"],
                        }
                        for i, entry in enumerate(comparisons)
                    ]
                    ranked = select_candidates(head_reports)
                    if ranked:
                        best_head = comparisons[int(ranked[0]["name"])]
                        write_json(
                            folder / "head_selection.json",
                            {
                                "candidate": name,
                                "readout": best_head["readout"],
                                "set_mixer_layers": best_head["set_mixer_layers"],
                                "pointer_aux": best_head["pointer_aux"],
                                "split": "dev",
                            },
                        )
                elif stage == "sft":
                    pointer_aux = 0.3
                    head_choice = folder / "head_selection.json"
                    if head_choice.exists():
                        head = json.loads(head_choice.read_text())
                        if head["candidate"] == name:
                            pointer_aux = head["pointer_aux"]
                            cfg = replace(
                                model.cfg,
                                readout=head["readout"],
                                set_mixer_layers=head["set_mixer_layers"],
                            )
                            model = ElectraDecisionModel(
                                cfg, model.backbone, model.text_config
                            ).cuda()
                    model.backbone.requires_grad_(False)
                    apply_lora(model)
                    items = [
                        it
                        for sample, traces in curriculum("train", 128)
                        for it in reasoning_items(sample, tok, model.cfg, traces)
                    ]
                    trainer = Trainer(
                        model,
                        tok,
                        TrainConfig(
                            steps=500,
                            bf16=True,
                            questions_per_step=8,
                            loss_weights=LossWeights(
                                pointer_aux=0 if model.cfg.readout == "lm" else pointer_aux
                            ),
                            max_train_seconds=max(1, sub_deadline - time.monotonic() - 30),
                        ),
                        "cuda",
                    )
                    history = trainer.train(training_stream(items))
                    if trainer.stopped_early:
                        unfinished.append(name)
                    checkpoint = folder / "sft" / name
                    save_checkpoint(
                        model,
                        str(checkpoint),
                        {
                            "history_last": history[-1],
                            "status": "incomplete" if trainer.stopped_early else "complete",
                            "gpu": gpu,
                        },
                    )
                    write_json(folder / "sft" / f"{name}.json", history)
                    trainer = None
                elif stage == "evaluate":
                    checkpoint = folder / "sft" / name
                    model = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    model = load_checkpoint(str(checkpoint), device="cuda", merge=False).eval()
                    decision = controlled_decision(model, tok)
                    from ..data.schema import Sample

                    natural = [
                        Sample.from_json(s)
                        for s in json.loads((folder / "natural_test.json").read_text())
                    ]
                    if natural:
                        write_json(
                            folder / "eval" / name / "natural_direct.json",
                            evaluate_efforts(decision, natural, efforts=(), deadline=sub_deadline),
                        )
                    paired = {}
                    for split in ("router_train", "dev", "calibration"):
                        paired[split] = evaluate_efforts(
                            decision, [s for s, _ in curriculum(split)], deadline=sub_deadline
                        )
                        write_json(folder / "eval" / name / f"{split}.json", paired[split])
                    if any(r["status"] != "complete" for r in paired.values()):
                        unfinished.append(name)
                    if all(r["status"] == "complete" for r in paired.values()):
                        train_pairs = [
                            row
                            for effort in ("low", "medium", "high")
                            for row in paired_training_rows(
                                paired["router_train"]["rows"]["off"],
                                paired["router_train"]["rows"][effort],
                                "router_train",
                            )
                        ]
                        dev_pairs = [
                            row
                            for effort in ("low", "medium", "high")
                            for row in paired_training_rows(
                                paired["dev"]["rows"]["off"], paired["dev"]["rows"][effort], "dev"
                            )
                        ]
                        router = fit_router(train_pairs, dev_pairs)
                        router.save(folder / "eval" / name / "router.json")
                        calibration = PathCalibration.fit(
                            [r for rs in paired["calibration"]["rows"].values() for r in rs]
                        )
                        write_json(
                            folder / "eval" / name / "calibration.json", calibration.temperatures
                        )
                        decision.router = router if router.promoted else None
                        decision.calibration = calibration
                        test = evaluate_efforts(
                            decision,
                            [s for s, _ in curriculum("test")],
                            deadline=sub_deadline,
                            include_auto=True,
                        )
                        write_json(folder / "eval" / name / "test.json", test)
                        if test["status"] != "complete":
                            unfinished.append(name)
                        # Basic repository-authored EN/KO/JA semantic regression;
                        # separate from broad multilingual benchmark claims.
                        from ..data.schema import Question, Sample

                        regressions = [
                            Sample(
                                text,
                                [Question.noul("greeting", instruction, 1)],
                                {
                                    "source_example_id": language,
                                    "language": language,
                                    "split": "test",
                                },
                            )
                            for text, instruction, language in [
                                ("The order has shipped.", "Has the order shipped?", "en"),
                                ("주문이 발송되었습니다.", "주문이 발송되었나요?", "ko"),
                                ("注文は発送済みです。", "注文は発送済みですか？", "ja"),
                            ]
                        ]
                        write_json(
                            folder / "eval" / name / "language_regression.json",
                            evaluate_efforts(
                                decision, regressions, efforts=(), deadline=sub_deadline
                            ),
                        )
                    decision = None
                elif stage == "reserve":
                    rerun = evaluate_efforts(
                        controlled_decision(model, tok),
                        [s for s, _ in curriculum("dev")],
                        efforts=("low",),
                        deadline=sub_deadline,
                    )
                    write_json(
                        folder / "reproduction.json", {"candidate": name, "evaluation": rerun}
                    )
                    if rerun["status"] != "complete":
                        unfinished.append(name)
            except Exception as exc:
                failed.append(name)
                write_json(
                    folder / f"{stage}-{name}-failure.json",
                    {
                        "status": "failed",
                        "candidate": name,
                        "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                    },
                )
            finally:
                model = tok = trainer = probe = decision = None
                gc.collect()
                torch.cuda.empty_cache()
    write_json(
        folder / f"{stage}_status.json",
        {
            "status": "complete"
            if time.monotonic() < deadline and not failed and not unfinished
            else "incomplete",
            "failed_candidates": failed,
            "unfinished_candidates": unfinished,
            "elapsed_s": time.monotonic() - start,
            "gpu": gpu,
            "code_sha256": code_hashes,
        },
    )


def bounded_run(
    manifest,
    out,
    *,
    stages=("screen", "heads", "sft", "evaluate", "reserve"),
    scale=1.0,
    stage_limits=None,
):
    gpu_info()
    if not 0 < scale <= 1:
        raise ValueError("budget scale must be in (0, 1]")
    stage_limits = stage_limits or {}
    for stage, seconds in stage_limits.items():
        if stage not in STAGE_SECONDS or not 0 < seconds <= STAGE_SECONDS[stage]:
            raise ValueError("stage limits must be positive and within the planned caps")
    ledger_path = Path(out) / "budget.json"
    ledger = (
        json.loads(ledger_path.read_text())
        if ledger_path.exists()
        else {"elapsed_s": 0, "stages": []}
    )
    # A new invocation cannot share a live worker. Retain the whole prior
    # reservation, including interrupted startup and elapsed GPU overhead.
    for entry in ledger["stages"]:
        if entry["status"] == "running":
            entry.update(status="interrupted", reason="previous invocation ended before completion")
    for stage in stages:
        if stage not in STAGE_SECONDS:
            raise ValueError("unknown budget stage")
        remaining = (
            TOTAL_SECONDS - ledger["elapsed_s"] - 120
        )  # container startup/final flush reserve
        limit = min(stage_limits.get(stage, STAGE_SECONDS[stage]) * scale, remaining)
        family = ("screen", "recover_screen") if stage in ("screen", "recover_screen") else (stage,)
        family_cap = (
            STAGE_SECONDS["screen"]
            if stage in ("screen", "recover_screen")
            else STAGE_SECONDS[stage]
        )
        spent = sum(
            entry.get("charged_s", entry["allocation_s"])
            for entry in ledger["stages"]
            if entry["stage"] in family
        )
        limit = min(limit, family_cap - spent)
        if stage in ("recover_screen", "reserve"):
            recovery_spent = sum(
                entry.get("charged_s", entry["allocation_s"])
                for entry in ledger["stages"]
                if entry["stage"] in ("recover_screen", "reserve")
            )
            limit = min(limit, 3600 - recovery_spent)
        if limit <= 0:
            ledger.setdefault("skipped_stages", []).append(
                {
                    "stage": stage,
                    "reason": "total_budget" if remaining <= 0 else "cumulative_stage_cap",
                }
            )
            write_json(ledger_path, ledger)
            if remaining <= 0:
                break
            continue
        started = time.monotonic()
        # Reserve before spawning. Unverified crashes retain the reservation;
        # only explicit externally observed closed-window reconciliation can
        # release unused time, without erasing consumed GPU time or overruns.
        entry = {"stage": stage, "allocation_s": limit, "status": "running"}
        ledger["elapsed_s"] += limit
        ledger["stages"].append(entry)
        write_json(ledger_path, ledger)
        command = [
            sys.executable,
            "-m",
            "ayaka.experiments.v2",
            "worker",
            "--manifest",
            manifest,
            "--out",
            out,
            "--stage",
            stage,
            "--seconds",
            str(max(1, limit - 10)),
        ]
        try:
            result = subprocess.run(command, timeout=limit, check=False)
            entry["status"] = "complete" if result.returncode == 0 else "failed"
            entry["returncode"] = result.returncode
            status_path = Path(out) / f"{stage}_status.json"
            if result.returncode == 0 and status_path.exists():
                entry["status"] = json.loads(status_path.read_text())["status"]
        except subprocess.TimeoutExpired:
            entry["status"] = "incomplete"
        elapsed = time.monotonic() - started
        ledger["elapsed_s"] += max(0, elapsed - limit)
        entry["actual_elapsed_s"] = elapsed
        write_json(ledger_path, ledger)
    return ledger


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "run", "worker"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--download-weights", action="store_true")
    ap.add_argument("--stage", choices=STAGE_SECONDS, default="screen")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--budget-scale", type=float, default=1)
    args = ap.parse_args(argv)
    if args.command == "prepare":
        prepare(args.manifest, args.out, args.download_weights)
    elif args.command == "run":
        bounded_run(args.manifest, args.out, scale=args.budget_scale)
    else:
        worker(args.manifest, args.out, args.stage, args.seconds)


if __name__ == "__main__":
    main()
