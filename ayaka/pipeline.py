"""Provider-agnostic pipeline CLI (RunPod / vast.ai / Modal / any GPU box).

    python -m ayaka.pipeline train   --model electra-small --run small-v1 [--set steps=2000 ...]
    python -m ayaka.pipeline teacher --ckpt runs/large-v1/checkpoint --out runs/large-v1/teacher.jsonl
    python -m ayaka.pipeline train   --model electra-small --run small-distill --set teacher_labels=runs/large-v1/teacher.jsonl
    python -m ayaka.pipeline export  --ckpt runs/small-distill/checkpoint --name electra-small
    python -m ayaka.pipeline eval    --model electra-small --zero-shot

``--set key=value`` overrides any RunConfig field; values are parsed as
JSON when possible (``steps=2000``, ``liger=true``, ``specs=["a","b"]``),
otherwise kept as strings. Artifacts go to ``$AYAKA_ARTIFACTS`` (default
``./runs``); Hugging Face downloads are cached under ``$HF_HOME``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields


def _artifacts() -> str:
    return os.environ.get("AYAKA_ARTIFACTS", "runs")


def parse_sets(pairs: list[str]) -> dict:
    from .training.run import RunConfig

    known = {f.name for f in fields(RunConfig)}
    out = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or key not in known:
            raise SystemExit(
                f"--set {pair!r}: expected <RunConfig field>=<value>; fields: {sorted(known)}"
            )
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    if isinstance(out.get("specs"), str):
        out["specs"] = [s for s in out["specs"].split(",") if s]
    return out


def cmd_train(args) -> dict:
    from .training.run import RunConfig, run_training

    overrides = parse_sets(args.set)
    cfg = RunConfig(
        model_size=args.model, run_name=args.run, artifacts_dir=_artifacts(), **overrides
    )
    return run_training(cfg)


def cmd_teacher(args) -> dict:
    from .training.distill import label_with_teacher
    from .training.run import DEFAULT_SPECS, RELEASE_SPECS

    base = RELEASE_SPECS if args.release else DEFAULT_SPECS
    specs = [s for s in args.specs.split(",") if s] if args.specs else base
    if args.release:
        specs = [s for s in specs if s in RELEASE_SPECS]
    return label_with_teacher(
        args.ckpt,
        args.out,
        specs,
        args.n_samples,
        spec_limits={"jev_distill": 200_000},
        seed=args.seed,
    )


def cmd_export(args) -> dict:
    from .export import main as export_main

    argv = [
        "--ckpt",
        args.ckpt,
        "--out",
        args.out or os.path.join(_artifacts(), "exports"),
        "--name",
        args.name,
    ]
    if args.no_int8:
        argv.append("--no-int8")
    argv += ["--parity", str(args.parity), "--code-url", args.code_url]
    return export_main(argv)


def cmd_eval(args) -> dict:
    """JevBench public tiers (+ Jev fidelity) for a checkpoint or zero-shot backbone."""
    import torch

    from .checkpoint import load_checkpoint
    from .config import model_config
    from .data.decontam import Decontaminator
    from .eval.jevbench import run_jevbench
    from .model.electra import ElectraDecisionModel
    from .tokenization import HFTokenizer
    from .training.calibrate import apply_temperatures, fit_temperatures
    from .training.run import items_from_spec
    from .training.trainer import TrainConfig, Trainer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    if args.zero_shot:
        model = ElectraDecisionModel.from_config(model_config(args.model), dtype=dtype, device=dev)
    else:
        model = load_checkpoint(args.ckpt, device=dev, dtype=dtype)
    model.requires_grad_(False)
    tok = HFTokenizer.from_pretrained(model.cfg.backbone)
    tr = Trainer(model, tok, TrainConfig(steps=1), dev)
    decon = Decontaminator.from_jevbench()
    name = args.name or (
        f"{args.model}-zeroshot" if args.zero_shot else os.path.basename(os.path.dirname(args.ckpt))
    )
    out_dir = os.path.join(_artifacts(), "evals")
    os.makedirs(out_dir, exist_ok=True)
    report: dict = {"model": args.model, "ckpt": args.ckpt, "zero_shot": args.zero_shot}
    if args.zero_shot and args.calibrate:
        cal = items_from_spec("jev_distill_calibration", 3000, tok, model.cfg, 0, decon)
        _, logits = tr.predict(cal, apply_temperature=False, return_logits=True)
        report["temperatures"] = fit_temperatures(
            logits, [it.target for it in cal], [it.type for it in cal]
        )
        apply_temperatures(model, report["temperatures"])
    if args.fidelity:
        fid = items_from_spec("jev_distill_test30k", args.fidelity, tok, model.cfg, 0, decon)
        report["jev_fidelity"] = tr.evaluate(fid)
    report["jevbench"] = run_jevbench(
        model, tok, out_path=os.path.join(out_dir, f"{name}-jevbench.json")
    )["summary"]
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(
        prog="python -m ayaka.pipeline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train (or distill with --set teacher_labels=...)")
    t.add_argument(
        "--model", required=True, choices=["electra-small", "electra-base", "electra-large", "tiny"]
    )
    t.add_argument("--run", required=True, help="run name (artifacts/<run>)")
    t.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    t.set_defaults(fn=cmd_train)

    te = sub.add_parser("teacher", help="label a training pool with a trained (Large) checkpoint")
    te.add_argument("--ckpt", required=True)
    te.add_argument("--out", required=True)
    te.add_argument("--n-samples", type=int, default=120_000)
    te.add_argument("--specs", default="")
    te.add_argument("--seed", type=int, default=1)
    te.add_argument("--release", action="store_true", help="label only release-licensed data")
    te.set_defaults(fn=cmd_teacher)

    ex = sub.add_parser("export", help="write bf16 + int8 model folders")
    ex.add_argument("--ckpt", required=True)
    ex.add_argument("--name", required=True)
    ex.add_argument("--out", default="")
    ex.add_argument("--no-int8", action="store_true")
    ex.add_argument("--parity", type=int, default=256)
    ex.add_argument(
        "--code-url", default="<code-url>", help="git URL for pip install in the model card"
    )
    ex.set_defaults(fn=cmd_export)

    ev = sub.add_parser("eval", help="JevBench public tiers + Jev fidelity")
    ev.add_argument("--model", default="electra-small")
    g = ev.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt")
    g.add_argument("--zero-shot", action="store_true")
    ev.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True)
    ev.add_argument("--fidelity", type=int, default=2000)
    ev.add_argument("--name", default="")
    ev.set_defaults(fn=cmd_eval)

    args = ap.parse_args(argv)
    result = args.fn(args)
    print(json.dumps(result, indent=2, default=str)[:4000], flush=True)
    return result


if __name__ == "__main__":
    main()
