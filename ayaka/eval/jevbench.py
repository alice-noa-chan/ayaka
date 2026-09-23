"""JevBench evaluator (https://github.com/fstandhartinger/jevbench).

Maps public-tier JSONL records onto the Decision primitive API:

- ``choice``: ``question.criteria`` is a {label: description} mapping;
  candidate descriptions come from criteria, labels stay exact.
- ``noul``:   labels are ["no", "yes"]; criteria carries {"false", "true"}
  descriptions used as the implicit pair's candidate text.
- ``score``:  ``criteria`` is an ordered list of level descriptions;
  labels are ordinal strings ("0".."k").

The answer is never leaked into the input: only the state, instruction,
and candidate *descriptions* reach the model; labels are attached back
to positions afterward, so ``probs`` covers every exact option.

Usage::

    python -m ayaka.eval.jevbench \
        --ckpt artifacts/run/checkpoint_final.pt \
        --tokenizer artifacts/run/tokenizer.json \
        --data jevbench/datasets/public --out report.json
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import torch

from ..config import MODEL_FAMILY, ElectraConfig, tiny_config
from ..model.model import ElectraDecisionModel
from ..primitives import Decision, QuestionSpec
from ..tokenizer import HashTokenizer, Tokenizer, load_bpe

TIERS = ("easy", "original", "hard")


@dataclass
class BenchItem:
    """One jevbench record normalized for the Decision API."""

    spec: QuestionSpec
    labels: list[str]  # exact option labels, aligned with spec.candidates
    expected: str
    family: str
    state: object


def _instruction(q: dict) -> str:
    return q.get("instructions") or q.get("instruction") or ""


def record_to_item(rec: dict) -> BenchItem:
    q = rec["question"]
    labels = list(rec["labels"])
    criteria = q.get("criteria") or {}
    ptype = q["type"]
    if ptype == "noul":
        # implicit false/true pair -> map back onto no/yes labels
        cands = [criteria.get("false", "false"), criteria.get("true", "true")]
        spec = QuestionSpec("noul", _instruction(q), cands)
    elif ptype == "score":
        if isinstance(criteria, list):
            cands = [str(c) for c in criteria]
        else:
            cands = [str(criteria.get(lb, lb)) for lb in labels]
        if len(cands) != len(labels):
            # criteria shorter/longer than labels: pad with the label
            cands = (cands + labels)[: len(labels)]
        ordinals = [int(lb) if str(lb).lstrip("-").isdigit() else i for i, lb in enumerate(labels)]
        spec = QuestionSpec("score", _instruction(q), cands, ordinals=ordinals)
    else:  # choice
        spec = QuestionSpec("choice", _instruction(q), [str(criteria.get(lb, lb)) for lb in labels])
    return BenchItem(
        spec=spec,
        labels=labels,
        expected=str(rec["expected"]),
        family=rec.get("family", "?"),
        state=rec.get("state"),
    )


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _ece(confs: list[float], correct: list[bool], n_bins: int = 10) -> float:
    ece = 0.0
    n = len(confs)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confs) if lo <= c < hi or (b == n_bins - 1 and c == hi)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            conf = sum(confs[i] for i in idx) / len(idx)
            ece += len(idx) / n * abs(acc - conf)
    return ece


def evaluate(
    decision: Decision,
    records: list[dict],
    device=None,
) -> dict:
    """Run records through the model; return accuracy + calibration
    metrics + per-record detail. probs_source is always "model"."""
    items = [record_to_item(r) for r in records]
    results = []
    n_correct = 0
    brier_sum = 0.0
    confs: list[float] = []
    correct: list[bool] = []
    lat: list[float] = []
    by_family: dict[str, list[bool]] = {}
    by_type: dict[str, list[bool]] = {}
    for item in items:
        t0 = time.perf_counter()
        res = decision.decide(item.state, [item.spec], device=device)[0]
        lat.append(time.perf_counter() - t0)
        # positional: distribution keys are candidate descriptions
        p = list(res.distribution.values())
        probs = {lb: p[i] for i, lb in enumerate(item.labels) if i < len(p)}
        pred = max(probs, key=probs.get) if probs else ""
        ok = pred == item.expected
        n_correct += ok
        brier_sum += sum(
            (probs.get(lb, 0.0) - (1.0 if lb == item.expected else 0.0)) ** 2 for lb in item.labels
        )
        confs.append(probs.get(pred, 0.0))
        correct.append(ok)
        by_family.setdefault(item.family, []).append(ok)
        by_type.setdefault(item.spec.type, []).append(ok)
        results.append(
            {
                "expected": item.expected,
                "pred": pred,
                "ok": ok,
                "probs": probs,
                "probs_source": "model",
                "latency_s": lat[-1],
                "family": item.family,
                "type": item.spec.type,
            }
        )
    n = len(items) or 1
    return {
        "n": len(items),
        "accuracy": n_correct / n,
        "brier": brier_sum / n,
        "ece": _ece(confs, correct) if items else 0.0,
        "latency_mean_s": sum(lat) / n,
        "by_family": {k: sum(v) / len(v) for k, v in by_family.items()},
        "by_type": {k: sum(v) / len(v) for k, v in by_type.items()},
        "results": results,
    }


def load_model(
    ckpt_path: str,
    tokenizer_path: str = "",
    device: str = "cpu",
) -> tuple[ElectraDecisionModel, Tokenizer, dict]:
    """Rebuild the model from a run checkpoint. The run's
    ``run_config.json`` (same dir) supplies model_size; tokenizer falls
    back to a sibling tokenizer.json, then HashTokenizer."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    art = os.path.dirname(ckpt_path)
    cfg_path = os.path.join(art, "run_config.json")
    model_size = "electra-small"
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            model_size = json.load(f).get("model_size", model_size)
    if model_size == "tiny":
        mcfg: ElectraConfig = tiny_config()
    else:
        mcfg = MODEL_FAMILY[model_size]
    model = ElectraDecisionModel(mcfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    if not tokenizer_path:
        cand = os.path.join(art, "tokenizer.json")
        tokenizer_path = cand if os.path.exists(cand) else ""
    tok = load_bpe(tokenizer_path) if tokenizer_path else HashTokenizer(mcfg.vocab_size)
    return model, tok, ckpt


def main(argv: list[str] | None = None) -> dict:
    import argparse

    ap = argparse.ArgumentParser(description="JevBench public-tier evaluator")
    ap.add_argument("--ckpt", required=True, help="checkpoint_final.pt path")
    ap.add_argument("--tokenizer", default="", help="tokenizer.json path")
    ap.add_argument("--data", required=True, help="dir with easy/original/hard.jsonl")
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--limit", type=int, default=0, help="per-tier record cap (0=all)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="", help="write full report JSON here")
    args = ap.parse_args(argv)

    model, tok, _ = load_model(args.ckpt, args.tokenizer, args.device)
    decision = Decision(model, tok)
    report = {"checkpoint": args.ckpt, "tokenizer": args.tokenizer or "hash", "tiers": {}}
    for tier in [t for t in args.tiers.split(",") if t]:
        path = os.path.join(args.data, f"{tier}.jsonl")
        if not os.path.exists(path):
            print(f"[jevbench] {tier}: missing {path}, skipped", flush=True)
            continue
        records = load_jsonl(path)
        if args.limit:
            records = records[: args.limit]
        metrics = evaluate(decision, records, device=torch.device(args.device))
        report["tiers"][tier] = metrics
        fams = " ".join(f"{k}={v:.2f}" for k, v in sorted(metrics["by_family"].items()))
        print(
            f"[jevbench] {tier}: n={metrics['n']} acc={metrics['accuracy']:.3f} "
            f"brier={metrics['brier']:.3f} ece={metrics['ece']:.3f} "
            f"lat={metrics['latency_mean_s'] * 1000:.0f}ms | {fams}",
            flush=True,
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[jevbench] report -> {args.out}", flush=True)
    return report


if __name__ == "__main__":
    main()
