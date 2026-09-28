"""JevBench public-tier evaluator (https://github.com/fstandhartinger/jevbench).

Maps public JSONL records onto the Decision API (answers never reach
the model — only state, instruction and option descriptions):

- ``choice``: ``criteria`` {label: description}; candidates are the
  descriptions, labels are re-attached by position.
- ``noul``:   labels no/yes; criteria {"false", "true"} descriptions.
- ``score``:  ``criteria`` is the ordered level list; labels "0".."k".

Scores are compared item-for-item with published per-task outcomes of
reference systems (Jev 1.13.0 and others) on the same public items.
The Intelligence proxy follows JevBench's chance-corrected formula
with the official tier weights renormalized over the public tiers
(easy 14 / standard 28 / hard 30; the judge tier has no public items).

Usage::

    python -m ayaka.eval.jevbench --ckpt artifacts/<run>/checkpoint --out report.json
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from importlib import resources

import torch

from ..primitives import Decision, QuestionSpec

TIERS = ("easy", "original", "hard")
TIER_WEIGHTS = {"easy": 14, "original": 28, "hard": 30}
NO_LABELS = {"no", "false"}

# JevBench v1.4.2.1 leaderboard axes (results/v1.4.2.1/*-results.json, "scoring").
# Self-hosted latency is adjusted x2 + 0.15 s by the benchmark; Speed is measured
# on the standard(+judge) run, whose public counterpart here is "original".
SELF_HOSTED_LATENCY = (2.0, 0.15)
SPEED_TIER = "original"
# Cost is estimated by the benchmark from a hosted list price of the base model
# for self-hosted systems. Values are the board's own estimates for systems on
# the same backbones (system-one-open on E2B, Winnow-12B on 12B); None = unknown.
BACKBONE_USD_PER_1000 = {
    "google/gemma-4-E2B-it": 0.01488,
    "google/gemma-4-E4B-it": None,
    "google/gemma-4-12B-it": 0.03709,
}


def data_dir() -> str:
    return str(resources.files("ayaka.eval") / "data" / "jevbench_public")


def reference_outcomes() -> dict:
    with open(
        resources.files("ayaka.eval") / "data" / "reference_public_outcomes.json", encoding="utf-8"
    ) as f:
        return json.load(f)["systems"]


@dataclass
class BenchItem:
    id: str
    spec: QuestionSpec
    labels: list[str]  # exact option labels aligned with spec.candidates
    expected: str
    family: str
    state: object


def record_to_item(rec: dict) -> BenchItem:
    q = rec["question"]
    labels = [str(lb) for lb in rec["labels"]]
    criteria = q.get("criteria") or {}
    instruction = q.get("instructions") or q.get("instruction") or ""
    ptype = q["type"]
    if ptype == "noul":
        if len(labels) != 2:
            raise ValueError(f"noul item {rec.get('id')} needs two labels")
        false_label = next((lb for lb in labels if lb.lower() in NO_LABELS), labels[0])
        true_label = next(lb for lb in labels if lb != false_label)
        labels = [false_label, true_label]
        spec = QuestionSpec(
            "noul",
            instruction,
            [str(criteria.get("false", "false")), str(criteria.get("true", "true"))],
        )
    elif ptype == "score":
        if isinstance(criteria, list):
            cands = [str(c) for c in criteria]
        else:
            cands = [str(criteria.get(lb, lb)) for lb in labels]
        cands = (cands + labels)[: len(labels)]
        ordinals = [int(lb) if lb.lstrip("-").isdigit() else i for i, lb in enumerate(labels)]
        spec = QuestionSpec("score", instruction, cands, ordinals=ordinals)
    else:
        spec = QuestionSpec("choice", instruction, [str(criteria.get(lb, lb)) for lb in labels])
    return BenchItem(
        rec.get("id", ""),
        spec,
        labels,
        str(rec["expected"]),
        rec.get("family", "?"),
        rec.get("state"),
    )


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(decision: Decision, records: list[dict], device=None) -> dict:
    items = [record_to_item(r) for r in records]
    results, confs, correct, lat = [], [], [], []
    brier = 0.0
    chance = 0.0
    by_family: dict[str, list[bool]] = {}
    by_type: dict[str, list[bool]] = {}
    for item in items:
        t0 = time.perf_counter()
        res = decision.decide(item.state, [item.spec], device=device)[0]
        lat.append(time.perf_counter() - t0)
        probs = dict(zip(item.labels, res.probs, strict=True))
        pred = max(probs, key=probs.get)
        ok = pred == item.expected
        brier += sum((probs[lb] - (lb == item.expected)) ** 2 for lb in item.labels)
        chance += 1 / len(item.labels)
        confs.append(probs[pred])
        correct.append(ok)
        by_family.setdefault(item.family, []).append(ok)
        by_type.setdefault(item.spec.type, []).append(ok)
        results.append(
            {
                "id": item.id,
                "expected": item.expected,
                "pred": pred,
                "ok": ok,
                "probs": probs,
                "latency_s": lat[-1],
                "family": item.family,
                "type": item.spec.type,
            }
        )
    n = max(len(items), 1)
    acc = sum(correct) / n
    ch = chance / n
    return {
        "n": len(items),
        "accuracy": acc,
        "chance": ch,
        "chance_corrected": max(0.0, (acc - ch) / (1 - ch)),
        "brier": brier / n,
        "ece": _ece(confs, correct),
        "latency_p50_s": _quantile(lat, 0.50),
        "latency_p95_s": _quantile(lat, 0.95),
        "by_family": {k: sum(v) / len(v) for k, v in by_family.items()},
        "by_type": {k: sum(v) / len(v) for k, v in by_type.items()},
        "results": results,
    }


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def speed_score(seconds: float) -> float:
    """100 - 20 log10(s / 0.1 s), clipped to 0..100 (0.1 s = 100, 1 s = 80)."""
    if seconds <= 0:
        return 100.0
    return min(100.0, max(0.0, 100 - 20 * math.log10(seconds / 0.1)))


def speed_axis(p50_s: float, p95_s: float, self_hosted: bool = True) -> float:
    """Mean of score(p50) and score(p95), after the self-hosted adjustment."""
    if self_hosted:
        scale, offset = SELF_HOSTED_LATENCY
        p50_s, p95_s = p50_s * scale + offset, p95_s * scale + offset
    return (speed_score(p50_s) + speed_score(p95_s)) / 2


def cost_axis(usd_per_1000: float) -> float:
    """100 - 30 log10($ per 1,000 decisions / $0.001), clipped to 0..100."""
    if usd_per_1000 <= 0:
        return 100.0
    return min(100.0, max(0.0, 100 - 30 * math.log10(usd_per_1000 / 0.001)))


def jevbench_score(intelligence: float, calibration: float, speed: float, cost: float) -> float:
    """Composite: equal-weight harmonic mean of the four axes, times
    (axis / 50)^2 for Intelligence, Speed and Cost each when below 50.
    Calibration has no penalty term; a label-only system scores 0 there."""
    axes = (intelligence, calibration, speed, cost)
    if min(axes) <= 0:
        return 0.0
    score = len(axes) / sum(1 / a for a in axes)
    for axis in (intelligence, speed, cost):
        if axis < 50:
            score *= (axis / 50) ** 2
    return score


def leaderboard_estimate(
    report_tiers: dict[str, dict],
    backbone: str | None = None,
    usd_per_1000: float | None = None,
    calibration: float | None = None,
) -> dict:
    """Estimate JevBench axes from a public-tier run.

    Only public tiers are available, so Intelligence is the public proxy (no
    judge tier, no sealed hard set, no public-minus-sealed gap penalty) and is
    optimistic. Latency is measured in-process, not over HTTP. Calibration
    uses the benchmark's own procedure, which is not reproduced here, so the
    composite is given only when a Calibration value is supplied.
    """
    acc = {t: m["accuracy"] for t, m in report_tiers.items()}
    chance = {t: m["chance"] for t, m in report_tiers.items()}
    out: dict = {"intelligence_proxy": intelligence_proxy(acc, chance)}
    timing = report_tiers.get(SPEED_TIER)
    if timing is None:
        lat = [r["latency_s"] for m in report_tiers.values() for r in m.get("results", [])]
        timing = {"latency_p50_s": _quantile(lat, 0.5), "latency_p95_s": _quantile(lat, 0.95)}
    out["speed"] = speed_axis(timing["latency_p50_s"], timing.get("latency_p95_s", 0.0))
    if usd_per_1000 is None and backbone is not None:
        usd_per_1000 = BACKBONE_USD_PER_1000.get(backbone)
    out["usd_per_1000"] = usd_per_1000
    out["cost"] = cost_axis(usd_per_1000) if usd_per_1000 is not None else None
    out["calibration"] = calibration
    out["jevbench_score"] = (
        jevbench_score(out["intelligence_proxy"], calibration, out["speed"], out["cost"])
        if calibration is not None and out["cost"] is not None
        else None
    )
    return out


def _ece(confs: list[float], correct: list[bool], n_bins: int = 10) -> float:
    out = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confs) if lo <= c < hi or (b == n_bins - 1 and c == hi)]
        if idx:
            out += (
                len(idx)
                / len(confs)
                * abs(
                    sum(correct[i] for i in idx) / len(idx) - sum(confs[i] for i in idx) / len(idx)
                )
            )
    return out


def intelligence_proxy(tier_acc: dict[str, float], tier_chance: dict[str, float]) -> float:
    tot = w_sum = 0.0
    for t, w in TIER_WEIGHTS.items():
        if t in tier_acc:
            cc = max(0.0, (tier_acc[t] - tier_chance[t]) / (1 - tier_chance[t]))
            tot += w * cc
            w_sum += w
    return 100 * tot / w_sum if w_sum else 0.0


def compare_references(tiers: dict[str, dict]) -> dict:
    """Reference systems on exactly the items we evaluated."""
    refs = reference_outcomes()
    chance = {t: m["chance"] for t, m in tiers.items()}
    out = {}
    for key, sysd in refs.items():
        acc = {}
        for t, m in tiers.items():
            oc = [sysd["outcomes"].get(r["id"]) for r in m["results"]]
            oc = [o for o in oc if o is not None]
            if oc:
                acc[t] = sum(o == "c" for o in oc) / len(oc)
        out[key] = {
            "display": sysd["display"],
            "accuracy": acc,
            "intelligence_proxy": intelligence_proxy(acc, chance),
        }
    return out


class EndpointDecision:
    """Decision-compatible client for a running ``/v1/systemone`` server.

    Lets ``evaluate`` score the real serving path, latency included (HTTP and
    JSON). Options are sent as positional labels; probabilities come back in
    candidate order.
    """

    def __init__(self, url: str, timeout: float = 300.0):
        self.url = url.rstrip("/")
        if not self.url.endswith("/v1/systemone"):
            self.url += "/v1/systemone"
        self.timeout = timeout

    def _question(self, spec):
        if spec.type == "noul":
            crit = {"false": spec.candidates[0], "true": spec.candidates[1]}
        elif spec.type == "score":
            crit = {str(o): c for o, c in zip(spec.ordinals, spec.candidates, strict=True)}
        else:
            crit = {str(i): c for i, c in enumerate(spec.candidates)}
        return {"type": spec.type, "instructions": spec.instruction, "criteria": crit}

    def decide(self, state, questions, device=None):
        import urllib.request

        from ..primitives import DecisionResult

        body = {
            "state": state,
            "questions": {f"q{i}": self._question(q) for i, q in enumerate(questions)},
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())
        results = []
        for i, spec in enumerate(questions):
            ans = out["answers"][f"q{i}"]
            if spec.type == "noul":
                probs = [1 - ans["noul"], ans["noul"]]
            elif spec.type == "score":
                probs = [ans["probabilities"][str(o)] for o in spec.ordinals]
            else:
                probs = [ans["probabilities"][str(j)] for j in range(len(spec.candidates))]
            results.append(
                DecisionResult(spec.type, probs, dict(zip(spec.candidates, probs, strict=True)))
            )
        return results


def run_jevbench(
    model,
    tok,
    out_path: str = "",
    tiers=TIERS,
    limit: int = 0,
    data: str | None = None,
    verbose: bool = True,
    max_seq_len: int | None = None,
    decision=None,
) -> dict:
    """``decision`` overrides the plain Decision (evidence route, HTTP endpoint)."""
    decision = decision or Decision(model, tok, max_seq_len=max_seq_len)
    dev = next(model.parameters()).device if model is not None else None
    report: dict = {"tiers": {}}
    for tier in tiers:
        path = os.path.join(data or data_dir(), f"{tier}.jsonl")
        records = load_jsonl(path)[: limit or None]
        m = evaluate(decision, records, device=dev)
        report["tiers"][tier] = m
        if verbose:
            fams = " ".join(f"{k}={v:.2f}" for k, v in sorted(m["by_family"].items()))
            print(
                f"[jevbench] {tier}: n={m['n']} acc={m['accuracy']:.3f} (chance {m['chance']:.2f}) brier={m['brier']:.3f} ece={m['ece']:.3f} p50={m['latency_p50_s'] * 1000:.0f}ms | {fams}",
                flush=True,
            )
    acc = {t: m["accuracy"] for t, m in report["tiers"].items()}
    chance = {t: m["chance"] for t, m in report["tiers"].items()}
    refs = compare_references(report["tiers"])
    backbone = getattr(getattr(model, "cfg", None), "backbone", None)
    report["summary"] = {
        "accuracy": acc,
        "intelligence_proxy": intelligence_proxy(acc, chance),
        "hard_ece": report["tiers"].get("hard", {}).get("ece"),
        "leaderboard_estimate": leaderboard_estimate(report["tiers"], backbone),
        "references": {
            k: {"accuracy": v["accuracy"], "intelligence_proxy": v["intelligence_proxy"]}
            for k, v in refs.items()
        },
    }
    if verbose:
        print(
            f"[jevbench] intelligence proxy: ours={report['summary']['intelligence_proxy']:.1f}",
            flush=True,
        )
        est = report["summary"]["leaderboard_estimate"]
        cost = "n/a" if est["cost"] is None else f"{est['cost']:.1f}"
        print(
            f"[jevbench] leaderboard axes (est.): speed={est['speed']:.1f} cost={cost}", flush=True
        )
        for v in refs.values():
            accs = " ".join(f"{t}={a:.3f}" for t, a in v["accuracy"].items())
            print(
                f"[jevbench]   {v['display'][:40]:40} {v['intelligence_proxy']:5.1f} | {accs}",
                flush=True,
            )
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def main(argv: list[str] | None = None) -> dict:
    import argparse

    from ..checkpoint import load_checkpoint
    from ..config import model_config
    from ..evidence_pipeline import reasoning_decision
    from ..model.electra import ElectraDecisionModel
    from ..tokenization import HFTokenizer, ToyTokenizer

    ap = argparse.ArgumentParser(description="JevBench public-tier evaluator")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", help="checkpoint directory")
    g.add_argument("--zero-shot", help="model size to evaluate untrained (e.g. electra-small)")
    g.add_argument("--export", help="export folder (ayaka.export), bf16 or int8")
    g.add_argument("--endpoint", help="URL of a running ayaka.serve (scores the HTTP path)")
    ap.add_argument(
        "--reasoning", action="store_true", help="gated worked-steps route (with --ckpt)"
    )
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="", help="bfloat16|float32 (default: bf16 on cuda)")
    ap.add_argument("--max-seq-len", type=int, default=0, help="0 = config serve_max_seq_len")
    ap.add_argument("--linear-mode", default="auto", help="int8 export runtime (see ayaka.export)")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.dtype:
        dtype = getattr(torch, args.dtype)
    else:
        dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    if args.endpoint:
        return run_jevbench(
            None,
            None,
            out_path=args.out,
            tiers=[t for t in args.tiers.split(",") if t],
            limit=args.limit,
            decision=EndpointDecision(args.endpoint),
        )
    tok = None
    if args.export:
        from ..export import load_exported

        model, tok = load_exported(
            args.export, device=args.device, dtype=dtype, linear_mode=args.linear_mode
        )
    elif args.ckpt:
        model = load_checkpoint(
            args.ckpt, device=args.device, dtype=dtype, merge=not args.reasoning
        )
    else:
        model = ElectraDecisionModel.from_config(
            model_config(args.zero_shot), dtype=dtype, device=args.device
        )
    if tok is None:
        tok = (
            ToyTokenizer()
            if model.cfg.backbone == "tiny"
            else HFTokenizer.from_pretrained(model.cfg.backbone)
        )
    return run_jevbench(
        model,
        tok,
        out_path=args.out,
        tiers=[t for t in args.tiers.split(",") if t],
        limit=args.limit,
        max_seq_len=args.max_seq_len or None,
        decision=reasoning_decision(model, tok, args.max_seq_len or None)
        if args.reasoning
        else None,
    )


if __name__ == "__main__":
    main()
