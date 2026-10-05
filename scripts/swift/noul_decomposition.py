"""Diagnostic: is the cygnet Noul gain discrimination or a threshold shift? Saved reads only.

For min and cygnet on v2 dev and hard dev Noul it reports:
- AUC of raw P(true)
- accuracy at 0.5
- accuracy at a logit threshold fit on calibration
- accuracy at the dev-oracle threshold (an upper bound, never a policy)
- CC under the fitted policies

Results are recorded in docs/experiments/SWIFT_NOUL_DECOMPOSITION_2026-10-05.md.
Nothing is fitted for serving.
"""

import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ayaka.swift.collect import load_reads
from ayaka.swift.policy import Policy
from ayaka.swift.score import noul_cc, prepare_rows

A7 = Path("runs/swift-gpu-20261005/attempt7/extracted/out/google_gemma-4-12B-it")
HD = Path("runs/swift-hard-dev-20261005/gpu/hard")
sel = json.loads((A7 / "variant_selection.json").read_text(encoding="utf-8"))
gate = json.loads(
    Path("runs/swift-hard-dev-20261005/hard_dev_gate.json").read_text(encoding="utf-8")
)


def noul(path):
    return [
        r for r in load_reads([path]) if r["type"] == "noul" and r["readout"] != "grouped_approx"
    ]


def p_true(r):
    return r["raw_probs"]["true"]


def auc(rows):
    pos = [p_true(r) for r in rows if r["gold"] == "true"]
    neg = [p_true(r) for r in rows if r["gold"] == "false"]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def logit(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def acc_at(rows, t):
    return sum((logit(p_true(r)) > t) == (r["gold"] == "true") for r in rows) / len(rows)


def best_t(rows):
    cands = sorted({logit(p_true(r)) for r in rows})
    grid = [-20] + [(a + b) / 2 for a, b in zip(cands, cands[1:], strict=False)] + [20]
    return max(grid, key=lambda t: (acc_at(rows, t), -abs(t)))


def report(name, cal, dev, policy):
    items = prepare_rows(dev, policy)
    abst = sum(0.2 < it.probs["true"] < 0.8 for it in items) / len(items)
    t = best_t(cal)
    pos_rate = sum(r["gold"] == "true" for r in dev) / len(dev)
    print(
        f"{name:14s} n={len(dev):3d} AUC {auc(dev):.3f} | acc@0.5 {acc_at(dev, 0):.3f} | cal-fit thr(logit {t:+.2f}) {acc_at(dev, t):.3f} | oracle {acc_at(dev, best_t(dev)):.3f} | policy CC {noul_cc(items):5.1f} abstain {abst:.2f} | gold-true {pos_rate:.2f} | mean P(true) {sum(map(p_true, dev)) / len(dev):.2f}"
    )


for v in ("min", "cygnet"):
    pol = Policy(**sel["policies"][v]) if isinstance(sel["policies"][v], dict) else None
    cal = noul(A7 / v / "v2_calibration.reads.jsonl")
    dev = noul(A7 / v / "v2_dev.reads.jsonl")
    report(f"v2 {v}", cal, dev, pol)
    by = collections.defaultdict(list)
    for r in dev:
        by[r["source"]].append(r)
    calby = collections.defaultdict(list)
    for r in cal:
        calby[r["source"]].append(r)
    for s, rows in sorted(by.items()):
        report(f"  {s[:12]}", calby[s], rows, pol)
for v in ("min", "cygnet"):
    pol = Policy(**gate["policies"][v])
    cal = noul(HD / v / "hard_calibration.reads.jsonl")
    dev = noul(HD / v / "hard_dev.reads.jsonl")
    report(f"hard {v}", cal, dev, pol)
