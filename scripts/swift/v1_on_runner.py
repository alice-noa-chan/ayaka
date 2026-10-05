"""Run the published v1 system with its frozen worked-steps route on canonical cohort rows.

The v1 system is ayaka-large at the pinned revision, an unmerged LoRA, and
``reasoning_decision`` with ``FROZEN_REASONING_POLICY``. That is the same path as
``python -m ayaka.eval.jevbench --ckpt ... --reasoning``. Each row records:

- the final route probabilities ("v1 on")
- the single-pass baseline that the route starts from ("v1 off")
- the route taken, worked-step length and timings

Rows are scored next to Swift reads by ``v1v2_compare.py``. Nothing is fitted here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.swift.collect import DatasetItem, iter_dataset  # noqa: E402


def jevbench_record(item: DatasetItem) -> dict:
    """The canonical item as a JevBench record, labels and descriptions in Swift order."""
    q = item.question
    return {
        "id": item.id,
        "state": item.state,
        "labels": list(q.labels),
        "expected": item.gold if isinstance(item.gold, str) else max(item.gold, key=item.gold.get),
        "family": item.family or item.source,
        "question": {
            "type": q.type,
            "instructions": q.instruction,
            "criteria": dict(zip(q.labels, q.descriptions, strict=True)),
        },
    }


class BaselineRecorder:
    """Wraps the route's inner Decision; its first call per item is the v1 single pass."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def decide(self, *args, **kwargs):
        result = self.inner.decide(*args, **kwargs)
        self.calls.append(result)
        return result

    def __getattr__(self, name):
        return getattr(self.inner, name)


def run_rows(decision, items, output: Path, device=None) -> int:
    from ayaka.eval.jevbench import record_to_item

    recorder = BaselineRecorder(decision.original)
    decision.original = recorder
    done = set()
    if output.exists():
        done = {
            json.loads(line)["id"]
            for line in output.read_text(encoding="utf-8").split("\n")
            if line
        }
    written = 0
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for item in items:
            if item.id in done:
                continue
            bench = record_to_item(jevbench_record(item))
            if bench.labels != list(item.question.labels):
                raise ValueError(f"{item.id}: v1 label order differs from the Swift order")
            recorder.calls.clear()
            tick = time.perf_counter()
            result = decision.decide(bench.state, [bench.spec], device=device)[0]
            latency = time.perf_counter() - tick
            baseline = recorder.calls[0][0].probs
            evidence = (result.extras or {}).get("evidence", {})
            row = {
                "id": item.id,
                "source": item.source,
                "tier": item.tier,
                "type": item.question.type,
                "labels": list(item.question.labels),
                "gold": item.gold,
                "gold_distribution": item.gold_distribution,
                "case_id": item.case_id or item.cluster_id or item.id,
                "cluster_id": item.cluster_id or item.case_id or item.id,
                "public": False,
                "readout": "v1_native_route",
                "raw_probs": dict(zip(bench.labels, result.probs, strict=True)),
                "baseline_probs": dict(zip(bench.labels, baseline, strict=True)),
                "route": evidence.get("route", "baseline"),
                "route_error": evidence.get("error"),
                "worked_steps_chars": len(evidence.get("worked_steps", "") or ""),
                "extraction_s": evidence.get("extraction_s", 0.0),
                "latency_s": latency,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="+")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    import torch

    from ayaka.checkpoint import load_checkpoint, resolve_checkpoint
    from ayaka.evidence_pipeline import reasoning_decision
    from ayaka.tokenization import HFTokenizer

    model = load_checkpoint(
        resolve_checkpoint(str(args.checkpoint)),
        device=args.device,
        dtype=torch.bfloat16,
        merge=False,
    )
    tok = HFTokenizer.for_config(model.cfg)
    decision = reasoning_decision(model, tok, args.max_seq_len)
    items = list(iter_dataset(args.dataset))
    written = run_rows(decision, items, args.output, device=next(model.parameters()).device)
    print(f"Wrote {written} v1 rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
