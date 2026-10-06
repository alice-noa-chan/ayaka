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
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.eval.read_artifact import fingerprint  # noqa: E402
from ayaka.swift.collect import DatasetItem, iter_dataset  # noqa: E402

RUNNER_VERSION = "v1-on-runner-2"
CHECKPOINT_REPO = "alice-noa-chan/ayaka-large"
CHECKPOINT_REVISION = "267605ee22f2b5f934d81e5fbee691952d2e6f55"
BACKBONE_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"


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


def item_binding(item: DatasetItem) -> dict:
    """The same semantic fields a Swift read binds: input, question, gold and provenance."""
    return {
        "state": item.state,
        "question": asdict(item.question),
        "gold": item.gold,
        "gold_distribution": item.gold_distribution,
        "source": item.source,
        "tier": item.tier,
        "public": item.public,
        "case_id": item.case_id or item.cluster_id or item.id,
        "cluster_id": item.cluster_id or item.case_id or item.id,
    }


def checkpoint_from_receipt(receipt: dict) -> tuple[Path, str]:
    """Only the pinned published checkpoint on the pinned backbone is the v1 system."""
    if (
        receipt.get("checkpoint_repo") != CHECKPOINT_REPO
        or receipt.get("checkpoint_revision") != CHECKPOINT_REVISION
        or receipt.get("base_revision") != BACKBONE_REVISION
        or not receipt.get("source_sha256")
    ):
        raise ValueError("v1 runs only the pinned ayaka-large checkpoint")
    return Path(receipt["checkpoint_path"]), fingerprint(receipt["source_sha256"])


def run_contract(checkpoint_files_sha256: str, max_seq_len: int) -> dict:
    from ayaka.evidence_pipeline import FROZEN_REASONING_POLICY

    return {
        "runner": RUNNER_VERSION,
        "checkpoint_repo": CHECKPOINT_REPO,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_files_sha256": checkpoint_files_sha256,
        "backbone_revision": BACKBONE_REVISION,
        "max_seq_len": max_seq_len,
        "max_new_tokens": 384,
        "reasoner_adapter": "off",
        "policy": asdict(FROZEN_REASONING_POLICY),
    }


def _existing(output: Path, run: dict, expected: dict) -> set[str]:
    """Rows from an earlier run must match this run's contract and inputs exactly."""
    done = set()
    if not output.exists():
        return done
    for line in output.read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        row = json.loads(line)
        if row["id"] in done:
            raise ValueError(f"{row['id']}: duplicate row in existing output")
        if row.get("run") != run:
            raise ValueError(f"{row['id']}: existing row has a different run contract")
        if row["id"] not in expected or fingerprint(row.get("binding")) != fingerprint(
            expected[row["id"]]
        ):
            raise ValueError(f"{row['id']}: existing row does not bind this input")
        done.add(row["id"])
    return done


def run_rows(decision, items, output: Path, run: dict, device=None) -> int:
    from ayaka.eval.jevbench import record_to_item

    items = list(items)
    expected = {}
    for item in items:
        if item.id in expected:
            raise ValueError(f"{item.id}: duplicate input id")
        if item.public:
            raise ValueError(f"{item.id}: public items are refused")
        expected[item.id] = item_binding(item)
    done = _existing(output, run, expected)
    recorder = BaselineRecorder(decision.original)
    decision.original = recorder
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
            binding = expected[item.id]
            row = {
                "id": item.id,
                "source": item.source,
                "tier": item.tier,
                "type": item.question.type,
                "labels": list(item.question.labels),
                "gold": item.gold,
                "gold_distribution": item.gold_distribution,
                "case_id": binding["case_id"],
                "cluster_id": binding["cluster_id"],
                "public": item.public,
                "readout": "v1_native_route",
                "run": run,
                "binding": binding,
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
            done.add(item.id)
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="+")
    parser.add_argument(
        "--checkpoint-receipt",
        type=Path,
        required=True,
        help="checkpoint.json from matched_native.py --prepare-checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    import torch

    from ayaka.checkpoint import load_checkpoint, resolve_checkpoint
    from ayaka.evidence_pipeline import reasoning_decision
    from ayaka.tokenization import HFTokenizer

    checkpoint, files_sha256 = checkpoint_from_receipt(
        json.loads(args.checkpoint_receipt.read_text(encoding="utf-8"))
    )
    model = load_checkpoint(
        resolve_checkpoint(str(checkpoint)),
        device=args.device,
        dtype=torch.bfloat16,
        merge=False,
    )
    tok = HFTokenizer.for_config(model.cfg)
    decision = reasoning_decision(model, tok, args.max_seq_len)
    items = list(iter_dataset(args.dataset))
    run = run_contract(files_sha256, args.max_seq_len)
    written = run_rows(decision, items, args.output, run, device=next(model.parameters()).device)
    print(f"Wrote {written} v1 rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
