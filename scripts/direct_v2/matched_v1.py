"""Offline preflight or checked execution of the frozen published v1 route.

GPU execution needs separate authorization. --preflight-only never loads model
weights, downloads dependencies or performs a model forward.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.eval.matched_contract import fingerprint, load_protocol, validate_rows  # noqa: E402
from ayaka.eval.matched_execution import (  # noqa: E402
    CONTEXT_VERSION,
    FullContextDecision,
    checkpoint_snapshot,
    consumed_files,
    context_preflight,
    validate_context_rows,
)
from ayaka.swift.collect import load_reads  # noqa: E402
from scripts.direct_v2.matched_compare import cohort_items  # noqa: E402
from scripts.swift import v1_on_runner  # noqa: E402


def checked_run(decision, items, output, protocol, preflight, *, device=None):
    """Publish each row only after input, policy, probability and context checks."""
    items = list(items)
    existing = load_reads([output]) if output.exists() else []
    validate_rows(existing, items, protocol, system="v1", complete=False)
    validate_context_rows(existing, preflight)
    done = {row["id"] for row in existing}
    checked = FullContextDecision(decision.original)
    decision.original = checked
    written = 0
    with (
        tempfile.TemporaryDirectory() as temp,
        output.open("a", encoding="utf-8", newline="\n") as dest,
    ):
        pending = Path(temp) / "pending.jsonl"
        for item in items:
            if item.id in done:
                continue
            checked.contexts.clear()
            pending.write_bytes(b"")
            v1_on_runner.run_rows(decision, [item], pending, protocol["v1_run"], device=device)
            decision.original = checked  # legacy recorder must not accumulate wrappers
            (row,) = load_reads([pending])
            c = {
                "version": CONTEXT_VERSION,
                "decisions": copy.deepcopy(checked.contexts),
                "extraction_input_tokens": preflight[item.id]["extraction_input_tokens"],
            }
            c["sha256"] = fingerprint(c)
            row["checked_context"] = c
            validate_rows([row], [item], protocol, system="v1")
            validate_context_rows([row])
            dest.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            dest.flush()
            done.add(item.id)
            written += 1
    return written


def execute(args, *, tokenizer_factory=None, model_factory=None):
    protocol = load_protocol(args.protocol, args.protocol_sha256)
    items = cohort_items(
        {
            "procedural": args.procedural,
            "hard_calibration": args.hard_calibration,
            "hard_dev": args.hard_dev,
        },
        protocol,
    )
    if args.receipt.exists():
        raise ValueError("checked execution receipts are written once")
    receipt = json.loads(args.checkpoint_receipt.read_bytes())
    root = Path(receipt["checkpoint_path"]).resolve()
    protected = {
        Path(p).resolve()
        for p in (
            args.protocol,
            args.procedural,
            args.hard_calibration,
            args.hard_dev,
            args.checkpoint_receipt,
        )
    }
    protected.update(root / name for name in consumed_files(root))
    destinations = [args.output.resolve(), args.receipt.resolve()]
    if destinations[0] == destinations[1] or any(
        path.is_relative_to(root)
        or path in protected
        or (
            path.exists() and any(source.exists() and path.samefile(source) for source in protected)
        )
        for path in destinations
    ):
        raise ValueError("output/receipt must not modify checkpoint or pinned control inputs")
    existing = load_reads([args.output]) if args.output.exists() else []
    validate_rows(existing, items, protocol, system="v1", complete=False)
    validate_context_rows(existing)
    with checkpoint_snapshot(receipt, protocol) as (snapshot, cfg):
        if tokenizer_factory is None:
            from transformers import AutoTokenizer

            from ayaka.tokenization import HFTokenizer

            tok = HFTokenizer(
                AutoTokenizer.from_pretrained(
                    cfg.backbone, revision=cfg.backbone_revision, local_files_only=True
                ),
                cfg.backbone,
            )
        else:
            tok = tokenizer_factory(cfg)
        counts = context_preflight(items, tok, cfg)
        validate_context_rows(existing, counts)
        report = {
            "version": "ayaka-checked-v1-execution-1",
            "protocol_file_sha256": args.protocol_sha256,
            "protocol": protocol,
            "context_preflight": counts,
            "no_truncation_preflight": True,
            "preflight_only": args.preflight_only,
            "complete": False,
            "new_rows": 0,
            "scope": protocol["scope"],
        }
        if not args.preflight_only:
            if model_factory is None:
                import torch

                from ayaka.checkpoint import load_checkpoint

                model = load_checkpoint(
                    str(snapshot),
                    device=args.device,
                    dtype=torch.bfloat16,
                    merge=False,
                    local_files_only=True,
                    strict_loading=True,
                )
            else:
                model = model_factory(snapshot, cfg)
            from ayaka.evidence_pipeline import reasoning_decision

            decision = reasoning_decision(model, tok, 8192)
            report["new_rows"] = checked_run(
                decision, items, args.output, protocol, counts, device=args.device
            )
            rows = validate_rows(load_reads([args.output]), items, protocol, system="v1")
            validate_context_rows(rows, counts)
            report.update(complete=True, output_sha256=fingerprint(rows))
    # Snapshot/source checks at context exit precede successful receipt publication.
    with args.receipt.open("x", encoding="utf-8", newline="\n") as dest:
        dest.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "protocol",
        "procedural",
        "hard-calibration",
        "hard-dev",
        "checkpoint-receipt",
        "output",
        "receipt",
    ):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)
    report = execute(args)
    print(
        json.dumps(
            {
                "preflight_only": report["preflight_only"],
                "complete": report["complete"],
                "new_rows": report["new_rows"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
