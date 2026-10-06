"""Checked v1-on/v2-off entrypoint; no fitting, model load or token generation.

Use this entrypoint instead of the legacy scripts/swift/v1v2_compare.py CLI.
Its protocol bytes must be pinned before collecting v1 rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.eval.matched_contract import (  # noqa: E402
    digest,
    fingerprint,
    item_binding,
    load_protocol,
    pinned_bytes,
    policy_from_bytes,
    validate_rows,
)
from ayaka.eval.matched_execution import validate_context_rows  # noqa: E402
from ayaka.swift.collect import iter_dataset, load_reads  # noqa: E402
from scripts.swift import v1v2_compare  # noqa: E402


def cohort_parts(paths, protocol):
    if set(paths) != set(protocol["cohort_sha256"]):
        raise ValueError("all pinned cohort files are required")
    parts = {}
    with tempfile.TemporaryDirectory() as scratch:
        for name, path in sorted(paths.items()):
            data = pinned_bytes(path, protocol["cohort_sha256"][name])
            copy = Path(scratch) / f"{name}.jsonl"
            copy.write_bytes(data)
            parts[name] = list(iter_dataset([copy]))
    items = [item for values in parts.values() for item in values]
    if len({item.id for item in items}) != len(items) or any(
        item.public is not False for item in items
    ):
        raise ValueError("pinned cohort repeats IDs or contains public inputs")
    return parts


def cohort_items(paths, protocol):
    return [item for values in cohort_parts(paths, protocol).values() for item in values]


def checked_compare(v2_rows, v1_rows, items, protocol, policy_bytes, execution_receipt):
    policy = policy_from_bytes(policy_bytes, protocol)
    items = list(items)
    v2_rows = validate_rows(v2_rows, items, protocol, system="v2")
    v1_rows = validate_rows(v1_rows, items, protocol, system="v1")
    if (
        execution_receipt.get("version") != "ayaka-checked-v1-execution-1"
        or execution_receipt.get("complete") is not True
        or execution_receipt.get("preflight_only") is not False
        or execution_receipt.get("output_sha256") != fingerprint(v1_rows)
        or fingerprint(execution_receipt.get("protocol")) != fingerprint(protocol)
    ):
        raise ValueError("scoring requires the complete checked execution receipt for these rows")
    validate_context_rows(v1_rows, execution_receipt.get("context_preflight") or {})
    report = v1v2_compare.compare(
        v2_rows, v1_rows, policy, {item.id: item_binding(item) for item in items}
    )
    report["validated_protocol"] = protocol
    report["validation"] = {
        "scoring_fields_bound": True,
        "native_reads_validated": True,
        "fresh_independence_attested": False,
    }
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--v1-execution-receipt", type=Path, required=True)
    p.add_argument("--v1-execution-receipt-sha256", required=True)
    for name in ("procedural", "hard-calibration", "hard-dev", "policy", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--v2-reads", nargs="+", type=Path, required=True)
    p.add_argument("--v1-rows", nargs="+", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("comparison reports are written once")
    protocol = load_protocol(args.protocol, args.protocol_sha256)
    policy_bytes = pinned_bytes(args.policy, protocol["policy_sha256"])
    execution_receipt = json.loads(
        pinned_bytes(args.v1_execution_receipt, args.v1_execution_receipt_sha256)
    )
    if execution_receipt.get("protocol_file_sha256") != args.protocol_sha256:
        raise ValueError("execution receipt does not bind the pinned protocol bytes")
    items = cohort_items(
        {
            "procedural": args.procedural,
            "hard_calibration": args.hard_calibration,
            "hard_dev": args.hard_dev,
        },
        protocol,
    )
    report = checked_compare(
        load_reads(args.v2_reads),
        load_reads(args.v1_rows),
        items,
        protocol,
        policy_bytes,
        execution_receipt,
    )
    report["protocol_file_sha256"] = args.protocol_sha256
    report["v1_execution_receipt_sha256"] = args.v1_execution_receipt_sha256
    data = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"output_sha256": digest(data), "checks": report["checks"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
