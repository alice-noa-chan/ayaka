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
    item_binding,
    load_protocol,
    pinned_bytes,
    policy_from_bytes,
    validate_rows,
)
from ayaka.swift.collect import iter_dataset, load_reads  # noqa: E402
from scripts.swift import v1v2_compare  # noqa: E402


def cohort_items(paths, protocol):
    if set(paths) != set(protocol["cohort_sha256"]):
        raise ValueError("all pinned cohort files are required")
    items = []
    with tempfile.TemporaryDirectory() as scratch:
        for name, path in sorted(paths.items()):
            data = pinned_bytes(path, protocol["cohort_sha256"][name])
            copy = Path(scratch) / f"{name}.jsonl"
            copy.write_bytes(data)
            items.extend(iter_dataset([copy]))
    return items


def checked_compare(v2_rows, v1_rows, items, protocol, policy_bytes):
    policy = policy_from_bytes(policy_bytes, protocol)
    items = list(items)
    v2_rows = validate_rows(v2_rows, items, protocol, system="v2")
    v1_rows = validate_rows(v1_rows, items, protocol, system="v1")
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
    for name in ("procedural", "hard-calibration", "hard-dev", "policy", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--v2-reads", nargs="+", type=Path, required=True)
    p.add_argument("--v1-rows", nargs="+", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("comparison reports are written once")
    protocol = load_protocol(args.protocol, args.protocol_sha256)
    policy_bytes = pinned_bytes(args.policy, protocol["policy_sha256"])
    items = cohort_items(
        {
            "procedural": args.procedural,
            "hard_calibration": args.hard_calibration,
            "hard_dev": args.hard_dev,
        },
        protocol,
    )
    report = checked_compare(
        load_reads(args.v2_reads), load_reads(args.v1_rows), items, protocol, policy_bytes
    )
    report["protocol_file_sha256"] = args.protocol_sha256
    data = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"output_sha256": digest(data), "checks": report["checks"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
