"""Prepare local protocol pins before any GPU work; no model/tokenizer download.

The existing checkpoint receipt must already cover every consumed config/meta/
head/adapter file. Fitting provenance is an externally pinned declaration, not
proof that fitting was executed or that prior corpus history is complete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.eval.matched_contract import (  # noqa: E402
    digest,
    make_protocol,
    pinned_bytes,
    policy_from_bytes,
)
from ayaka.eval.matched_execution import checkpoint_snapshot  # noqa: E402


def prepare(checkpoint_receipt_bytes, policy_bytes, fit_provenance_bytes):
    receipt = json.loads(checkpoint_receipt_bytes)
    fit = json.loads(fit_provenance_bytes)
    if set(fit) != {"inputs_sha256", "source_sha256"}:
        raise ValueError("fitting provenance requires input and source anchors")
    protocol = make_protocol(
        checkpoint_hashes=receipt["source_sha256"],
        policy_sha256=digest(policy_bytes),
        fit_input_hashes=fit["inputs_sha256"],
        fit_source_hashes=fit["source_sha256"],
    )
    policy_from_bytes(policy_bytes, protocol)
    with checkpoint_snapshot(receipt, protocol):
        pass
    return protocol


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint-receipt", "policy", "fit-provenance"):
        p.add_argument(f"--{name}", type=Path, required=True)
        p.add_argument(f"--{name}-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("protocol pins are published once before GPU observation")
    checkpoint_bytes = pinned_bytes(args.checkpoint_receipt, args.checkpoint_receipt_sha256)
    root = Path(json.loads(checkpoint_bytes)["checkpoint_path"]).resolve()
    output = args.output.resolve()
    protected = {p.resolve() for p in (args.checkpoint_receipt, args.policy, args.fit_provenance)}
    if output.is_relative_to(root) or output in protected:
        raise ValueError("protocol output must not modify checkpoint or preparation inputs")
    protocol = prepare(
        checkpoint_bytes,
        pinned_bytes(args.policy, args.policy_sha256),
        pinned_bytes(args.fit_provenance, args.fit_provenance_sha256),
    )
    data = (json.dumps(protocol, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as dest:
        dest.write(data)
    print(json.dumps({"protocol": str(args.output), "protocol_sha256": digest(data)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
