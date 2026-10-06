"""Validate existing Swift reads and the saved policy before vLLM startup."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.eval.matched_contract import (  # noqa: E402
    BACKBONE_REVISION,
    SWIFT_IMPLEMENTATION,
    digest,
    load_policy,
    load_protocol,
    validate_rows,
)
from ayaka.input_errors import ContextLimitError  # noqa: E402
from ayaka.swift.collect import load_reads  # noqa: E402
from ayaka.swift.prompt import render_question  # noqa: E402
from ayaka.swift.readers import cached_tokenizer, token_input  # noqa: E402
from ayaka.training.tokenizer_identity import tokenizer_identity_scope  # noqa: E402
from scripts.direct_v2.matched_compare import cohort_parts  # noqa: E402


def verify_swift_source(root=None):
    """Future native reads must carry the same raw implementation bytes as saved reads."""
    root = Path(sys.modules[token_input.__module__].__file__).parent if root is None else Path(root)
    actual = {name: digest((root / name).read_bytes()) for name in SWIFT_IMPLEMENTATION}
    if actual != SWIFT_IMPLEMENTATION:
        raise ValueError(
            "Swift source differs from pinned collection bytes; use the frozen LF archive"
        )
    return actual


def preflight_procedural(items):
    """Validate the full pending native input on the offline pinned tokenizer."""
    tokenizer = cached_tokenizer("google/gemma-4-12B-it", BACKBONE_REVISION)
    checks = {}
    with tokenizer_identity_scope(tokenizer):
        for item in items:
            messages, mapping = render_question(
                item.state, item.question, prompt_variant="min", state_format="pretty"
            )
            encoded = token_input(tokenizer, messages, list(mapping), {"enable_thinking": False})
            if len(encoded["input_token_ids"]) + 1 > 16384:
                raise ContextLimitError(
                    "complete procedural Swift input exceeds native context; refuse GPU collection"
                )
            checks[item.id] = {
                "input_tokens": len(encoded["input_token_ids"]),
                "input_token_ids_sha256": encoded["input_token_ids_sha256"],
                "canonical_token_ids_sha256": encoded["canonical_token_ids_sha256"],
            }
    return checks


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "policy", "procedural", "hard-calibration", "hard-dev"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--v2-reads", nargs="+", type=Path, required=True)
    args = p.parse_args(argv)
    protocol = load_protocol(args.protocol, args.protocol_sha256)
    load_policy(args.policy, protocol)
    parts = cohort_parts(
        {
            "procedural": args.procedural,
            "hard_calibration": args.hard_calibration,
            "hard_dev": args.hard_dev,
        },
        protocol,
    )
    items = [*parts["hard_calibration"], *parts["hard_dev"]]
    rows = validate_rows(load_reads(args.v2_reads), items, protocol, system="v2")
    source = verify_swift_source()
    pending = preflight_procedural(parts["procedural"])
    print(
        json.dumps(
            {
                "validated_existing_v2_rows": len(rows),
                "complete_cohort_rows": sum(map(len, parts.values())),
                "procedural_rows_to_collect": len(parts["procedural"]),
                "procedural_native_preflight": pending,
                "swift_implementation_sha256": source,
                "model_forward_calls": 0,
                "fresh_independence_attested": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
