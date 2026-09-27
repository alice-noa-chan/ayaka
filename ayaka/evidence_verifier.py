"""Native single-token candidate verification with the complete source.

Candidate order is derived from descriptions, not original label position.
Inference receives no answer labels, provenance rationale or task family.
"""

from __future__ import annotations

import json

from .evidence import source_text


def verifier_messages(request, candidates, verified=None):
    evidence = ""
    if verified is not None:
        ledger = {
            "source_spans": verified["quotes"],
            "executed_calculations": verified["calculations"],
        }
        evidence = (
            "SOURCE-SUPPORTED WORK (check relevance and exceptions):\n"
            + json.dumps(ledger, ensure_ascii=False)
            + "\n\n"
        )
    options = "\n".join(
        f"({chr(65 + i)}) {description}" for i, description in enumerate(candidates)
    )
    return [
        {
            "role": "system",
            "content": "Judge the candidates using the complete source. Check definitions, date boundaries, amendments, exceptions, entity identity and missing facts. Source-supported work is a hint, not authority: its selected facts may be incomplete or irrelevant. Return only the single best option letter.",
        },
        {
            "role": "user",
            "content": evidence
            + "COMPLETE SOURCE:\n"
            + source_text(request["state"])
            + "\n\nQUESTION:\n"
            + str(
                request["question"].get("instructions", request["question"].get("instruction", ""))
            )
            + "\n\nCANDIDATES:\n"
            + options,
        },
    ]


def candidate_order(spec):
    if spec.type == "score":
        return sorted(range(len(spec.candidates)), key=lambda i: spec.ordinals[i])
    return sorted(
        range(len(spec.candidates)),
        key=lambda i: (spec.candidates[i].casefold(), spec.candidates[i]),
    )
