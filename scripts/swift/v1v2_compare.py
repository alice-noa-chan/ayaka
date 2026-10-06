"""Predeclared paired comparison: v2 reasoning off (Swift) against v1 with its reasoning route.

Protocol: docs/experiments/V1ON_V2OFF_PROTOCOL_2026-10-05.md. The rule below is fixed
before any v1 row on this cohort exists.

Systems, all scored by the same Swift scorer on the same ids:
- v2_off: frozen 12B Swift `min` reads, scored with the saved production policy and its
  reasoning route removed (zero generated tokens).
- v1_on: published ayaka-large with the frozen worked-steps route. These are its final
  probabilities, unmodified.
- v1_off: v1's own single pass (diagnostic).

Every row of both systems is checked against the pinned cohort files: state, full question,
gold, soft gold, source, tier and case/cluster, plus each system's run recipe. Missing,
extra, duplicate or mismatched rows are refused.

Success for "v2 off beats v1 on" (from V2_DIRECT_OVER_V1_REASONING_2026-10-04.md), all on
the full cohort:
1. thresholded Choice/Noul gold credit +5 points or more
2. equal-type chance-corrected score (mean of per-type CC) +5 or more
3. bootstrap 95% CI lower bound of (2) > 0, resampling dependence components: rows that
   share a case, or share a HotpotQA paragraph title anywhere in the cohort
4. Score normalized RPS not worse

The result is a matched comparison on this exposed cohort. The hard part was seen by the
cygnet prompt decision, so it is not a fresh independent confirmation. Speed and Cost are
not compared: v1 ran on HF eager and v2 on vLLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v1_on_runner import (  # noqa: E402
    BACKBONE_REVISION,
    CHECKPOINT_REPO,
    CHECKPOINT_REVISION,
    item_binding,
)

from ayaka.eval.read_artifact import fingerprint  # noqa: E402
from ayaka.swift.collect import iter_dataset, load_reads  # noqa: E402
from ayaka.swift.evaluate import percentile  # noqa: E402
from ayaka.swift.policy import Policy  # noqa: E402
from ayaka.swift.score import (  # noqa: E402
    TYPES,
    calibration,
    intelligence,
    normalized_rps,
    noul_labels,
    prepare_rows,
)

B, SEED = 2000, 15
SUCCESS = {"credit_pp": 5.0, "typed_cc": 5.0, "typed_cc_ci_lower": 0.0}
IDENTITY = Policy(noul_commit=False)
COHORT = {
    "procedural": "abf80932e93b1236296af3e79c399b6ce1b517bbe3ffe7f4eda6bbb31db4c92d",
    "hard_calibration": "01f17bc9a454e69f771f4afb0d3d73cf415b09609f6dded36517da36b261b58c",
    "hard_dev": "8a237769ee38d51adfa89bf6c6fedb5d49e966196497f165dbf5ca2072c3f7ee",
}
SWIFT_IMPLEMENTATION = {
    "readers.py": "0234f36ea1c1787de1facac32102c9d75b5210890efa98fa90d4e493320f7926",
    "prompt.py": "2fa3444e430982097843ebe636f3cddac11d5225fd0667a6bf10f39e71b8a995",
    "grouping.py": "7fde0ebacc94d494690d7dfae998f613f0303bbc288f49a7687746a008e57b44",
}
V1_RUN = {
    "runner": "v1-on-runner-2",
    "checkpoint_repo": CHECKPOINT_REPO,
    "checkpoint_revision": CHECKPOINT_REVISION,
    "backbone_revision": BACKBONE_REVISION,
    "max_seq_len": 8192,
    "max_new_tokens": 384,
    "reasoner_adapter": "off",
}
BINDING_KEYS = ("state", "question", "gold", "gold_distribution", "source", "tier", "case_id")


def load_cohort(paths: dict[str, Path]) -> dict[str, dict]:
    """Hash each file's bytes, then parse those same bytes; return the expected bindings."""
    if set(paths) != set(COHORT):
        raise ValueError(f"cohort must be exactly {sorted(COHORT)}")
    expected = {}
    with tempfile.TemporaryDirectory() as scratch:
        for name, path in paths.items():
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != COHORT[name]:
                raise ValueError(f"{name}: file does not match the frozen cohort hash")
            copy = Path(scratch) / f"{name}.jsonl"
            copy.write_bytes(data)
            for item in iter_dataset([copy]):
                if item.id in expected:
                    raise ValueError(f"{item.id}: duplicate id across cohort files")
                if item.public:
                    raise ValueError(f"{item.id}: public item in the cohort")
                expected[item.id] = item_binding(item)
    return expected


def check_v2(rows, expected) -> None:
    for row in rows:
        b, runtime = row.get("binding") or {}, (row.get("binding") or {}).get("runtime") or {}
        if (
            row.get("prompt_variant") != "min"
            or row.get("model") != "google/gemma-4-12B-it"
            or row.get("revision") != BACKBONE_REVISION
            or row.get("readout") != "canonical_letter_raw"
            or row.get("passes") != 1
            or row.get("public") is not False
            or row.get("diagnostic")
            or any(key in row for key in ("reasoned_read", "trace", "generated_tokens"))
            or runtime.get("prompt_variant") != "min"
            or runtime.get("adapter_sha256") is not None
            or runtime.get("implementation_sha256") != SWIFT_IMPLEMENTATION
        ):
            raise ValueError(f"{row.get('id')}: not a frozen Swift min direct read")
        want = expected.get(row["id"])
        if want is None or any(
            fingerprint(b.get(key)) != fingerprint(want[key]) for key in BINDING_KEYS
        ):
            raise ValueError(f"{row['id']}: Swift read does not bind the cohort input")


def check_v1(rows, expected) -> None:
    for row in rows:
        run = row.get("run") or {}
        if (
            row.get("readout") != "v1_native_route"
            or row.get("public") is not False
            or {k: run.get(k) for k in V1_RUN} != V1_RUN
            or not run.get("checkpoint_files_sha256")
            or not run.get("policy")
        ):
            raise ValueError(f"{row.get('id')}: not the pinned v1 route run")
        if fingerprint(row.get("binding")) != fingerprint(expected.get(row["id"])):
            raise ValueError(f"{row['id']}: v1 row does not bind the cohort input")
    if len({fingerprint(row["run"]) for row in rows}) != 1:
        raise ValueError("v1 rows come from more than one run contract")


def components(rows) -> list[list[int]]:
    """Union rows that share a case, or a HotpotQA paragraph title ("Title: text" lines)."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[str, int] = {}
    for index, row in enumerate(rows):
        keys = ["case:" + row["case_id"]]
        state = row["binding"]["state"]
        if row["source"] == "hotpot_val" and isinstance(state, str):
            keys += ["title:" + p.split(": ", 1)[0] for p in state.split("\n") if ": " in p]
        for key in keys:
            if key in first:
                parent[find(index)] = find(first[key])
            else:
                first[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    return [groups[k] for k in sorted(groups, key=lambda k: rows[groups[k][0]]["id"])]


def credit(item) -> bool:
    if item.type == "noul":
        false, true = noul_labels(item.probs)
        p = item.probs[true]
        prediction = false if p <= 0.2 else true if p >= 0.8 else None
        return prediction == item.gold
    return max(item.labels, key=item.probs.get) == item.gold


def metrics(items) -> dict:
    report = intelligence(items)
    per_type = report["per_type_CC"]
    scored = [i for i in items if i.type in ("choice", "noul")]
    rps = [normalized_rps(i) for i in items if i.type == "score"]
    return {
        "n": len(items),
        "I": report["I"],
        "per_type_CC": per_type,
        "typed_cc": mean(per_type.values()) if len(per_type) == len(TYPES) else None,
        "credit": 100 * mean(map(credit, scored)) if scored else None,
        "score_rps": mean(rps) if rps else None,
        "C": calibration(items)["C"] if any(i.type == "choice" for i in items) else None,
    }


def compare(v2_rows, v1_rows, policy: Policy, expected: dict) -> dict:
    for name, rows in (("v2", v2_rows), ("v1", v1_rows)):
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(expected):
            raise ValueError(f"{name} rows must cover exactly the frozen cohort once")
    check_v2(v2_rows, expected)
    check_v1(v1_rows, expected)
    v1 = {row["id"]: row for row in v1_rows}
    v2_rows = sorted(v2_rows, key=lambda r: r["id"])
    v1_rows = [v1[row["id"]] for row in v2_rows]
    for a, b in zip(v2_rows, v1_rows, strict=True):
        if a["labels"] != b["labels"] or a["type"] != b["type"]:
            raise ValueError(f"{a['id']}: label order or type differs between systems")
    systems = {
        "v2_off": prepare_rows(v2_rows, policy),
        "v1_on": prepare_rows(v1_rows, IDENTITY),
        "v1_off": prepare_rows(
            [{**row, "raw_probs": row["baseline_probs"]} for row in v1_rows], IDENTITY
        ),
    }
    groups = components(v1_rows)

    def deltas(indices):
        a = metrics([systems["v2_off"][i] for i in indices])
        b = metrics([systems["v1_on"][i] for i in indices])
        return {k: a[k] - b[k] for k in ("typed_cc", "I", "credit") if None not in (a[k], b[k])}

    everything = list(range(len(v2_rows)))
    rng = random.Random(SEED)
    draws = {"typed_cc": [], "I": [], "credit": []}
    for _ in range(B):
        sample = [i for _ in groups for i in rng.choice(groups)]
        for key, value in deltas(sample).items():
            draws[key].append(value)
    point = {name: metrics(items) for name, items in systems.items()}
    delta = deltas(everything)
    ci = {k: [percentile(v, 0.025), percentile(v, 0.975)] for k, v in draws.items() if v}
    checks = {
        "credit_plus_5pp": delta["credit"] >= SUCCESS["credit_pp"],
        "typed_cc_plus_5": delta["typed_cc"] >= SUCCESS["typed_cc"],
        "typed_cc_ci_lower_above_0": ci["typed_cc"][0] > SUCCESS["typed_cc_ci_lower"],
        "score_rps_not_worse": point["v2_off"]["score_rps"] <= point["v1_on"]["score_rps"],
    }
    by_source = {}
    for source in sorted({row["source"] for row in v2_rows}):
        idx = [i for i, row in enumerate(v2_rows) if row["source"] == source]
        by_source[source] = {
            name: {k: metrics([items[i] for i in idx])[k] for k in ("n", "credit", "per_type_CC")}
            for name, items in systems.items()
        }
    routes: dict[str, int] = {}
    for row in v1_rows:
        routes[row["route"]] = routes.get(row["route"], 0) + 1
    return {
        "scope": "matched comparison on an exposed cohort; not a fresh independent confirmation",
        "systems": point,
        "delta_v2off_minus_v1on": delta,
        "ci_95": ci,
        "bootstrap": {
            "B": B,
            "seed": SEED,
            "unit": "case and shared-HotpotQA-paragraph components",
            "components": len(groups),
            "largest_component": max(map(len, groups)),
        },
        "success_rule": SUCCESS,
        "checks": checks,
        "v2_off_beats_v1_on_on_this_cohort": all(checks.values()),
        "by_source": by_source,
        "v1_routes": routes,
        "v1_route_errors": sum(1 for row in v1_rows if row.get("route_error")),
        "v1_run": v1_rows[0]["run"],
        "comparison_scope": "Intelligence/credit/calibration only; Speed and Cost not compared",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--procedural", type=Path, required=True)
    parser.add_argument("--hard-calibration", type=Path, required=True)
    parser.add_argument("--hard-dev", type=Path, required=True)
    parser.add_argument("--v2-reads", nargs="+", required=True, type=Path)
    parser.add_argument("--v1-rows", nargs="+", required=True, type=Path)
    parser.add_argument("--policy", type=Path, required=True, help="saved production policy.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    expected = load_cohort(
        {
            "procedural": args.procedural,
            "hard_calibration": args.hard_calibration,
            "hard_dev": args.hard_dev,
        }
    )
    policy = replace(Policy.load(args.policy), reasoning_route=None)
    report = compare(load_reads(args.v2_reads), load_reads(args.v1_rows), policy, expected)
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; reports are written once")
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    s, d, c = report["systems"], report["delta_v2off_minus_v1on"], report["ci_95"]
    for name in s:
        print(
            f"{name:7s} I {s[name]['I']:.2f} typedCC {s[name]['typed_cc']:.2f}"
            f" credit {s[name]['credit']:.2f}"
        )
    print(f"delta typedCC {d['typed_cc']:+.2f} CI {c['typed_cc']} credit {d['credit']:+.2f}")
    print("v2_off_beats_v1_on_on_this_cohort:", report["v2_off_beats_v1_on_on_this_cohort"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
