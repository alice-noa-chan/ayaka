"""Deterministic finite workload accounting shared with the production sampler."""

import hashlib
import json
import math
import random
from collections import Counter


def sample_indices(languages, seed, weights=None):
    if not languages:
        raise ValueError("sample stream needs nonempty languages")
    rng = random.Random(seed)
    if weights is None:
        while True:
            order = list(range(len(languages)))
            rng.shuffle(order)
            yield from order
    pools = {}
    for index, language in enumerate(languages):
        pools.setdefault(language, []).append(index)
    names = sorted(pools)
    values = [weights.get(name, 0) for name in names]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("each training language needs a finite positive sampling weight")
    pending = {name: [] for name in names}
    while True:
        language = rng.choices(names, weights=values)[0]
        if not pending[language]:
            pending[language] = list(pools[language])
            rng.shuffle(pending[language])
        yield pending[language].pop()


def describe_rows(sample, items):
    return {
        "language": sample.metadata["language"],
        "source_lineage": sample.metadata["source_lineage"],
        "data_kind": sample.metadata.get("data_kind", "authored"),
        "rows": [
            {
                "type": item.type,
                "length": item.length,
                "trace_tokens": len(item.reasoning_labels or []),
                "proposal_tokens": len(item.proposal_labels or []),
                "image": item.native_inputs is not None,
                "flagged": item.flagged,
            }
            for item in items
        ],
    }


def finite_workload(inventory, steps, rows_per_step, seed, weights=None):
    if any(type(n) is not int or n < 1 for n in (steps, rows_per_step)):
        raise ValueError("workload requires positive fixed steps and rows per step")
    if not inventory or any(not sample["rows"] for sample in inventory):
        raise ValueError("workload inventory needs every nonempty prepared sample")
    counts = {key: Counter() for key in ("language", "type", "route", "data_kind")}
    digest = hashlib.sha256()
    seen, lineages = set(), set()
    totals = Counter()
    pending = []
    indices = sample_indices([row["language"] for row in inventory], seed, weights)
    for _ in range(steps):
        while len(pending) < rows_per_step:
            index = next(indices)
            pending.extend((index, row) for row in range(len(inventory[index]["rows"])))
        for index, position in pending[:rows_per_step]:
            sample, row = inventory[index], inventory[index]["rows"][position]
            digest.update(f"{index}:{position};".encode())
            seen.add((index, position))
            lineages.add(sample["source_lineage"])
            counts["language"][sample["language"]] += 1
            counts["type"][row["type"]] += 1
            route = (
                "image_trace"
                if row["image"] and row["trace_tokens"]
                else (
                    "image"
                    if row["image"]
                    else (
                        "proposal"
                        if row["proposal_tokens"]
                        else ("trace" if row["trace_tokens"] else "direct")
                    )
                )
            )
            counts["route"][route] += 1
            counts["data_kind"][sample["data_kind"]] += 1
            totals["text_tokens"] += row["length"]
            totals["trace_tokens"] += row["trace_tokens"]
            totals["proposal_tokens"] += row["proposal_tokens"]
        pending = pending[rows_per_step:]
    inventory_digest = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "steps": steps,
        "rows_per_step": rows_per_step,
        "total_rows": steps * rows_per_step,
        "unique_prepared_rows": len(seen),
        "available_prepared_rows": sum(len(sample["rows"]) for sample in inventory),
        "unique_source_lineages": len(lineages),
        "counts": {key: dict(value) for key, value in counts.items()},
        "tokens": dict(totals),
        "schedule_sha256": digest.hexdigest(),
        "inventory_sha256": inventory_digest,
        "seed": seed,
        "optimizer_steps_executed": 0,
        "semantics": "fixed complete schedule; repeated prepared rows are not independent questions or epochs",
    }


def stress_indices(inventory):
    """Cover every language/type/route/flag stratum using its largest prepared row."""
    selected = {}
    for index, sample in enumerate(inventory):
        for row in sample["rows"]:
            key = (
                sample["language"],
                row["type"],
                row["image"],
                bool(row["trace_tokens"]),
                bool(row["proposal_tokens"]),
                row["flagged"],
            )
            cost = row["length"] + row["trace_tokens"] + row["proposal_tokens"]
            if key not in selected or cost > selected[key][0]:
                selected[key] = cost, index
    return sorted({index for _, index in selected.values()})
