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
        "content_sha256": hashlib.sha256(
            json.dumps(
                sample.to_json(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
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


def scheduled_batches(inventory, steps, rows_per_step, seed, weights=None):
    """Yield the exact finite source/row indices, including leftovers between steps."""
    if any(type(n) is not int or n < 1 for n in (steps, rows_per_step)):
        raise ValueError("workload requires positive fixed steps and rows per step")
    if not inventory or any(not sample["rows"] for sample in inventory):
        raise ValueError("workload inventory needs every nonempty prepared sample")
    pending = []
    indices = sample_indices([row["language"] for row in inventory], seed, weights)
    for _ in range(steps):
        while len(pending) < rows_per_step:
            index = next(indices)
            pending.extend((index, row) for row in range(len(inventory[index]["rows"])))
        yield pending[:rows_per_step]
        pending = pending[rows_per_step:]


def profile_schedule(inventory, steps, rows_per_step, seed, weights=None, *, evenly_spaced=20):
    """Sample across the whole plan and include its maxima for six cost proxies.

    These are actual mixed production batches, not artificial homogeneous stress
    batches. Proxy maxima cover long decisions, padding, trace/proposal labels,
    cold images and total token work; they are not exact runtime upper bounds.
    """
    if type(evenly_spaced) is not int or evenly_spaced < 2:
        raise ValueError("schedule profiling needs at least two spaced batches")
    batches = list(scheduled_batches(inventory, steps, rows_per_step, seed, weights))
    selected = {round(i * (steps - 1) / (evenly_spaced - 1)) for i in range(evenly_spaced)}
    costs = []
    for batch in batches:
        rows = [inventory[index]["rows"][position] for index, position in batch]
        costs.append(
            (
                sum(row["length"] for row in rows),
                max(row["length"] for row in rows),
                sum(row["trace_tokens"] for row in rows),
                sum(row["proposal_tokens"] for row in rows),
                sum(row["image"] for row in rows),
                sum(row["length"] + row["trace_tokens"] + row["proposal_tokens"] for row in rows),
            )
        )
    selected.update(max(range(steps), key=lambda index: costs[index][axis]) for axis in range(6))
    return [(index, batches[index]) for index in sorted(selected)]


def finite_workload(inventory, steps, rows_per_step, seed, weights=None):
    if any(type(n) is not int or n < 1 for n in (steps, rows_per_step)):
        raise ValueError("workload requires positive fixed steps and rows per step")
    if not inventory or any(not sample["rows"] for sample in inventory):
        raise ValueError("workload inventory needs every nonempty prepared sample")
    counts = {key: Counter() for key in ("language", "type", "route", "data_kind")}
    digest = hashlib.sha256()
    seen, lineages = set(), set()
    totals = Counter()
    for batch in scheduled_batches(inventory, steps, rows_per_step, seed, weights):
        for index, position in batch:
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
