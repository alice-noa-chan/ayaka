"""Explicitly reconcile reservations only after externally observed closure."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path


def reconcile_closed_windows(out, observations):
    from .v2 import TOTAL_SECONDS, write_json

    path = Path(out) / "budget.json"
    ledger = json.loads(path.read_text())
    old_charge = sum(entry.get("charged_s", entry["allocation_s"]) for entry in ledger["stages"])
    overrun = max(0, ledger["elapsed_s"] - old_charge)
    indices, containers = set(), set()
    for observation in observations:
        group = observation.get("stage_indices", [observation.get("stage_index")])
        if not group or any(
            type(i) is not int or not 0 <= i < len(ledger["stages"]) for i in group
        ):
            raise ValueError("closed-window reservation index is invalid")
        if group != list(range(group[0], group[-1] + 1)):
            raise ValueError("closed sequential stages must be unique and contiguous")
        if any(i in indices for i in group) or observation["container_id"] in containers:
            raise ValueError("each closed container/reservation can be reconciled only once")
        indices.update(group)
        containers.add(observation["container_id"])
        if not observation.get("app_id") or not observation.get("closure_evidence"):
            raise ValueError("observed app identity and closure evidence are required")
        start, end = [
            datetime.fromisoformat(observation[key]) for key in ("started_at", "ended_before")
        ]
        if (
            start.tzinfo is None
            or end.tzinfo is None
            or not start < end <= datetime.now(timezone.utc)
        ):
            raise ValueError("closed windows require ordered timezone-aware past bounds")
        overhead = observation.get("startup_shutdown_margin_s", 0)
        if type(overhead) is not int or overhead < 120:
            raise ValueError("retain at least 120 seconds of container overhead")
        if any(
            i not in group
            and other.get("closed_gpu_window", {}).get("container_id")
            == observation["container_id"]
            for i, other in enumerate(ledger["stages"])
        ):
            raise ValueError("a container cannot settle a second reservation")
        entries = [ledger["stages"][i] for i in group]
        if any(e.get("closed_gpu_window") not in (None, observation) for e in entries):
            raise ValueError("closed-window evidence cannot be replaced")
        upper = (end - start).total_seconds() + overhead
        if len(group) == 1:
            if "actual_elapsed_s" in entries[0]:
                raise ValueError("completed stages require a closed sequential-window audit")
            charges = [upper]
        else:
            if observation.get("sequential_stages") is not True:
                raise ValueError("explicit sequential-stage evidence is required")
            measured = []
            for entry in entries[:-1]:
                elapsed = entry.get("actual_elapsed_s")
                if (
                    entry["status"] not in ("complete", "failed", "incomplete")
                    or not isinstance(elapsed, (int, float))
                    or not math.isfinite(elapsed)
                    or elapsed < 0
                ):
                    raise ValueError("preceding sequential stages need measured completion")
                measured.append(elapsed)
            if sum(measured) > upper:
                raise ValueError("measured stages exceed the observed window")
            charges = [*measured, upper - sum(measured)]
            if entries[-1].get("actual_elapsed_s", 0) > charges[-1]:
                raise ValueError("measured final stage exceeds the observed window")
        for entry, charge in zip(entries, charges, strict=True):
            entry.update(closed_gpu_window=observation, charged_s=charge)
            if entry["status"] == "running":
                entry["status"] = "interrupted"
    ledger["elapsed_s"] = overrun + sum(
        entry.get("charged_s", entry["allocation_s"]) for entry in ledger["stages"]
    )
    ledger["accounting"] = "closed container upper bounds plus live/unverified reservations"
    ledger["budget_exhausted"] = ledger["elapsed_s"] >= TOTAL_SECONDS
    write_json(path, ledger)
    return ledger
