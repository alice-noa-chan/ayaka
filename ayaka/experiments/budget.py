"""Explicitly reconcile reservations only after externally observed closure."""

import json
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
        index = observation["stage_index"]
        if type(index) is not int or not 0 <= index < len(ledger["stages"]):
            raise ValueError("closed-window reservation index is invalid")
        if index in indices or observation["container_id"] in containers:
            raise ValueError("each closed container/reservation can be reconciled only once")
        indices.add(index)
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
        entry = ledger["stages"][index]
        if any(
            i != index
            and other.get("closed_gpu_window", {}).get("container_id")
            == observation["container_id"]
            for i, other in enumerate(ledger["stages"])
        ):
            raise ValueError("a container cannot settle a second reservation")
        if entry.get("closed_gpu_window") not in (None, observation):
            raise ValueError("closed-window evidence cannot be replaced")
        if "actual_elapsed_s" in entry:
            raise ValueError(
                "reconcile measured completed stages through their existing accounting"
            )
        entry.update(
            status="interrupted",
            closed_gpu_window=observation,
            charged_s=(end - start).total_seconds() + overhead,
        )
    ledger["elapsed_s"] = overrun + sum(
        entry.get("charged_s", entry["allocation_s"]) for entry in ledger["stages"]
    )
    ledger["accounting"] = "closed container upper bounds plus live/unverified reservations"
    ledger["budget_exhausted"] = ledger["elapsed_s"] >= TOTAL_SECONDS
    write_json(path, ledger)
    return ledger
