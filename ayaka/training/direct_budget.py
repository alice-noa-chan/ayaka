"""Whole-workflow prepaid admission, including work outside the optimizer loop."""

from __future__ import annotations

import math
from decimal import ROUND_CEILING, Decimal

STAGES = (
    "environment_setup",
    "weight_download",
    "teacher_collection",
    "model_load",
    "preflight",
    "training",
    "evaluation",
    "export",
    "artifact_download",
    "teardown",
    "recovery",
)
VERSION = "ayaka-complete-prepaid-plan-1"


def admit_workflow(budget, *, measured_training_seconds=None, elapsed_seconds=0, completed=()):
    """Forecast completion of every stage; never shrink a schedule to fit credit.

    Rates, prepaid credit, billing quantum and stage allowances are explicit
    caller-provided inputs. The result is an estimate, not provider attestation
    or a launch authorization. Already billed time must be included in elapsed.
    """
    if (
        not isinstance(budget, dict)
        or set(budget)
        != {
            "version",
            "hourly_usd",
            "prepaid_usd",
            "other_reserved_usd",
            "billing_quantum_seconds",
            "safety_factor",
            "stages",
        }
        or budget["version"] != VERSION
    ):
        raise ValueError("an exact versioned complete prepaid budget is required")
    for key in ("hourly_usd", "prepaid_usd", "other_reserved_usd", "safety_factor"):
        value = budget[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(
                "budget rates, balances and margins must be finite nonnegative numbers"
            )
    quantum = budget["billing_quantum_seconds"]
    if type(quantum) is not int or quantum < 1 or budget["safety_factor"] < 1:
        raise ValueError("billing quantum must be positive and safety factor at least one")
    stages = budget["stages"]
    if not isinstance(stages, dict) or set(stages) != set(STAGES):
        raise ValueError("budget must explicitly include every setup/train/test/recovery stage")
    for stage in stages.values():
        if (
            not isinstance(stage, dict)
            or set(stage) != {"seconds", "basis"}
            or (
                type(stage["seconds"]) not in (int, float)
                or not math.isfinite(stage["seconds"])
                or stage["seconds"] < 0
                or not isinstance(stage["basis"], str)
                or not stage["basis"].strip()
            )
        ):
            raise ValueError("every stage needs finite nonnegative seconds and its evidence basis")
    if (
        type(elapsed_seconds) not in (int, float)
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0
    ):
        raise ValueError("already billed elapsed time must be finite and nonnegative")
    if set(completed) - set(STAGES) or len(set(completed)) != len(completed):
        raise ValueError("completed stages must be unique known workflow stages")
    forecast = {key: dict(value) for key, value in stages.items() if key not in completed}
    if measured_training_seconds is not None:
        if "training" not in forecast or (
            type(measured_training_seconds) not in (int, float)
            or not math.isfinite(measured_training_seconds)
            or measured_training_seconds <= 0
        ):
            raise ValueError("measured complete training forecast must be positive and pending")
        forecast["training"] = {
            "seconds": measured_training_seconds,
            "basis": "measured native backward/optimizer/checkpoint forecast for every fixed step",
        }
    seconds = (
        elapsed_seconds
        + math.fsum(v["seconds"] for v in forecast.values()) * budget["safety_factor"]
    )
    # Decimal avoids a binary float crossing a provider's billing boundary.
    billed = (
        int((Decimal(str(seconds)) / Decimal(quantum)).to_integral_value(rounding=ROUND_CEILING))
        * quantum
    )
    cost = Decimal(billed) * Decimal(str(budget["hourly_usd"])) / Decimal(3600)
    total = cost + Decimal(str(budget["other_reserved_usd"]))
    fits = total <= Decimal(str(budget["prepaid_usd"]))
    return {
        "version": VERSION,
        "fits": fits,
        "complete_workflow": True,
        "partial_curriculum": False,
        "stages": forecast,
        "elapsed_billed_seconds": elapsed_seconds,
        "safety_factor": budget["safety_factor"],
        "forecast_total_seconds": seconds,
        "rounded_billed_seconds": billed,
        "forecast_compute_usd": float(cost),
        "other_reserved_usd": budget["other_reserved_usd"],
        "forecast_total_usd": float(total),
        "prepaid_usd": budget["prepaid_usd"],
        "remaining_margin_usd": float(Decimal(str(budget["prepaid_usd"])) - total),
        "provider_verified": False,
        "limitations": "quoted caller inputs and sampled runtime forecast; no wall-clock guarantee or account verification",
    }
