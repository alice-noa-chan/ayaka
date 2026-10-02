"""Admission for one fixed pilot; listings are not a promise of execution time."""

from decimal import ROUND_CEILING, Decimal


def amount(value):
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("money must be finite and nonnegative")
    return result


def admit_offer(
    offer, *, credits, managed_credits, spend_cap="5", seconds=10800, preparation_reserve="0.30"
):
    credit, cap, reserve = map(amount, (credits, spend_cap, preparation_reserve))
    if amount(managed_credits) < 25:
        raise ValueError(
            "managed compute needs $25 eligible credit; serverless credit does not apply"
        )
    if type(seconds) is not int or not 0 < seconds <= 10800:
        raise ValueError("require a positive reservation of at most three hours")
    if offer.get("gpu") != "A100-80" or int(offer.get("gpu_count", 0)) != 1:
        raise ValueError("require one full A100 80GB")
    if int(offer.get("available", 0)) < 1 or not all(
        offer.get(key) for key in ("id", "provider", "region", "cloud")
    ):
        raise ValueError("require an available, identified offer")
    if int(offer.get("memory_mb", 0)) < 65536:
        raise ValueError("require at least 64GiB host memory")
    rate = amount(offer["hourly_cost_micros"]) / 1_000_000
    if not rate:
        raise ValueError("a missing or zero quote is not free capacity")
    reservation = (rate * seconds / 3600).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    total = reservation + reserve
    if total > min(credit, cap):
        raise ValueError("complete reservation plus preparation exceeds available budget")
    return {
        "scope": "fixed 200-step clean pilot, evaluation and artifact receipt",
        "offer": offer,
        "credits_usd": str(credit),
        "managed_credits_usd": str(amount(managed_credits)),
        "hourly_usd": str(rate),
        "reservation_seconds": seconds,
        "reservation_ceiling_usd": str(reservation),
        "preparation_reserve_usd": str(reserve),
        "planned_ceiling_usd": str(total),
        "spend_cap_usd": str(cap),
        "automatic_extension": False,
        "automatic_retry": False,
        "full_training": False,
    }


def admit_serverless(inventory, *, credits, spend_cap, seconds=9700):
    """Use the observed GPU-attached rates, including CPU and RAM, for one task."""
    credit, cap = map(amount, (credits, spend_cap))
    if inventory.get("gpu") != "RTX5090" or inventory.get("serverless") != "ready":
        raise ValueError("require currently ready serverless RTX5090 capacity")
    if type(seconds) is not int or not 0 < seconds <= 9700:
        raise ValueError("require a finite serverless task of at most 9700 seconds")
    gpu_rate = Decimal("0.000303")
    cpu_rate = Decimal("0.000105")
    ram_rate = Decimal("0.0000055")
    rate = gpu_rate + 2 * cpu_rate + 32 * ram_rate
    compute = (rate * seconds).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    total = compute + Decimal("0.10")
    if total > min(credit, cap):
        raise ValueError("complete GPU+CPU+RAM task and margin exceed the remaining budget")
    return {
        "scope": "fixed 200-step clean pilot, evaluation and durable artifact receipt",
        "compute_class": "serverless",
        "gpu": "RTX5090",
        "cpu_cores": 2,
        "memory_gib": 32,
        "task_timeout_seconds": seconds,
        "hourly_usd": str(rate * 3600),
        "compute_ceiling_usd": str(compute),
        "planned_ceiling_usd": str(total),
        "credit_usd": str(credit),
        "remaining_spend_cap_usd": str(cap),
        "rate_source": "https://www.beam.cloud/pricing",
        "rate_checked_on": "2026-10-02",
        "automatic_retry": False,
        "automatic_extension": False,
        "full_training": False,
    }
