"""Admission for one fixed pilot; listings are not a promise of execution time."""

from decimal import ROUND_CEILING, Decimal


def amount(value):
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("money must be finite and nonnegative")
    return result


def admit_offer(offer, *, credits, spend_cap="5", seconds=10800, preparation_reserve="0.30"):
    credit, cap, reserve = map(amount, (credits, spend_cap, preparation_reserve))
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
