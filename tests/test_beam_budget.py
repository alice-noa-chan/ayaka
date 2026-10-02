import pytest

from scripts.beam_v2.budget import admit_offer


def offer(**changes):
    return {
        "id": "A100_sxm4_80G_DGX",
        "provider": "shadeform",
        "region": "beltsville-usa-1",
        "cloud": "massedcompute",
        "gpu": "A100-80",
        "gpu_count": 1,
        "available": 2,
        "memory_mb": "163840",
        "hourly_cost_micros": "1518000",
        **changes,
    }


def test_whole_reservation_and_preparation_fit_without_spending_all_credit():
    plan = admit_offer(offer(), credits="7.032868")
    assert plan["reservation_ceiling_usd"] == "4.56"
    assert plan["planned_ceiling_usd"] == "4.86"
    assert not plan["automatic_extension"]


def test_small_remaining_balance_refuses_before_allocation():
    with pytest.raises(ValueError, match="exceeds"):
        admit_offer(offer(), credits="4.55999")


def test_budget_reserves_preparation_and_rounds_up_fractional_cents():
    with pytest.raises(ValueError, match="exceeds"):
        admit_offer(offer(hourly_cost_micros="1666666"), credits="7.03", spend_cap="5")


@pytest.mark.parametrize("credit", ["NaN", "Infinity", "-1"])
def test_invalid_money_is_not_an_unlimited_budget(credit):
    with pytest.raises(ValueError):
        admit_offer(offer(), credits=credit)


@pytest.mark.parametrize(
    "changes",
    [{"gpu_count": 4}, {"gpu": "A100-40"}, {"available": 0}, {"hourly_cost_micros": "0"}],
)
def test_unsuitable_or_unavailable_offers_are_rejected(changes):
    with pytest.raises(ValueError):
        admit_offer(offer(**changes), credits="7.03")
