import pytest

from scripts.beam_v2.budget import admit_offer, admit_serverless


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
    plan = admit_offer(offer(), credits="7.032868", managed_credits="25")
    assert plan["reservation_ceiling_usd"] == "4.56"
    assert plan["planned_ceiling_usd"] == "4.86"
    assert not plan["automatic_extension"]


def test_small_remaining_balance_refuses_before_allocation():
    with pytest.raises(ValueError, match="exceeds"):
        admit_offer(offer(), credits="4.55999", managed_credits="25")


def test_budget_reserves_preparation_and_rounds_up_fractional_cents():
    with pytest.raises(ValueError, match="exceeds"):
        admit_offer(
            offer(hourly_cost_micros="1666666"),
            credits="7.03",
            managed_credits="25",
            spend_cap="5",
        )


@pytest.mark.parametrize("credit", ["NaN", "Infinity", "-1"])
def test_invalid_money_is_not_an_unlimited_budget(credit):
    with pytest.raises(ValueError):
        admit_offer(offer(), credits=credit, managed_credits="25")


@pytest.mark.parametrize(
    "changes",
    [{"gpu_count": 4}, {"gpu": "A100-40"}, {"available": 0}, {"hourly_cost_micros": "0"}],
)
def test_unsuitable_or_unavailable_offers_are_rejected(changes):
    with pytest.raises(ValueError):
        admit_offer(offer(**changes), credits="7.03", managed_credits="25")


def test_serverless_balance_cannot_authorize_managed_reservation():
    with pytest.raises(ValueError, match="eligible credit"):
        admit_offer(offer(), credits="7.03", managed_credits="0")


def test_serverless_budget_includes_gpu_attached_cpu_ram_and_margin():
    row = {"gpu": "RTX5090", "serverless": "ready"}
    plan = admit_serverless(row, credits="6.976841", spend_cap="6.793973")
    assert plan["hourly_usd"] == "2.4804000"
    assert plan["compute_ceiling_usd"] == "6.69"
    assert plan["planned_ceiling_usd"] == "6.79"
    with pytest.raises(ValueError, match="exceed"):
        admit_serverless(row, credits="6.68", spend_cap="6.793973")


def test_serverless_refuses_unavailable_or_changed_resources():
    for row in (
        {"gpu": "H100", "serverless": "none"},
        {"gpu": "RTX5090", "serverless": "none"},
    ):
        with pytest.raises(ValueError, match="capacity"):
            admit_serverless(row, credits="7", spend_cap="7")
