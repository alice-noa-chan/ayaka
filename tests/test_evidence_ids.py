import pytest

from ayaka.evidence import EvidenceError
from ayaka.evidence_ids import index_source, oracle_id_plan, recover_id_evidence, validate_id_plan


def test_index_offsets_and_dates_do_not_become_numeric_parts():
    state = {"note": "Amount 10.25 EUR on 2026-09-27. Gross 9 kg."}
    index = index_source(state)
    assert all(index["source"][s["start"] : s["end"]] == s["text"] for s in index["spans"])
    assert any(v["value"] == "2026-09-27" for v in index["fields"].values())
    assert not any(v["value"] in ("2026", "27") for v in index["fields"].values())


@pytest.mark.parametrize("age,passes", [(29, False), (30, False), (31, True)])
def test_window_counterfactuals_keep_same_conditional_program(age, passes):
    import datetime as dt

    now = dt.date(2026, 9, 27)
    prior = now - dt.timedelta(days=age)
    quotes = [
        f"Now {now}; current 100 EUR.",
        f"Prior {prior}; 50 EUR.",
        "Window 30 days inclusive; cap 120 EUR.",
    ]
    state = "\n".join(quotes)
    oracle = {
        "quotes": quotes,
        "calculations": {
            "recent": f"le(days_between('{prior}','{now}'),30)",
            "total": "100+if_(recent,50,0)",
            "question_holds": "le(total,120)",
        },
    }
    plan = oracle_id_plan(state, oracle)
    verified = validate_id_plan(state, plan)
    assert "if_" in plan["c"]["total"]
    assert verified["calculations"]["question_holds"]["result"] is passes


def test_unselected_field_and_unknown_ids_rejected():
    state = "Amount 15 EUR.\nLimit 10 EUR."
    index = index_source(state)
    name = next(k for k, v in index["fields"].items() if v["span"] == 1)
    with pytest.raises(EvidenceError, match="unselected"):
        validate_id_plan(state, {"e": [0], "c": {"x": name}})
    with pytest.raises(EvidenceError, match="unknown source"):
        validate_id_plan(state, {"e": [200], "c": {}})


def test_malicious_calculation_and_field_shadowing_rejected():
    with pytest.raises(EvidenceError):
        validate_id_plan("Amount 15 EUR.", {"e": [0], "c": {"x": "__import__('os').system('x')"}})
    with pytest.raises(EvidenceError, match="shadows"):
        validate_id_plan("Amount 15 EUR.", {"e": [0], "c": {"n0": "1"}})


def test_long_span_quote_mapping_preserves_all_parts():
    quote = "Definition " + "legal entity and delegated authority " * 25 + "end."
    state = "Header\n" + quote
    plan = oracle_id_plan(state, {"quotes": [quote], "calculations": {}})
    verified = validate_id_plan(state, plan)
    assert len(plan["e"]) > 1
    assert "".join(verified["quotes"]).replace(" ", "") == quote.replace(" ", "")


def test_source_decimal_reference_does_not_round_through_float():
    source = "Amount 12345678901234567890.12345 units."
    out = validate_id_plan(source, {"e": [0], "c": {"amount": "n0"}})
    assert out["calculations"]["amount"]["result"] == "12345678901234567890.12345"


def test_recovered_ids_never_execute_failed_or_unfinished_calculations():
    raw = '{"e":[0,0],"c":{"question_holds":"__import__('
    recovered = recover_id_evidence("Limit 120 EUR.", raw)
    assert recovered["source_ids"] == [0]
    assert recovered["calculations"] == {}
    with pytest.raises(EvidenceError):
        recover_id_evidence("Limit 120 EUR.", '{"e":[0,888],"c":{}}')
    with pytest.raises(EvidenceError):
        recover_id_evidence("Limit 120 EUR.", '{"e":[0,')


def test_prompt_example_plan_is_valid_for_its_indexed_source():
    import json

    from ayaka.evidence_ids import SYSTEM, index_source

    source = "Subtotal 240.00.\nDiscount 15 percent.\nTax 8 percent after discount.\nBudget 225.00."
    fields = index_source(source)["fields"]
    assert [m["value"] for m in fields.values()] == ["240.00", "15", "8", "225.00"]
    plan = json.loads(next(ln for ln in SYSTEM.splitlines() if ln.startswith("Output: "))[8:])
    out = validate_id_plan(source, plan)["calculations"]
    assert out["total"]["result"] == "220.32" and out["question_holds"]["result"] is True
