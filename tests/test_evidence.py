import json
from decimal import Decimal

import pytest

from ayaka.evidence import (
    Calculator,
    EvidenceError,
    add_months,
    business_days,
    extraction_messages,
    parse_program,
    recover_grounded_quotes,
    validate_program,
)


def test_gross_weight_unit_conversion_and_boundary():
    state = "Cargo 5280 lb. Tare 105 kg. Straps 12.1 kg. Gross limit 2500 kg."
    out = validate_program(
        state,
        {
            "quotes": [state],
            "calculations": {
                "gross": "ceil(convert(5280,'lb','kg')+105+12.1)",
                "ok": "gross <= 2500",
            },
        },
    )
    assert out["calculations"]["gross"]["result"] == "2513"
    assert out["calculations"]["ok"]["result"] is False


def test_calendar_clamping_and_leap_year():
    assert add_months("2024-01-31", 1).isoformat() == "2024-02-29"
    assert add_months("2025-01-31", 1).isoformat() == "2025-02-28"
    assert business_days("2026-09-15", "2026-09-18") == 3


def test_exact_conditional_probability():
    c = Calculator(["Incident prevalence: 2%. Sensitivity 98%. False alarm rate 5%."])
    assert c.expression("(2*98)/(2*98+(100-2)*5)") == Decimal(196) / 686


def test_effective_date_branch_and_money_rounding():
    c = Calculator(["Amount 109.975. Limits: 110, 90. Starts 2026-09-10; request 2026-09-10."])
    assert c.expression("cents(109.975)") == Decimal("109.98")
    assert c.expression("if_(le(date('2026-09-10'),date('2026-09-10')),90,110)") == 90


@pytest.mark.parametrize(
    "expression",
    ["__import__('os')", "(1).__class__", "[x for x in [1]]", "open('x')", "2 ** 1000000"],
)
def test_no_external_execution(expression):
    with pytest.raises(EvidenceError):
        Calculator(["2 1000000"]).expression(expression)


def test_rejects_hallucinated_sources_and_operands():
    with pytest.raises(EvidenceError):
        validate_program("Weight 12 kg", {"quotes": ["Weight 99 kg"], "calculations": {}})
    with pytest.raises(EvidenceError):
        validate_program(
            "Weight 12 kg", {"quotes": ["Weight 12 kg"], "calculations": {"x": "12+99"}}
        )


def test_labels_and_private_metadata_never_enter_extraction():
    r = {
        "state": "Actual input",
        "question": {"type": "noul", "instructions": "Is it true?"},
        "labels": ["no", "yes"],
        "expected": "yes",
        "provenance": {"rationale": "SECRET"},
    }
    first = extraction_messages(r)
    r.update(expected="no", family="SECRET", id="SECRET", target={"yes": 1})
    assert first == extraction_messages(r)
    assert "SECRET" not in json.dumps(first)


def test_invalid_and_unknown_programs_are_rejected():
    with pytest.raises(EvidenceError):
        parse_program('{"quotes":[')
    with pytest.raises(EvidenceError):
        parse_program('{"quotes":[],"answer":"yes"}')


def test_unicode_citation_matches_the_actual_structured_request():
    state = {"memo": "보험 한도 — 12 kg", "weight": 12}
    verified = validate_program(
        state, {"quotes": ["보험 한도 — 12 kg"], "calculations": {"x": "12"}}
    )
    assert verified["calculations"]["x"]["result"] == "12"


def test_recovery_drops_unfinished_or_hallucinated_quotes_and_all_calculations():
    raw = '{"quotes":["Weight 12 kg", "Weight 99 kg", "Limit 15 kg", "Unfinished'
    recovered = recover_grounded_quotes("Weight 12 kg. Limit 15 kg.", raw)
    assert recovered["quotes"] == ["Weight 12 kg", "Limit 15 kg"]
    assert recovered["calculations"] == {}
    complete = '{"quotes":["Weight 12 kg"],"calculations":{"answer":"__import__(\'os\')"}}'
    assert recover_grounded_quotes("Weight 12 kg", complete)["calculations"] == {}


def test_recovery_requires_actual_source_support():
    with pytest.raises(EvidenceError):
        recover_grounded_quotes("Weight 12 kg", '{"quotes":["Weight 99 kg"]}')


def _prompt_example(system, source_prefix, output_prefix):
    lines = system.splitlines()
    source = next(ln for ln in lines if ln.startswith(source_prefix))
    output = next(ln for ln in lines if ln.startswith(output_prefix))
    return json.loads(source[len(source_prefix) :]), parse_program(output[len(output_prefix) :])


def test_prompt_examples_execute_and_share_no_ngram_with_public_benchmark():
    from ayaka.data.decontam import Decontaminator
    from ayaka.evidence import EXTRACTION_SYSTEM
    from ayaka.evidence_ids import SYSTEM

    source, program = _prompt_example(EXTRACTION_SYSTEM, "Example source: ", "Example output: ")
    out = validate_program(source, program)["calculations"]
    assert out["total"]["result"] == "220.32" and out["question_holds"]["result"] is True
    decon = Decontaminator.from_jevbench()
    assert not decon.text_hit(EXTRACTION_SYSTEM) and not decon.text_hit(SYSTEM)
