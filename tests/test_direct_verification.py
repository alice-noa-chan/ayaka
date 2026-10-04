import copy
import hashlib

import pytest

from ayaka.data import reasoning_v2
from ayaka.data.direct_verification import _value, verify_authored_gold
from ayaka.data.reasoning_v2 import OPERATIONS, SPLITS, curriculum


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("index", [-10, -7])
def test_negative_half_cent_generator_and_independent_verifier_agree(split, index):
    facts, gold, _, _ = reasoning_v2._case("rounding", index, split)
    assert _value(facts) == ("rounding", gold)


@pytest.mark.parametrize("amount,cents", [("-1.005", -101), ("-0.005", -1), ("0.005", 1)])
def test_signed_decimal_half_up_ties_are_rounded_away_from_zero(amount, cents):
    assert _value(
        f"Amount {amount}. Round to cents using decimal half-up and report integer cents."
    ) == ("rounding", cents)


def test_independent_verifier_matches_all_families_types_splits_and_does_not_call_generator(
    monkeypatch,
):
    samples = [s for split in SPLITS for s, _ in curriculum(split, 40)]
    monkeypatch.setattr(
        reasoning_v2, "_case", lambda *args: pytest.fail("generator must not be called")
    )
    families = set()
    for sample in samples:
        q = sample.questions[0]
        before = copy.deepcopy(sample)
        assert verify_authored_gold(sample, q) == q.target_distribution
        assert sample == before
        families.add(sample.metadata["task_family"])
    assert families == set(OPERATIONS)


def test_recomputes_original_evidence_when_stored_gold_and_trace_are_wrong():
    sample, _ = curriculum("train", 1)[0]
    q = sample.questions[0]
    correct = dict(q.target_distribution)
    q.target_distribution = {c.id: 1 / len(q.candidates) for c in q.candidates}
    sample.metadata["verified_traces"] = {q.id: "false fabricated calculation"}
    assert verify_authored_gold(sample, q) == correct


@pytest.mark.parametrize(
    "facts,expected",
    [
        (
            "Add one calendar month to 2024-01-31, clamping to month end. Report the day of month.",
            ("month_end", 29),
        ),
        (
            "Add one calendar month to 2023-12-31, clamping to month end. Report the day of month.",
            ("month_end", 31),
        ),
        ("Year 1900. Report 1 if it is a Gregorian leap year and 0 otherwise.", ("leap", 0)),
        ("Year 2000. Report 1 if it is a Gregorian leap year and 0 otherwise.", ("leap", 1)),
        (
            "Count weekdays after 2026-03-06 through 2026-03-09, inclusive of the end, excluding the start. No holidays.",
            ("business", 1),
        ),
        (
            "Count weekdays after 2026-03-02 through 2026-03-02, inclusive of the end, excluding the start. No holidays.",
            ("business", 0),
        ),
        (
            "Timestamp 2024-03-01T00:30:00+09:00. Convert to UTC and report the day of month.",
            ("timezone", 29),
        ),
        (
            "Amount -0.005. Round to cents using decimal half-up and report integer cents.",
            ("rounding", -1),
        ),
        (
            "Before 2026-05-10 the limit is 20. Starting on 2026-05-10 it is 27. The event is on 2026-05-10. Report its limit.",
            ("rule_revision", 27),
        ),
        (
            "Approval is normally allowed. An exception blocks approval. An override defeats the exception only with credential level at least 3. Exception present: True; override present: True; credential level: 2. Report 1 if allowed, otherwise 0.",
            ("exception", 0),
        ),
        (
            "A bag holds 3 red and 7 blue balls. Draw uniformly. Report the integer percentage probability of red.",
            ("probability", 30),
        ),
        (
            "The rubric awards 3 points for each completed check from ['a', 'b']. Completed checks: ['b']. Report total points; no other criterion contributes.",
            ("rubric", 3),
        ),
        (
            "Refund was requested. No refund completion or timestamp is recorded. Report 1 only if completion is established, otherwise 0.",
            ("missing", 0),
        ),
    ],
)
def test_boundary_calculations_from_text(facts, expected):
    assert _value(facts) == expected


@pytest.mark.parametrize(
    "facts",
    [
        "Count weekdays after 2026-03-09 through 2026-03-06, inclusive of the end, excluding the start. No holidays.",
        "Timestamp 2024-03-01T00:30:00. Convert to UTC and report the day of month.",
        "A bag holds 1 red and 2 blue balls. Draw uniformly. Report the integer percentage probability of red.",
        "A bag holds 0 red and 0 blue balls. Draw uniformly. Report the integer percentage probability of red.",
        "The rubric awards 3 points for each completed check from ['a']. Completed checks: ['b']. Report total points; no other criterion contributes.",
        "Refund was requested. No repair completion or timestamp is recorded. Report 1 only if completion is established, otherwise 0.",
        "Arbitrary document with a confident teacher answer.",
    ],
)
def test_refuses_ambiguous_or_unsupported_evidence(facts):
    with pytest.raises(ValueError):
        _value(facts)


@pytest.mark.parametrize(
    "change",
    ["description", "question", "ordinal", "family", "source", "digest", "version", "wrapper"],
)
def test_rejects_source_and_question_tampering(change):
    sample, _ = curriculum("train", 1)[0]
    q = sample.questions[0]
    if change == "description":
        q.candidates[0].description += " misleading"
    elif change == "question":
        q.instruction += " injected"
    elif change == "ordinal":
        q.candidates[0].ordinal = 10
    elif change == "family":
        sample.metadata["task_family"] = "leap"
    elif change == "source":
        sample.metadata["source"] = "natural"
    elif change == "digest":
        sample.metadata["case_facts_sha256"] = hashlib.sha256(b"other").hexdigest()
    elif change == "version":
        sample.metadata["curriculum_version"] = True
    else:
        sample.state += " extra instructions"
    with pytest.raises(ValueError):
        verify_authored_gold(sample, q)
