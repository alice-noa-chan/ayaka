"""Repository-authored, deterministic traces; calculators run only offline.

This curriculum checks training mechanics. Its procedural families are not
evidence of performance on independent natural language or JevBench items.
"""

import hashlib
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

from ..evidence import add_months, business_days
from .schema import Candidate, Question, Sample

SPLITS = ("train", "router_train", "dev", "calibration", "test")
CURRICULUM_VERSION = 2
VOICES = {
    "train": "Record {i}. {facts} Determine the requested value.",
    "router_train": "Case file {i}: {facts} Apply the stated rule to this case.",
    "dev": "Review memo {i}\nFacts: {facts}\nUse only these facts.",
    "calibration": "Entry {i} lists the following evidence: {facts}",
    "test": "Independent audit {i} — {facts} What follows from this evidence?",
}


def _case(op, i, split):
    variant = SPLITS.index(split)
    a = 20 + i * 3 + variant
    if op == "month_end":
        start = f"{2024 + variant}-01-31"
        result = add_months(start, 1).day
        return (
            f"Add one calendar month to {start}, clamping to month end. Report the day of month.",
            result,
            f"The next month is February. Clamp day 31 to its last valid day, {result}.",
            31,
        )
    if op == "leap":
        year = 2000 + 100 * (i % 4) + variant
        result = int(year % 400 == 0 or (year % 4 == 0 and year % 100 != 0))
        return (
            f"Year {year}. Report 1 if it is a Gregorian leap year and 0 otherwise.",
            result,
            f"A leap year is divisible by 4, except centuries not divisible by 400. Applying the rule to {year} gives {result}.",
            1,
        )
    if op == "business":
        start = date(2026 + variant, 3, 2) + timedelta(days=i % 7)
        end = start + timedelta(days=8 + variant)
        result = business_days(start.isoformat(), end.isoformat())
        return (
            f"Count weekdays after {start} through {end}, inclusive of the end, excluding the start. No holidays.",
            result,
            f"Enumerate dates after the start through the end. Count Monday through Friday only: {result} days.",
            12,
        )
    if op == "timezone":
        local = datetime(
            2026 + variant, 3 + variant, 1, 0, 30, tzinfo=timezone(timedelta(hours=9 + variant))
        )
        utc = local.astimezone(timezone.utc)
        result = utc.day
        return (
            f"Timestamp {local.isoformat()}. Convert to UTC and report the day of month.",
            result,
            f"Subtract {9 + variant} hours: {utc.isoformat()}. The day of month is {result}.",
            31,
        )
    if op == "rounding":
        amount = Decimal(a) + Decimal("0.495")
        result = int(amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) * 100)
        return (
            f"Amount {amount}. Round to cents using decimal half-up and report integer cents.",
            result,
            f"Decimal half-up rounds {amount} to {amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}. Multiply by 100: {result} cents.",
            result + 100,
        )
    if op == "rule_revision":
        effective = date(2026, 5, 10 + variant)
        event = effective + timedelta(days=(i % 3) - 1)
        old, new = a, a + 7 + variant
        result = new if event >= effective else old
        return (
            f"Before {effective} the limit is {old}. Starting on {effective} it is {new}. The event is on {event}. Report its limit.",
            result,
            f"Compare event {event} with inclusive effective date {effective}. The {'new' if event >= effective else 'old'} rule applies, giving {result}.",
            new + 10,
        )
    if op == "exception":
        exception, override = i % 2 == 0, i % 3 == 0
        required, credential = variant + 1, i % 5 + 1
        result = int(not exception or (override and credential >= required))
        return (
            f"Approval is normally allowed. An exception blocks approval. An override defeats the exception only with credential level at least {required}. Exception present: {exception}; override present: {override}; credential level: {credential}. Report 1 if allowed, otherwise 0.",
            result,
            f"An override is authorized when {credential} >= {required}. Evaluate not {exception} or ({override} and {credential} >= {required}): {result}.",
            1,
        )
    if op == "probability":
        multiplier = variant + 1
        red = (a % 7 + 1) * multiplier
        total = 10 * multiplier
        blue = total - red
        result = red * 100 // total
        return (
            f"A bag holds {red} red and {blue} blue balls. Draw uniformly. Report the integer percentage probability of red.",
            result,
            f"Total balls: {red}+{blue}={total}. Probability red is {red}/{total}, or {result} percent.",
            100,
        )
    if op == "rubric":
        checks = [
            "acknowledge",
            "explain",
            "resolve",
            "verify",
            "summarize",
            "confirm",
            "follow up",
        ][: 3 + variant]
        completed = checks[: i % (len(checks) + 1)]
        weight = variant + 1
        points = len(completed) * weight
        return (
            f"The rubric awards {weight} points for each completed check from {checks}. Completed checks: {completed}. Report total points; no other criterion contributes.",
            points,
            f"Count the explicitly completed checks: {len(completed)}. Each contributes {weight} points. Total {points}.",
            len(checks) * weight,
        )
    # Unknown evidence is a proposition about what is established, not a guess.
    action = ("Delivery", "Payment", "Repair", "Cancellation", "Refund")[variant]
    return (
        f"{action} was requested. No {action.lower()} completion or timestamp is recorded. Report 1 only if completion is established, otherwise 0.",
        0,
        "A request does not establish completion. The record supplies no completion evidence, so the established-completion indicator is 0.",
        1,
    )


OPERATIONS = (
    "month_end",
    "leap",
    "business",
    "timezone",
    "rounding",
    "rule_revision",
    "exception",
    "probability",
    "rubric",
    "missing",
)


def curriculum(split, per_type=32):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    result = []
    for kind in ("choice", "noul", "score"):
        for i in range(per_type):
            op = OPERATIONS[i % len(OPERATIONS)]
            facts, value, trace, upper = _case(op, i, split)
            state = VOICES[split].format(i=i, facts=facts)
            if kind == "noul":
                asked = value if i % 2 == 0 else value + 1
                q = Question.noul("q", f"Is the requested value {asked}?", float(asked == value))
                trace += f" Compare {value} with the proposed value {asked}: {'equal' if asked == value else 'different'}."
            else:
                if kind == "score":
                    # Semantic numeric levels with sparse ordinals are valid score criteria.
                    numbers = sorted({max(0, value - 1), value, value + 1, max(upper, value + 2)})
                else:
                    numbers = [value + 1, value, value + 2, max(0, value - 1)]
                    numbers = list(dict.fromkeys(numbers))
                cands = [
                    Candidate(str(v), f"The requested value is {v}", v if kind == "score" else None)
                    for v in numbers
                ]
                q = Question(
                    "q",
                    kind,
                    "Select the requested value from the criteria.",
                    cands,
                    {str(v): float(v == value) for v in numbers},
                )
            sample = Sample(
                state,
                [q],
                {
                    "source": "ayaka-v2-verified",
                    "license": "MIT",
                    "split": split,
                    "language": "en",
                    "source_example_id": f"{split}/{kind}/{i}",
                    "task_family": op,
                    "generator_template_id": f"{op}/{split}",
                    "rule_combination": f"{op}/variant-{SPLITS.index(split)}",
                    "document_voice": split,
                    "trace_validator": "ayaka.data.reasoning_v2._case",
                    "curriculum_version": CURRICULUM_VERSION,
                    "case_facts_sha256": hashlib.sha256(facts.encode()).hexdigest(),
                },
            )
            result.append((sample, {"q": trace}))
    return result
