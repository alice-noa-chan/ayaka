"""Recompute authored curriculum gold from original text, only during preparation.

This verifier intentionally supports a small versioned grammar. It neither
reads stored gold/traces nor calls the generator's calculation functions.
Unsupported natural-language documents need a separate reviewed verifier.
It must not be installed as an online calculator in the decision path.
"""

from __future__ import annotations

import ast
import hashlib
import re
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

VERSION = "ayaka-authored-direct-gold-1"
WRAPPERS = {
    "train": r"Record [0-9a-f]{8}\. (.+) Determine the requested value\.",
    "router_train": r"Case file [0-9a-f]{8}: (.+) Apply the stated rule to this case\.",
    "dev": r"Review memo [0-9a-f]{8}\nFacts: (.+)\nUse only these facts\.",
    "calibration": r"Entry [0-9a-f]{8} lists the following evidence: (.+)",
    "test": r"Independent audit [0-9a-f]{8} — (.+) What follows from this evidence\?",
}
INTEGER = r"(?:0|-?[1-9][0-9]*)"
ISO_DATE = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"


def _value(facts):
    m = re.fullmatch(
        rf"Add one calendar month to ({ISO_DATE}), clamping to month end\. Report the day of month\.",
        facts,
    )
    if m:
        start = date.fromisoformat(m[1])
        following = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
        after = date(following.year + (following.month == 12), following.month % 12 + 1, 1)
        return "month_end", min(start.day, (after - timedelta(days=1)).day)
    m = re.fullmatch(
        r"Year ([0-9]{4})\. Report 1 if it is a Gregorian leap year and 0 otherwise\.", facts
    )
    if m:
        # Calendar construction is independent of the generator's modular test.
        return "leap", int((date(int(m[1]), 2, 28) + timedelta(days=1)).month == 2)
    m = re.fullmatch(
        rf"Count weekdays after ({ISO_DATE}) through ({ISO_DATE}), inclusive of the end, excluding the start\. No holidays\.",
        facts,
    )
    if m:
        start, end = date.fromisoformat(m[1]), date.fromisoformat(m[2])
        days = (end - start).days
        if not 0 <= days <= 36600:
            raise ValueError("weekday interval is reversed or outside verifier bounds")
        weeks, remainder = divmod(days, 7)
        extra = sum((start + timedelta(days=i)).weekday() < 5 for i in range(1, remainder + 1))
        return "business", weeks * 5 + extra
    m = re.fullmatch(r"Timestamp (\S+)\. Convert to UTC and report the day of month\.", facts)
    if m:
        local = datetime.fromisoformat(m[1])
        if local.tzinfo is None:
            raise ValueError("timezone calculation requires an explicit offset")
        return "timezone", datetime.fromtimestamp(local.timestamp(), timezone.utc).day
    m = re.fullmatch(
        r"Amount (-?[0-9]+(?:\.[0-9]+)?)\. Round to cents using decimal half-up and report integer cents\.",
        facts,
    )
    if m:
        return "rounding", int((Decimal(m[1]) * 100).to_integral_value(rounding=ROUND_HALF_UP))
    m = re.fullmatch(
        rf"Before ({ISO_DATE}) the limit is ({INTEGER})\. Starting on ({ISO_DATE}) it is ({INTEGER})\. The event is on ({ISO_DATE})\. Report its limit\.",
        facts,
    )
    if m:
        if m[1] != m[3]:
            raise ValueError("rule boundaries must describe the same effective date")
        effective, event = date.fromisoformat(m[1]), date.fromisoformat(m[5])
        return "rule_revision", int(m[4] if event >= effective else m[2])
    m = re.fullmatch(
        r"Approval is normally allowed\. An exception blocks approval\. An override defeats the exception only with credential level at least ([0-9]+)\. Exception present: (True|False); override present: (True|False); credential level: ([0-9]+)\. Report 1 if allowed, otherwise 0\.",
        facts,
    )
    if m:
        allowed = True
        if m[2] == "True":
            allowed = m[3] == "True" and int(m[4]) >= int(m[1])
        return "exception", int(allowed)
    m = re.fullmatch(
        r"A bag holds ([0-9]+) red and ([0-9]+) blue balls\. Draw uniformly\. Report the integer percentage probability of red\.",
        facts,
    )
    if m:
        red, blue = int(m[1]), int(m[2])
        if red + blue == 0 or 100 * red % (red + blue):
            raise ValueError("integer probability requires nonempty evidence and exact percentage")
        return "probability", 100 * red // (red + blue)
    m = re.fullmatch(
        r"The rubric awards ([0-9]+) points for each completed check from (\[.*\])\. Completed checks: (\[.*\])\. Report total points; no other criterion contributes\.",
        facts,
    )
    if m:
        checks, completed = ast.literal_eval(m[2]), ast.literal_eval(m[3])
        for values in (checks, completed):
            if (
                not isinstance(values, list)
                or len(values) > 64
                or any(not isinstance(v, str) or not v for v in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError("rubric checks must be bounded unique text lists")
        if not checks or set(completed) - set(checks):
            raise ValueError("completed rubric checks must be stated criteria")
        return "rubric", int(m[1]) * sum(check in completed for check in checks)
    m = re.fullmatch(
        r"(Delivery|Payment|Repair|Cancellation|Refund) was requested\. No ([a-z]+) completion or timestamp is recorded\. Report 1 only if completion is established, otherwise 0\.",
        facts,
    )
    if m:
        if m[1].lower() != m[2]:
            raise ValueError("missing-evidence statements must refer to the same action")
        return "missing", 0
    raise ValueError("unsupported authored evidence grammar; refuse inferred gold")


def verify_authored_gold(sample, question):
    """Return a fresh full gold distribution, never sample.target_distribution."""
    metadata = sample.metadata
    if not isinstance(metadata, dict):
        raise ValueError("authored provenance metadata is required")
    split = metadata.get("split")
    if (
        metadata.get("source") != "ayaka-v2-verified"
        or metadata.get("license") != "MIT"
        or type(metadata.get("curriculum_version")) is not int
        or metadata["curriculum_version"] != 3
        or not isinstance(split, str)
        or split not in WRAPPERS
        or not isinstance(sample.state, str)
        or len(sample.state) > 4096
    ):
        raise ValueError("unsupported authored source, split or curriculum version")
    wrapper = re.fullmatch(WRAPPERS[split], sample.state)
    if not wrapper:
        raise ValueError("document must match its versioned source wrapper")
    facts = wrapper[1]
    if metadata.get("case_facts_sha256") != hashlib.sha256(facts.encode()).hexdigest():
        raise ValueError("source fact fingerprint mismatch")
    family, value = _value(facts)
    if metadata.get("task_family") != family:
        raise ValueError("declared family disagrees with interpreted evidence")
    if not sample.questions or len(sample.questions) != 1 or question.id != sample.questions[0].id:
        raise ValueError("authored verifier requires the single source question")
    candidates = question.candidates
    if question.type == "noul":
        match = re.fullmatch(rf"Is the requested value ({INTEGER})\?", question.instruction)
        if (
            not match
            or {c.id for c in candidates} != {"false", "true"}
            or len(candidates) != 2
            or any(c.description != c.id or c.ordinal is not None or c.is_nota for c in candidates)
        ):
            raise ValueError("unsupported authored Noul question")
        p_true = float(int(match[1]) == value)
        return {c.id: p_true if c.id == "true" else 1 - p_true for c in candidates}
    if (
        question.type not in {"choice", "score"}
        or question.instruction != "Select the requested value from the criteria."
    ):
        raise ValueError("unsupported authored selection instruction")
    for c in candidates:
        if (
            not isinstance(c.id, str)
            or not re.fullmatch(INTEGER, c.id)
            or c.description != f"The requested value is {c.id}"
            or c.is_nota
        ):
            raise ValueError("numeric candidate descriptions must match their value identifiers")
        if question.type == "score" and (
            type(c.ordinal) not in (int, float, Decimal)
            or (isinstance(c.ordinal, Decimal) and not c.ordinal.is_finite())
            or c.ordinal != int(c.id)
        ):
            raise ValueError("Score ordinals must match numeric candidate meanings")
        if question.type == "choice" and c.ordinal is not None:
            raise ValueError("Choice candidates must not carry Score ordinals")
    if len(candidates) < 2 or not any(int(c.id) == value for c in candidates):
        raise ValueError("the verified value must be in a nonempty decision set")
    return {c.id: float(int(c.id) == value) for c in candidates}
