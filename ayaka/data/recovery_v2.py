"""Multi-stage date, invoice and exception evidence with offline verified traces.

This is authored training/diagnostic data, not JevBench or a natural-language
generalization guarantee. Translations and all questions share one case lineage.
"""

from __future__ import annotations

import calendar
import hashlib
import random
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

from .reasoning_v2 import SPLITS
from .schema import Candidate, Question, Sample


def verified_case(split, index):
    if split not in SPLITS or index < 0:
        raise ValueError("unknown split or negative case")
    variant = SPLITS.index(split)
    rng = random.Random(f"ayaka-recovery-1/{split}/{index}")
    family = ("temporal_numeric", "numeric", "rule_revision")[index % 3]
    year = 2028 + variant * 7 + index % 6
    month = rng.choice([1, 2, 3, 11, 12])
    day = rng.choice([1, 2, calendar.monthrange(year, month)[1]])
    offset = rng.choice([-8, -5, 1, 9, 12])
    local = datetime(
        year, month, day, rng.choice([0, 1, 22, 23]), 30, tzinfo=timezone(timedelta(hours=offset))
    )
    utc = local.astimezone(timezone.utc)
    facts = {
        "local": local.isoformat(),
        "utc": utc.isoformat(),
        "offset": offset,
        "family": family,
        "split": split,
        "index": index,
    }
    conversion = (
        f"Convert {local.isoformat()} to UTC by subtracting {offset} hours: {utc.isoformat()}."
    )
    if family == "temporal_numeric":
        required = rng.randint(2, 8)
        holidays = {utc.date() + timedelta(days=rng.randint(1, 12)) for _ in range(1 + variant % 3)}
        current, counted = utc.date(), []
        while len(counted) < required:
            current += timedelta(days=1)
            if current.weekday() < 5 and current not in holidays:
                counted.append(current)
        delta = rng.randint(-3, 6)
        actual = current + timedelta(days=delta)
        value = max(0, delta)
        facts.update(
            required=required,
            holidays=sorted(map(str, holidays)),
            deadline=str(current),
            actual=str(actual),
            value=value,
        )
        trace = (
            f"{conversion} Start from UTC date {utc.date()}, excluding that date. "
            f"Exclude Saturdays, Sundays and holidays {sorted(map(str, holidays))}. "
            f"Eligible dates in order: {', '.join(map(str, counted))}. "
            f"The {required}th eligible date is {current}. Delivery was {actual}. "
            f"Calendar-day difference is {delta}; lateness=max(0,{delta})={value}."
        )
    elif family == "numeric":
        cents = rng.randint(1200, 45000)
        quantity, discount, shipping = (
            rng.randint(1, 5),
            rng.choice([5, 10, 15, 20]),
            rng.randint(100, 1500),
        )
        effective = date(year, month, min(2, calendar.monthrange(year, month)[1]))
        old_tax, new_tax = rng.choice([5, 7, 9]), rng.choice([11, 13, 17])
        tax = new_tax if utc.date() >= effective else old_tax
        subtotal = Decimal(cents * quantity) / 100
        discounted = (subtotal * (100 - discount) / 100).quantize(
            Decimal(".01"), rounding=ROUND_HALF_UP
        )
        taxed = (discounted * (100 + tax) / 100).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
        value = int(taxed * 100) + shipping
        facts.update(
            unit_cents=cents,
            quantity=quantity,
            discount_percent=discount,
            shipping_cents=shipping,
            effective=str(effective),
            old_tax=old_tax,
            new_tax=new_tax,
            value=value,
        )
        trace = (
            f"{conversion} Invoice UTC date {utc.date()} is "
            f"{'on/after' if utc.date() >= effective else 'before'} {effective}, so tax={tax}%. "
            f"Items subtotal={cents}/100*{quantity}={subtotal}. "
            f"Apply {discount}% discount and decimal half-up cents: {discounted}. "
            f"Apply {tax}% tax to discounted items, half-up cents: {taxed}. "
            f"Shipping is untaxed and undiscounted: {shipping} cents. "
            f"Total integer cents={taxed}*100+{shipping}={value}."
        )
    else:
        effective = date(year, month, min(2, calendar.monthrange(year, month)[1]))
        credential, required = rng.randint(0, 5), 1 + variant % 4
        exception, override = bool(rng.getrandbits(1)), bool(rng.getrandbits(1))
        cost, old_limit, new_limit = (
            rng.randint(100, 1000),
            rng.randint(200, 500),
            rng.randint(600, 900),
        )
        limit = new_limit if utc.date() >= effective else old_limit
        authorized = override and credential >= required
        value = int(cost <= limit and (not exception or authorized))
        facts.update(
            effective=str(effective),
            credential=credential,
            required=required,
            exception=exception,
            override=override,
            cost=cost,
            old_limit=old_limit,
            new_limit=new_limit,
            value=value,
        )
        trace = (
            f"{conversion} Apply the rule for UTC date {utc.date()}: limit={limit}. "
            f"Budget test {cost}<={limit} is {cost <= limit}. "
            f"Override authorization is {override} and {credential}>={required}: {authorized}. "
            f"Exception test is not {exception} or {authorized}: {not exception or authorized}. "
            f"Both budget and exception tests must pass. Approval indicator={value}."
        )
    return facts, value, trace


def evidence(facts, language):
    family = facts["family"]
    descriptions = {
        "en": {
            "temporal_numeric": "Shipment timestamp is local. Count business days after its UTC date; exclude weekends and listed holidays. Delivery after the deadline is late; report nonnegative calendar days late.",
            "numeric": "Use the invoice UTC date to select the tax rule, effective inclusively. Multiply unit price by quantity, discount items, round half-up to cents, tax items and round half-up again. Add untaxed, undiscounted shipping. Report integer cents.",
            "rule_revision": "Select the limit by event UTC date; the new rule starts inclusively on the effective date. Approval requires cost within limit and no exception, unless an override with sufficient credential defeats the exception. An override never waives the cost limit. Report 1 for approved and 0 otherwise.",
        },
        "ko": {
            "temporal_numeric": "발송 시각은 현지 시각이다. UTC 날짜 다음 날부터 영업일을 센다. 주말과 명시된 휴일은 제외한다. 기한 이후 배송만 지연이며, 지연된 달력 일수를 0 이상으로 계산한다.",
            "numeric": "청구 시각의 UTC 날짜로 세율을 고른다. 시행일 당일부터 새 세율이다. 단가×수량에 할인을 적용하고 half-up으로 센트까지 반올림한다. 상품에 세금을 적용하고 다시 반올림한다. 할인·과세 대상이 아닌 배송비를 더해 정수 센트를 구한다.",
            "rule_revision": "사건의 UTC 날짜로 한도를 고른다. 시행일 당일부터 새 한도이다. 비용이 한도 이하여야 하고 예외가 없어야 승인된다. 단, 충분한 자격을 가진 override가 예외를 해제한다. override도 비용 한도를 면제하지 않는다. 승인 1, 거절 0이다.",
        },
        "ja": {
            "temporal_numeric": "発送時刻は現地時刻である。UTC日付の翌日から営業日を数える。週末と指定された休日は除外する。期限後の配達のみ遅延とし、遅延した暦日数を0以上で求める。",
            "numeric": "請求時刻のUTC日付で税率を選ぶ。施行日当日から新税率である。単価×数量を割引し、half-upでセントに丸める。商品に税を適用して再度丸める。割引・課税されない送料を加え、整数セントを求める。",
            "rule_revision": "事象のUTC日付で上限を選ぶ。施行日当日から新上限である。費用が上限以下で、例外がなければ承認する。十分な資格を持つoverrideは例外を解除できるが、費用上限は免除しない。承認1、拒否0とする。",
        },
    }
    hidden = {"utc", "deadline", "value", "family", "split", "index", "offset"}
    visible = {k: v for k, v in facts.items() if k not in hidden}
    # Each split uses a different document presentation; formulas remain shared.
    if facts["split"] in {"train", "router_train"}:
        return {
            "policy": descriptions[language][family],
            "record": visible,
            "unrelated_record": {"archived": True, "reference": "Q-19", "count": 42},
        }
    lines = [f"{key}: {visible[key]}" for key in sorted(visible, reverse=True)]
    return (
        f"{descriptions[language][family]}\nAudit memo\n"
        + "\n".join(lines)
        + "\nUnrelated archive count: 42."
    )


def recovery_curriculum(split, cases=128):
    if split not in SPLITS or type(cases) is not int or cases < 1:
        raise ValueError("positive cases and known split required")
    samples = []
    for index in range(cases):
        facts, value, trace = verified_case(split, index)
        lineage = hashlib.sha256(f"recovery-1/{split}/{index}".encode()).hexdigest()
        for language in ("en", "ko", "ja"):
            questions, traces = [], {}
            rng = random.Random(f"distractors/{split}/{index}")
            numbers = (
                [0, 1]
                if facts["family"] == "rule_revision"
                else list(range(max(0, value - 2), value + 3))
            )
            rng.shuffle(numbers)
            for kind in ("choice", "noul", "score"):
                if kind == "noul":
                    asked = value if index % 2 else rng.choice([n for n in numbers if n != value])
                    question = Question.noul(
                        kind, f"Is the requested result {asked}?", float(asked == value)
                    )
                    notes = (
                        trace
                        + f" Compare the computed result {value} with {asked}: {asked == value}."
                    )
                else:
                    candidates = [
                        Candidate(str(n), f"Result {n}", n if kind == "score" else None)
                        for n in numbers
                    ]
                    question = Question(
                        kind,
                        kind,
                        "Select the result required by the policy.",
                        candidates,
                        {str(n): float(n == value) for n in numbers},
                    )
                    notes = trace
                questions.append(question)
                traces[kind] = notes
            samples.append(
                Sample(
                    evidence(facts, language),
                    questions,
                    {
                        "source": "ayaka-recovery-verified-1",
                        "license": "MIT",
                        "split": split,
                        "language": language,
                        "modality": "text",
                        "source_lineage": lineage,
                        "source_example_id": f"recovery-1/{split}/{index}/{language}",
                        "task_family": facts["family"],
                        "generator_template_id": f"recovery/{split}/{facts['family']}",
                        "rule_combination": f"recovery/{split}/{facts['family']}",
                        "document_voice": f"recovery/{split}",
                        "verified_traces": traces,
                        "trace_validator": "ayaka.data.recovery_v2.verified_case",
                        "oracle_facts": facts,
                    },
                )
            )
    return samples
