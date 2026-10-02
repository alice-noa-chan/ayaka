"""Independent authored rule combinations; never used for training or calibration.

Uses its own facts, arithmetic and rendering, not recovery_v2's oracle or traces.
It remains a synthetic evaluation, not a natural-language or JevBench guarantee.
"""

import calendar
import hashlib
import random
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from .schema import Candidate, Question, Sample


def holdout_case(split, index):
    if split not in {"dev", "test"} or type(index) is not int or index < 0:
        raise ValueError("holdout requires dev/test and a nonnegative integer case")
    rng = random.Random(f"independent-recovery-1/{split}/{index}")
    year = rng.randrange(2033, 2059) if split == "dev" else rng.randrange(2061, 2089)
    kind = index % 6
    facts = {"year": year, "kind": kind}
    if kind == 0:
        start = date(year, rng.choice([2, 3, 12]), 25)
        required = rng.randint(2, 7)
        weekends = {6} if split == "dev" else {4, 5}
        holidays = {start + timedelta(days=2), start + timedelta(days=5)}
        # Inclusive start differs from training's exclusive UTC-start rule.
        eligible = [
            start + timedelta(days=d)
            for d in range(40)
            if (start + timedelta(days=d)).weekday() not in weekends
            and start + timedelta(days=d) not in holidays
        ]
        deadline = eligible[required - 1]
        value = rng.randrange(7)
        delivered = deadline + timedelta(days=value)
        record = {
            "start": str(start),
            "required": required,
            "weekend_weekdays": sorted(weekends),
            "holidays": sorted(map(str, holidays)),
            "delivery": str(delivered),
        }
        facts.update(record, deadline=str(deadline))
        rule = "inclusive_business"
        domain = list(range(7))
    elif kind == 1:
        bucket = (index // 6) % 4
        years = range(2033, 2059) if split == "dev" else range(2061, 2089)
        if bucket < 2:
            year = rng.choice([y for y in years if calendar.isleap(y) == bool(bucket)])
        destination_month = (2, 2, 4, 7)[bucket]
        start_month = 1 if split == "dev" else 12
        months = destination_month - start_month
        start = date(year, start_month, 31)
        absolute = start.year * 12 + start.month - 1 + months
        y, m = divmod(absolute, 12)
        value = min(start.day, calendar.monthrange(y, m + 1)[1])
        record = {"start": str(start), "months": months}
        facts.update(record, year=year, destination_year=y, destination_month=m + 1)
        rule, domain = "month_clipping", [28, 29, 30, 31]
    elif kind in {2, 3}:
        unit, quantity = rng.randint(101, 9901), rng.randint(1, 7)
        shipping, discount, tax = (
            rng.randint(30, 1200),
            rng.choice([7, 13, 19]),
            rng.choice([6, 12, 18]),
        )
        cents = Decimal(unit * quantity)
        discounted = (cents * (100 - discount) / 100).quantize(Decimal(1), rounding=ROUND_HALF_UP)
        if kind == 2:
            # Shipping is taxable here, unlike the training rule.
            taxable = discounted + shipping
            value = int((taxable * (100 + tax) / 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))
            rule = "taxable_shipping"
        else:
            second = rng.choice([3, 8, 14])
            discounted = (discounted * (100 - second) / 100).quantize(
                Decimal(1), rounding=ROUND_HALF_UP
            )
            value = (
                int((discounted * (100 + tax) / 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))
                + shipping
            )
            rule = "serial_discount"
        record = {
            "unit_cents": unit,
            "quantity": quantity,
            "shipping_cents": shipping,
            "discount_percent": discount,
            "tax_percent": tax,
            "event_year": year,
        }
        if kind == 3:
            record["second_discount_percent"] = second
        facts.update(record)
        rank = rng.randrange(5)
        scale = max(20, value // 3)
        lower, upper = rng.sample(range(1, scale), rank), rng.sample(range(1, scale), 4 - rank)
        domain = [value, *(value - n for n in lower), *(value + n for n in upper)]
    else:
        cost, limit = rng.randint(100, 900), rng.randint(300, 800)
        override, credential, required = (
            bool(rng.getrandbits(1)),
            rng.randint(0, 4),
            rng.randint(1, 4),
        )
        cancelled = bool(rng.getrandbits(1))
        if kind == 4:
            # Cancellation is absolute and overrides only the credential test.
            value = int(not cancelled and cost <= limit and (override or credential >= required))
            rule = "absolute_cancellation"
            record = {
                "cost": cost,
                "limit": limit,
                "override": override,
                "credential": credential,
                "required": required,
                "cancelled": cancelled,
                "event_year": year,
            }
        else:
            # Explicit equiprobable missing fact: proper soft targets, not a guessed answer.
            value = {"0": 0.5, "1": 0.5}
            rule = "missing_fair_approval"
            record = {"approval_flag": "unobserved", "prior_true": 0.5, "event_year": year}
        domain = [0, 1]
        facts.update(record)
    facts.update(rule=rule, value=value)
    return record, facts, domain


POLICIES = {
    "en": {
        "inclusive_business": "Count the start date if eligible. Weekday numbering is Monday=0. Exclude the specified weekend days and holidays. Find the required eligible date. Report nonnegative calendar days late.",
        "month_clipping": "Shift the start date by the signed number of calendar months. Clip its day to the last valid day of the destination month. Report the destination day of month.",
        "taxable_shipping": "All amounts are integer cents. Discount items and round half-up to integer cents. Add shipping before tax. Tax that combined amount and round half-up to integer cents.",
        "serial_discount": "All amounts are integer cents. Apply the first then the second discount to items, rounding half-up to integer cents after each. Tax items and round again. Add untaxed shipping.",
        "absolute_cancellation": "Approve only if not cancelled, cost is within limit, and either override is true or credential meets required. Neither override nor credential waives cancellation or the cost limit. Approval=1, denial=0.",
        "missing_fair_approval": "The only rule is approval=1 if the unobserved flag is true, else 0. Its prior is explicitly 50/50 and there is no further evidence. Return the corresponding probability distribution.",
    },
    "ko": {
        "inclusive_business": "시작일도 유효하면 센다. 월요일=0이다. 지정된 주말과 휴일을 제외하고 required번째 유효 날짜를 찾는다. 지연된 달력 일수를 0 이상으로 답한다.",
        "month_clipping": "시작 날짜에 부호가 있는 달 수를 더한다. 도착 월에 해당 일이 없으면 그 월 말일로 제한한다. 도착 날짜의 일을 답한다.",
        "taxable_shipping": "금액은 정수 센트다. 상품을 할인하고 half-up으로 정수 반올림한다. 배송비를 먼저 더하고 합계에 과세한 다음 다시 반올림한다.",
        "serial_discount": "금액은 정수 센트다. 상품에 첫 할인과 두 번째 할인을 차례로 적용하며 매번 half-up 정수 반올림한다. 상품에 과세하고 다시 반올림한 뒤 비과세 배송비를 더한다.",
        "absolute_cancellation": "취소되지 않았고 비용이 한도 이하며 override가 참이거나 자격이 required 이상일 때만 승인한다. override도 취소나 비용 한도를 면제하지 않는다. 승인=1, 거절=0이다.",
        "missing_fair_approval": "관측되지 않은 플래그가 참이면 승인=1, 아니면 0이다. 사전확률은 명시적으로 반반이며 추가 증거는 없다. 해당 확률분포를 답한다.",
    },
    "ja": {
        "inclusive_business": "開始日も有効なら数える。月曜日=0である。指定の週末と休日を除きrequired番目の有効日を求める。遅延した暦日数を0以上で答える。",
        "month_clipping": "開始日に符号付きの月数を加える。移動先の月にその日がなければ月末日にする。移動後の日付の日を答える。",
        "taxable_shipping": "金額は整数セントである。商品を割引しhalf-upで整数に丸める。先に送料を加え、合計に課税して再度丸める。",
        "serial_discount": "金額は整数セントである。商品に第一と第二の割引を順に適用し毎回half-upで整数に丸める。商品に課税して再度丸め、非課税の送料を加える。",
        "absolute_cancellation": "取消されておらず費用が上限以内で、overrideが真または資格がrequired以上の場合のみ承認する。overrideも取消や費用上限を免除しない。承認=1、拒否=0である。",
        "missing_fair_approval": "未観測のフラグが真なら承認=1、そうでなければ0である。事前確率は明示的に半々で追加証拠はない。対応する確率分布を答える。",
    },
}


def independent_holdout(split, cases=240):
    if type(cases) is not int or cases < 1:
        raise ValueError("positive integer cases required")
    samples = []
    for index in range(cases):
        record, facts, domain = holdout_case(split, index)
        rng = random.Random(f"holdout-menus/{split}/{index}")
        rng.shuffle(domain)
        value = facts["value"]
        target = value if isinstance(value, dict) else {str(n): float(n == value) for n in domain}
        # Ask an independently sampled candidate; no parity-encoded binary label.
        hard_value = next((int(n) for n, p in target.items() if p == 1), None)
        asked = (
            (
                hard_value
                if rng.getrandbits(1)
                else rng.choice([n for n in domain if n != hard_value])
            )
            if hard_value is not None
            else rng.choice(domain)
        )
        truth = target.get(str(asked), 0)
        lineage = hashlib.sha256(f"independent-recovery-1/{split}/{index}".encode()).hexdigest()
        for language in ("en", "ko", "ja"):
            instruction = {
                "en": "Return the result under this rule.",
                "ko": "이 규칙에 따른 결과를 답하라.",
                "ja": "この規則に従う結果を答えよ。",
            }[language]
            proposition = {
                "en": f"Is the result {asked}?",
                "ko": f"결과가 {asked}인가?",
                "ja": f"結果は{asked}か。",
            }[language]
            questions = [Question.noul("noul", proposition, truth)]
            for kind in ("choice", "score"):
                questions.append(
                    Question(
                        kind,
                        kind,
                        instruction,
                        [Candidate(str(n), str(n), n if kind == "score" else None) for n in domain],
                        target,
                    )
                )
            state = (
                {"policy": POLICIES[language][facts["rule"]], "record": record}
                if split == "dev"
                else POLICIES[language][facts["rule"]]
                + "\n"
                + "\n".join(f"{k} = {v}" for k, v in reversed(list(record.items())))
            )
            samples.append(
                Sample(
                    state,
                    questions,
                    {
                        "source": "ayaka-independent-recovery-1",
                        "split": split,
                        "license": "MIT",
                        "language": language,
                        "modality": "text",
                        "task_family": "temporal_numeric"
                        if facts["kind"] < 2
                        else ("numeric" if facts["kind"] < 4 else "rule_revision"),
                        "source_lineage": lineage,
                        "source_example_id": f"independent-recovery-1/{split}/{index}/{language}",
                        "generator_template_id": f"independent/{split}/{facts['rule']}",
                        "rule_combination": f"independent/{split}/{facts['rule']}",
                        "document_voice": f"independent/{split}",
                        "semantic_rule": facts["rule"],
                        "evaluation_only": True,
                        "oracle_facts": facts,
                    },
                )
            )
    return samples
