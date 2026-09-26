"""Independent, procedural compositional decisions with exact reference labels.

No benchmark text or API output is used. Long cases spread decisive facts
across aliases, dated FX tables, a ledger, rule revisions and a routing matrix.
Short cases cover calendar-month expiry, fixed-offset conversion and exact
conditional / without-replacement probabilities. Metadata identifies scenario
groups for held-out rule-combination evaluation.
"""

from __future__ import annotations

import calendar
import datetime as dt
import math
import random
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

from .schema import Candidate, Question, Sample, one_hot

CENT = Decimal("0.01")
DOMAINS = (
    "laboratory supplies",
    "fleet maintenance",
    "warehouse equipment",
    "publishing services",
    "campus facilities",
    "retail logistics",
    "network operations",
    "field instrumentation",
)
TIERS = ["team lead", "department manager", "finance director", "executive committee"]


def choice(
    rng: random.Random, qid: str, instruction: str, options: list[str], correct: str
) -> Question:
    options = list(dict.fromkeys(options))
    rng.shuffle(options)
    candidates = [Candidate(f"o{i}", text) for i, text in enumerate(options)]
    gold = next(c.id for c in candidates if c.description == correct)
    return Question(qid, "choice", instruction, candidates, one_hot(candidates, gold))


def money(value: Decimal) -> Decimal:
    return value.quantize(CENT, ROUND_HALF_UP)


def long_rules_sample(rng: random.Random, idx: int) -> Sample:
    """Alias -> country override -> effective window -> FX -> sum -> tier."""
    domain = rng.choice(DOMAINS)
    current = dt.date(2024, 1, 1) + dt.timedelta(days=rng.randrange(1800))
    amended = rng.choice([False, True])
    effective = current + dt.timedelta(days=15 if not amended else -15)
    window = 30 if amended else 14
    scale = rng.choice([1, 2, 3])
    elevated = rng.choice([False, True])
    vendor = f"V-{idx + 10000}"
    po = f"P-{idx + 20000}"
    alias = f"Cedar Relay {idx} Ltd"
    currency = rng.choice(["USD", "GBP", "CHF"])
    thresholds = [
        Decimal(x * scale) for x in ((1000, 5000, 15000) if elevated else (2000, 10000, 25000))
    ]
    tier = rng.randrange(4)
    lower = Decimal(300) if tier == 0 else thresholds[tier - 1] + 300
    upper = thresholds[tier] - 100 if tier < 3 else thresholds[-1] + 15000
    desired = Decimal(rng.randint(int(lower), int(upper)))
    dates = [current + dt.timedelta(days=d) for d in (0, -1, -7, -20, -35, 3)]
    fx = {
        (day, cur): Decimal(rng.randint(65, 160)) / 100
        for day in dates
        for cur in ("USD", "GBP", "CHF")
    }
    prior = []
    for j, days in enumerate((7, 20)):
        day = current - dt.timedelta(days=days)
        foreign = money(Decimal(rng.randint(50, 100)) * fx[day, currency])
        prior.append((f"L-{idx}-prior{j}", vendor, po, day, currency, foreign))
    prior_eur = sum(
        (money(row[5] / fx[row[3], row[4]]) for row in prior if (current - row[3]).days <= window),
        Decimal(0),
    )
    amount = money((desired - prior_eur) * fx[current, currency])
    current_eur = money(amount / fx[current, currency])
    aggregate = current_eur + prior_eur
    tier = sum(aggregate > cutoff for cutoff in thresholds)

    vendors = [(vendor, alias, "US" if elevated else "DE")]
    for j in range(rng.randint(12, 20)):
        # Similar legal names deliberately belong to different vendor IDs.
        name = f"Cedar Relay {idx} LLC" if j == 0 else f"Cedar Unit {idx}-{j} Ltd"
        vendors.append((f"N-{idx}-{j}", name, rng.choice(["DE", "FR", "US", "CH"])))
    rng.shuffle(vendors)
    ledger = list(prior)
    for j in range(rng.randint(20, 32)):
        vid = rng.choice([row[0] for row in vendors])
        order = f"OTHER-{j}" if vid == vendor else po
        ledger.append(
            (
                f"L-{idx}-{j}",
                vid,
                order,
                rng.choice(dates),
                rng.choice(["USD", "GBP", "CHF"]),
                Decimal(rng.randint(1000, 500000)) / 100,
            )
        )
    rng.shuffle(ledger)
    vendor_table = "\n".join(f"Vendor {vid} | {name} | {country}" for vid, name, country in vendors)
    ledger_table = "\n".join(
        f"Invoice {inv} | {vid} | {order} | {day.isoformat()} | {cur} | {value:.2f}"
        for inv, vid, order, day, cur, value in ledger
    )
    fx_rows = list(fx.items())
    rng.shuffle(fx_rows)
    fx_table = "\n".join(
        f"FX {day.isoformat()} | {cur} | {rate:.2f}" for (day, cur), rate in fx_rows
    )
    matrix = "\n".join(
        f"Tier {i + 1} ({name}): STANDARD <= {a * scale} EUR; ELEVATED <= {b * scale} EUR."
        for i, (name, a, b) in enumerate(
            zip(TIERS[:3], (2000, 10000, 25000), (1000, 5000, 15000), strict=True)
        )
    )
    sections = [
        "VENDOR REGISTER (exact legal trading name | bank country)\n" + vendor_table,
        "INVOICE LEDGER (invoice | vendor ID | purchase order | date | currency | amount)\n"
        + ledger_table,
        "TREASURY REFERENCE TABLE (date | currency | foreign units per EUR)\n" + fx_table,
        "APPROVAL MATRIX\n"
        + matrix
        + "\nTier 4 (executive committee): all larger amounts. Use the first tier whose upper bound includes the amount.",
        f"AGGREGATION REVISIONS\nWindow version 1: effective 2020-01-01; days 14.\nWindow version 2: effective {effective.isoformat()}; days 30.\nUse the latest version effective on the current invoice date. Include earlier invoices from the SAME vendor ID and SAME purchase order in the preceding window, including its first day and excluding the current day. Add the current invoice once. Future-dated records are excluded.",
    ]
    operational = []
    for j in range(rng.randint(2, 6)):
        operational.append(
            f"Control {j + 1} for {rng.choice(DOMAINS)}: records filed under purchase order Q-{idx}-{j} "
            f"require reviewer desk {rng.randint(1, 9)} and archive class {rng.choice(['A', 'B', 'C'])}. "
            "This control changes document retention, not the approval tier or currency basis. "
            "The receipt date is used for archive retention; the invoice date is used for treasury conversion. "
            "A credit against a different purchase order is reconciled separately and cannot reduce this order's routing amount."
        )
    sections.append("OTHER OPERATING CONTROLS\n" + "\n".join(operational))
    rng.shuffle(sections)
    state = (
        f"PROCUREMENT CONTROL MANUAL: {domain}\nCase C-{idx}\n"
        "Match the trading name exactly, including the legal suffix. All vendor master risks are STANDARD. "
        "Approved bank countries: DE, FR, NL. A bank outside that list raises the risk to ELEVATED. "
        "Convert EACH invoice using foreign amount divided by its own date's FX rate; round each converted amount to cents before summing. "
        "The matrix uses this aggregate EUR amount and the vendor's final risk.\n\n"
        + "\n\n".join(sections)
        + f"\n\nCURRENT PAYABLE\nTrading name: {alias}\nPurchase order: {po}\nInvoice date: {current.isoformat()}\nCurrency: {currency}\nAmount: {amount:.2f}\n"
    )
    alternatives = [
        f"{v:.2f} EUR"
        for v in (
            aggregate,
            current_eur,
            aggregate + Decimal(500),
            money(amount * fx[current, currency]) + prior_eur,
        )
    ]
    questions = [
        choice(
            rng,
            "routing",
            "Which approval tier must handle the current payable?",
            TIERS,
            TIERS[tier],
        ),
        choice(
            rng,
            "aggregate",
            "What aggregate EUR amount is used for routing the current payable?",
            alternatives,
            f"{aggregate:.2f} EUR",
        ),
        Question.noul(
            "senior",
            "Does the current payable require the finance director or the executive committee?",
            float(tier >= 2),
        ),
    ]
    return Sample(
        state,
        questions,
        {
            "generator": "long_rules",
            "source_example_id": f"long-rules-{idx}",
            "split_group": f"long-rules:{domain}:{window}:{scale}:{elevated}",
        },
    )


def calendar_sample(rng: random.Random, idx: int) -> Sample:
    year, month = rng.randint(2023, 2029), rng.randint(1, 12)
    start = dt.date(year, month, calendar.monthrange(year, month)[1])
    months = rng.randint(1, 24)
    absolute = start.year * 12 + start.month - 1 + months
    end_year, end_month = absolute // 12, absolute % 12 + 1
    end = dt.date(end_year, end_month, min(start.day, calendar.monthrange(end_year, end_month)[1]))
    issuer_offset, customer_offset = rng.sample([-8, -5, -3, 0, 1, 3, 5, 8, 10], 2)
    cutoff = dt.datetime.combine(end + dt.timedelta(days=1), dt.time())
    delta = (
        rng.choice([1, 30, 60, 61, 300, 1440, 1441]) * -1
        if idx % 2 == 0
        else rng.choice([0, 1, 30, 60, 61, 300, 1440, 1441])
    )
    issuer_arrival = cutoff + dt.timedelta(minutes=delta)
    customer_arrival = issuer_arrival + dt.timedelta(hours=customer_offset - issuer_offset)
    state = (
        f"SERVICE COVER CERTIFICATE SC-{idx}\nActivation date: {start.isoformat()}\nTerm: {months} calendar months\n"
        "Expiry rule: add the term to the activation month, keeping its day number. If that day does not exist in the destination month, use that month's last day. "
        "Coverage ends at 00:00 on the day AFTER that expiry date, in the issuer's clock. A submission exactly at the cutoff is late.\n"
        f"Issuer fixed offset: UTC{issuer_offset:+d}\nCustomer fixed offset: UTC{customer_offset:+d}\n"
        "These fixed offsets apply throughout this certificate; no daylight-saving adjustment is permitted.\n"
        f"Submission in customer clock: {customer_arrival.isoformat(timespec='minutes')}\n"
        "A draft saved locally is not a submission. Registration occurs when the complete form reaches the portal. The timestamp above records that completed submission.\n"
    )
    offsets = rng.sample([-3, -2, -1, 1, 2, 3], 3)
    dates = [(end + dt.timedelta(days=d)).isoformat() for d in [0, *offsets]]
    levels = [
        "before cutoff",
        "late by at most 60 minutes",
        "late by more than 60 minutes but at most 24 hours",
        "late by more than 24 hours",
    ]
    bucket = 0 if delta < 0 else 1 if delta <= 60 else 2 if delta <= 1440 else 3
    candidates = [Candidate(f"s{i}", text, ordinal=i) for i, text in enumerate(levels)]
    questions = [
        Question.noul(
            "on_time",
            "Was the completed submission received before coverage ended?",
            float(delta < 0),
        ),
        choice(
            rng, "expiry", "What is the expiry date in the issuer's clock?", dates, end.isoformat()
        ),
        Question(
            "lateness",
            "score",
            "How does submission time compare with the cutoff?",
            candidates,
            one_hot(candidates, f"s{bucket}"),
        ),
    ]
    return Sample(
        state,
        questions,
        {
            "generator": "calendar",
            "source_example_id": f"calendar-{idx}",
            "split_group": f"calendar:{start}:{months}:{issuer_offset}",
        },
    )


def probability_sample(rng: random.Random, idx: int) -> Sample:
    positive = idx % 2 == 0
    if rng.choice([False, True]):
        while True:
            population, defective, draws = rng.randint(10, 40), rng.randint(1, 8), rng.randint(1, 6)
            p = 1 - Fraction(math.comb(population - defective, draws), math.comb(population, draws))
            if (
                Fraction(1, 10) < p < Fraction(9, 10)
                and (p > Fraction(1, 2)) == positive
                and p != Fraction(1, 2)
            ):
                break
        state = (
            f"QUALITY INSPECTION LOT QC-{idx}\nUnits: {population}\nDefective units: {defective}\nRandom sample size: {draws}\n"
            "The count of defective units is exact. Defects cannot be identified visually. Draw uniformly without replacement. "
            "Testing detects every defective unit and never rejects a good unit. The lot is rejected if at least one sampled unit is defective. No unit has been sampled yet."
        )
        instruction = "Will this inspection reject the lot? Return probabilities reflecting the specified sampling process."
        group = f"probability:sample:{population}:{defective}:{draws}"
    else:
        while True:
            prevalence, sensitivity, false_alarm = (
                rng.randint(1, 80),
                rng.randint(70, 99),
                rng.randint(1, 40),
            )
            true_alarms = prevalence * sensitivity
            p = Fraction(true_alarms, true_alarms + (100 - prevalence) * false_alarm)
            if (p > Fraction(1, 2)) == positive and p != Fraction(1, 2):
                break
        state = (
            f"MONITORING REVIEW MR-{idx}\nIncident prevalence: {prevalence}%\nSensitivity: {sensitivity}%\nFalse-alarm rate: {false_alarm}%\n"
            "Prevalence is the probability that a monitored case has a real incident BEFORE observing an alarm. "
            "Sensitivity is P(alarm | incident); the false-alarm rate is P(alarm | no incident). These rates are exact and refer to the same population. "
            "The current case was selected uniformly from that population, its monitor raised an alarm, and no other evidence is available."
        )
        instruction = "Does the current alarm correspond to a real incident? Return the conditional probabilities given the alarm."
        group = f"probability:conditional:{prevalence}:{sensitivity}:{false_alarm}"
    return Sample(
        state,
        [Question.noul("event", instruction, float(p))],
        {
            "generator": "probability",
            "source_example_id": f"probability-{idx}",
            "split_group": group,
        },
    )


GENERATORS = {
    "long_rules": long_rules_sample,
    "calendar": calendar_sample,
    "probability": probability_sample,
}
