"""Procedurally generated decisions with exactly computed labels.

JevBench's weakest families for the trained Small model were
temporal/numeric reasoning and long policies. Openly licensed human data
for these is scarce, so they are generated here: the generator is part of
this repository, and every label is computed, never guessed.

Three families, each emitting one state with several typed questions:

- ``temporal``: an order record with mixed date formats and relative dates.
  Asks about shipping promises, return windows (day counting), elapsed
  days, weekdays, lateness buckets and event order.
- ``numeric``: an invoice with discount, tax, untaxed shipping and a
  budget. Asks about budget overrun, the largest line, the final total,
  the budget gap bucket and the average unit price.
- ``policy``: a long reimbursement policy (requirements, an exception,
  plus unrelated clauses) and one request. Asks whether the request is
  permitted, which clause it fails first, and whether single requirements
  are met. Facts the record does not state count as not established.
"""

from __future__ import annotations

import datetime as dt
import random
from decimal import ROUND_HALF_UP, Decimal

from .schema import Candidate, Question, Sample, one_hot

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip


def fmt_date(d: dt.date, style: int) -> str:
    if style == 0:
        return d.isoformat()
    if style == 1:
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def _choice(qid: str, instruction: str, options: list[str], gold: int) -> Question:
    cands = [Candidate(f"o{i}", o) for i, o in enumerate(options)]
    return Question(qid, "choice", instruction, cands, one_hot(cands, f"o{gold}"))


def _score(qid: str, instruction: str, levels: list[str], gold: int) -> Question:
    cands = [Candidate(f"s{i}", lv, ordinal=i) for i, lv in enumerate(levels)]
    return Question(qid, "score", instruction, cands, one_hot(cands, f"s{gold}"))


def _near_distinct(rng: random.Random, value: int, k: int, lo: int = 0) -> list[int]:
    """value plus k-1 distinct nearby integers >= lo."""
    out = {value}
    while len(out) < k:
        cand = value + rng.choice([-3, -2, -1, 1, 2, 3, 7, -7])
        if cand >= lo:
            out.add(cand)
    return sorted(out)


# ---------------------------------------------------------------- temporal


def temporal_sample(rng: random.Random, idx: int) -> Sample:
    order = dt.date(2023, 1, 1) + dt.timedelta(days=rng.randrange(0, 5 * 365))
    ship_days = rng.randint(0, 12)
    transit = rng.randint(1, 9)
    shipped = order + dt.timedelta(days=ship_days)
    delivered = shipped + dt.timedelta(days=transit)
    today = delivered + dt.timedelta(days=rng.randint(0, 45))
    ticket = order + dt.timedelta(days=rng.randint(1, (today - order).days or 1))
    promise = rng.choice([2, 3, 5, 7])
    window = rng.choice([14, 21, 30])
    style = lambda: rng.randrange(3)  # noqa: E731  (formats vary per line)

    lines = [
        f"Order placed: {fmt_date(order, style())}",
        (
            f"Shipped: {ship_days} days after the order was placed"
            if rng.random() < 0.35
            else f"Shipped: {fmt_date(shipped, style())}"
        ),
        f"Delivered: {fmt_date(delivered, style())}",
        f"Support ticket opened: {fmt_date(ticket, style())}",
    ]
    rng.shuffle(lines)
    state = "\n".join(
        [f"Order record #{10000 + idx}", *lines, f"Today: {fmt_date(today, style())}", ""]
        + [
            f"Shipping promise: orders ship within {promise} days of the order date.",
            f"Returns: accepted if requested within {window} days of delivery "
            "(the delivery day is day 0).",
        ]
    )

    qs = [
        Question.noul("shipped_on_time", f"Was the order shipped within the promised {promise} days?", float(ship_days <= promise)),
        Question.noul("return_open", "Could the customer still request a return today?", float((today - delivered).days <= window)),
    ]  # fmt: skip
    elapsed = (delivered - order).days
    opts = _near_distinct(rng, elapsed, 4, lo=0)
    qs.append(
        _choice(
            "days_to_delivery",
            "How many days passed between placing the order and delivery?",
            [f"{o} day" if o == 1 else f"{o} days" for o in opts],
            opts.index(elapsed),
        )
    )
    qs.append(
        _choice(
            "order_weekday", "On which weekday was the order placed?", WEEKDAYS, order.weekday()
        )
    )
    late = max(0, ship_days - promise)
    bucket = 0 if late == 0 else 1 if late <= 2 else 2 if late <= 6 else 3
    qs.append(
        _score(
            "shipping_delay",
            "How late was the shipment relative to the promise?",
            ["on time or early", "1-2 days late", "3-6 days late", "7 or more days late"],
            bucket,
        )
    )
    events = {"the support ticket was opened": ticket, "the order was delivered": delivered}
    if ticket != delivered:
        names = list(events)
        qs.append(
            _choice(
                "first_event",
                "Which happened first?",
                names,
                names.index(min(events, key=events.get)),
            )
        )
    return Sample(
        state=state,
        questions=qs,
        metadata={"generator": "temporal", "source_example_id": f"temporal-{idx}"},
    )


# ----------------------------------------------------------------- numeric

ITEMS = [
    "printer paper", "toner cartridge", "desk lamp", "USB hub", "monitor arm", "office chair",
    "whiteboard", "label maker", "headset", "keyboard", "webcam", "filing cabinet",
    "extension cord", "stapler", "shipping boxes", "packing tape", "notebooks", "router",
]  # fmt: skip
CENT = Decimal("0.01")


def _money(x: Decimal) -> str:
    return f"{x.quantize(CENT, ROUND_HALF_UP):,.2f}"


def numeric_sample(rng: random.Random, idx: int) -> Sample:
    while True:
        names = rng.sample(ITEMS, rng.randint(3, 7))
        lines = [(n, rng.randint(1, 40), Decimal(rng.randint(50, 50000)) / 100) for n in names]
        totals = [q * p for _, q, p in lines]
        if totals.count(max(totals)) == 1:
            break
    discount = rng.choice([0, 5, 10, 15, 20])
    tax = rng.choice([0, 5, 8, 10, 20])
    shipping = Decimal(rng.randint(0, 6000)) / 100
    subtotal = sum(totals)
    after_discount = subtotal * (100 - discount) / 100
    total = (after_discount * (100 + tax) / 100 + shipping).quantize(CENT, ROUND_HALF_UP)
    budget = (total * Decimal(rng.choice([0.8, 0.9, 0.95, 1.05, 1.1, 1.25]))).quantize(Decimal(1))
    if budget == total:
        budget += 1

    rows = "\n".join(f"- {n}: {q} x {_money(p)}" for n, q, p in lines)
    state = (
        f"Purchase request PR-{20000 + idx}\nLine items (quantity x unit price):\n{rows}\n"
        f"Discount: {discount}% on the item subtotal\nSales tax: {tax}% applied after the discount\n"
        f"Shipping: {_money(shipping)} (not taxed)\nApproved budget: {_money(budget)}"
    )
    qs = [
        Question.noul(
            "over_budget",
            "Does the final amount due exceed the approved budget?",
            float(total > budget),
        )
    ]
    qs.append(
        _choice(
            "largest_line",
            "Which line item has the largest line total?",
            names,
            totals.index(max(totals)),
        )
    )
    wrong = {
        _money(subtotal + shipping),  # discount and tax forgotten
        _money(subtotal * (100 + tax) / 100 + shipping),  # discount forgotten
        _money((after_discount + shipping) * (100 + tax) / 100),  # shipping taxed
        _money(after_discount * (100 + tax) / 100),  # shipping forgotten
    }
    options = [_money(total)] + sorted(wrong - {_money(total)})[:3]
    if len(options) >= 2:
        qs.append(_choice("final_total", "What is the final amount due?", options, 0))
    gap = (total - budget) / budget
    bucket = 0 if gap < Decimal("-0.1") else 1 if gap <= 0 else 2 if gap <= Decimal("0.1") else 3
    qs.append(
        _score(
            "budget_gap",
            "How does the final amount compare with the budget?",
            ["more than 10% under", "at most 10% under", "at most 10% over", "more than 10% over"],
            bucket,
        )
    )
    avg = sum(p for _, _, p in lines) / len(lines)
    threshold = (avg * Decimal(rng.choice([0.85, 0.95, 1.05, 1.15]))).quantize(CENT)
    if threshold != avg.quantize(CENT):
        qs.append(
            Question.noul(
                "avg_price",
                f"Is the average unit price across the line items above {_money(threshold)}?",
                float(avg > threshold),
            )
        )
    return Sample(
        state=state,
        questions=qs,
        metadata={"generator": "numeric", "source_example_id": f"numeric-{idx}"},
    )


# ------------------------------------------------------------------ policy

FILLER = [
    "Reimbursements are paid with the next regular payroll run.",
    "Questions about this policy go to the finance helpdesk during office hours.",
    "Receipts are retained for seven years in line with record-keeping rules.",
    "This policy does not cover payroll advances or salary corrections.",
    "Mileage is reimbursed under the separate travel policy, not under this one.",
    "Managers review submitted claims in the order they are received.",
    "Claims may be submitted in any currency; finance converts them on payment.",
    "Corporate card statements are reconciled monthly by the finance team.",
    "Gifts to clients are governed by the gifts and hospitality policy.",
    "Employees may withdraw a pending claim at any time before approval.",
    "The finance team may request additional documentation for audits.",
    "Training courses are reimbursed only through the learning budget process.",
    "Claims are processed in the order they arrive; processing usually takes five to ten "
    "business days, and longer at the end of the fiscal quarter.",
    "An employee who leaves the company may still submit claims for purchases made while "
    "employed, subject to every other section of this policy.",
    "Finance may reject claims that are duplicated, altered or otherwise inconsistent with "
    "the attached documentation, and may refer such cases to internal audit.",
    "Where a purchase was split across several claims to stay under a limit, the claims are "
    "assessed together as if they had been submitted as one.",
    "Personal loyalty points earned on business purchases may be kept by the employee.",
    "Tips are reimbursable only as part of a meal or taxi receipt and never on their own.",
    "Team events are budgeted centrally and are not claimed through this process.",
    "Home office equipment follows the separate remote work allowance.",
    "Claims in a foreign currency should state the original amount; finance applies the "
    "exchange rate published on the payment date.",
    "This policy is reviewed annually by the finance committee; the current version applies "
    "to all claims submitted after its publication date.",
    "Employees on unpaid leave may submit claims only after returning to work.",
    "Contractors are reimbursed under the terms of their individual contracts instead.",
]


def _policy_rules(rng: random.Random) -> dict:
    return {
        "limit": rng.choice([200, 300, 500, 750, 1000]),
        "days": rng.choice([30, 45, 60, 90]),
        "tenure": rng.choice([3, 6]),
        "approval_over": rng.choice([100, 150, 250]),
        "exempt_under": rng.choice([25, 40, 50]),
        "banned": rng.sample(
            ["alcohol", "fines", "personal travel", "entertainment", "gift cards"], 2
        ),
        "meal_cap": rng.choice([40, 60, 80]),
    }


def evaluate_policy(r: dict, f: dict) -> list[str]:
    """Clause keys the request fails, in document order. Missing facts
    (None) do not establish a requirement."""
    fails = []
    if f["amount"] is None or f["amount"] > r["limit"]:
        fails.append("limit")
    receipt_needed = not (
        f["role"] == "executive" and f["amount"] is not None and f["amount"] < r["exempt_under"]
    )
    if receipt_needed and f["receipt"] is not True:
        fails.append("receipt")
    if f["days_ago"] is None or f["days_ago"] > r["days"]:
        fails.append("age")
    if f["category"] in r["banned"]:
        fails.append("category")
    if f["tenure"] is None or f["tenure"] < r["tenure"]:
        fails.append("tenure")
    if f["amount"] is not None and f["amount"] > r["approval_over"] and f["approved"] is not True:
        fails.append("approval")
    # conditional clauses: they bind only their own category
    if f["category"] == "business travel" and f["trip_id"] is not True:
        fails.append("travel")
    if f["category"] == "client meals" and (
        f["attendees"] is None
        or f["amount"] is None
        or f["amount"] > r["meal_cap"] * f["attendees"]
    ):
        fails.append("meals")
    return fails


CLAUSE_TITLES = {
    "limit": "claim amount limit",
    "receipt": "receipt requirement",
    "age": "submission deadline",
    "category": "excluded categories",
    "tenure": "minimum employment period",
    "approval": "manager approval",
    "travel": "trip number requirement",
    "meals": "client meal cap",
}


def policy_sample(rng: random.Random, idx: int) -> Sample:
    r = _policy_rules(rng)
    clauses = {
        "limit": f"A single claim may not exceed {r['limit']} dollars.",
        "receipt": "Every claim needs an itemized receipt. Exception: executives claiming "
        f"less than {r['exempt_under']} dollars do not need a receipt.",
        "age": f"Claims must be submitted within {r['days']} days of the purchase.",
        "category": f"The following are never reimbursed: {', '.join(r['banned'])}.",
        "tenure": f"Only employees with at least {r['tenure']} months of employment may claim.",
        "approval": f"Claims above {r['approval_over']} dollars also need written manager approval.",
        "travel": "Business travel claims must quote the pre-approved trip number; "
        "claims in other categories do not need one.",
        "meals": f"Client meal claims may not exceed {r['meal_cap']} dollars per attendee "
        "and must state the number of attendees.",
    }
    keys = list(clauses)
    body = [clauses[k] for k in keys] + rng.sample(FILLER, rng.randint(5, len(FILLER)))
    order = list(range(len(body)))
    rng.shuffle(order)
    numbered, section_of = [], {}
    for n, i in enumerate(order, start=1):
        numbered.append(f"Section {n}. {body[i]}")
        if i < len(keys):
            section_of[keys[i]] = n

    maybe = lambda value, p=0.06: None if rng.random() < p else value  # noqa: E731
    facts = {
        "amount": maybe(
            rng.choice([rng.randint(5, 60), rng.randint(60, 400), rng.randint(60, 1200)])
        ),
        "receipt": maybe(rng.random() < 0.85),
        "days_ago": maybe(rng.choice([rng.randint(1, 30), rng.randint(1, 120)])),
        "category": rng.choice(
            ["office supplies", "software", "books"]
            + ["client meals", "business travel"] * 2
            + r["banned"]
        ),
        "tenure": maybe(rng.randint(2, 60)),
        "approved": maybe(rng.random() < 0.8),
        "role": rng.choice(["staff", "staff", "manager", "executive"]),
        "trip_id": maybe(rng.random() < 0.8),
        "attendees": maybe(rng.randint(1, 8)),
    }
    record = [f"Employee role: {facts['role']}", f"Category: {facts['category']}"]
    if facts["amount"] is not None:
        record.append(f"Amount claimed: {facts['amount']} dollars")
    if facts["receipt"] is not None:
        record.append("Itemized receipt attached: " + ("yes" if facts["receipt"] else "no"))
    if facts["days_ago"] is not None:
        record.append(f"Purchase made {facts['days_ago']} days before submission")
    if facts["tenure"] is not None:
        record.append(f"Months employed: {facts['tenure']}")
    if facts["approved"] is not None:
        record.append("Manager approval on file: " + ("yes" if facts["approved"] else "no"))
    if facts["trip_id"] is not None:
        record.append("Pre-approved trip number quoted: " + ("yes" if facts["trip_id"] else "no"))
    if facts["attendees"] is not None:
        record.append(f"Attendees: {facts['attendees']}")
    rng.shuffle(record)
    state = (
        "EXPENSE REIMBURSEMENT POLICY\n" + "\n".join(numbered)
        + "\nAnything a claim record does not state is treated as not established.\n\n"
        f"CLAIM #{30000 + idx}\n" + "\n".join(record)
    )  # fmt: skip

    fails = evaluate_policy(r, facts)
    first = min(fails, key=lambda k: section_of[k]) if fails else None
    qs = [
        Question.noul(
            "permitted", "Under the policy, can this claim be reimbursed?", float(not fails)
        )
    ]
    by_section = sorted(keys, key=lambda k: section_of[k])
    options = [f"Section {section_of[k]} ({CLAUSE_TITLES[k]})" for k in by_section] + [
        "None: the claim can be reimbursed"
    ]
    gold = by_section.index(first) if first else len(options) - 1
    qs.append(
        _choice(
            "first_failure",
            "Which requirement does the claim fail first, in section order?",
            options,
            gold,
        )
    )
    for k in rng.sample(keys, 2):
        qs.append(
            Question.noul(
                f"meets_{k}",
                f"Does the claim satisfy the {CLAUSE_TITLES[k]} (Section {section_of[k]})?",
                float(k not in fails),
            )
        )
    return Sample(
        state=state,
        questions=qs,
        metadata={"generator": "policy", "source_example_id": f"policy-{idx}"},
    )


GENERATORS = {"temporal": temporal_sample, "numeric": numeric_sample, "policy": policy_sample}


def generate(kind: str, n: int, seed: int, metadata: dict) -> list[Sample]:
    rng = random.Random(f"{kind}:{seed}")
    out = []
    for i in range(n):
        s = GENERATORS[kind](rng, i)
        s.metadata = {**metadata, **s.metadata}
        out.append(s)
    return out
