"""Procedural temporal / numeric / policy data: labels recomputed independently."""

import datetime as dt
import re
from decimal import ROUND_HALF_UP, Decimal

import pytest
import torch

from ayaka.data.synthetic import evaluate_policy, generate
from ayaka.training.calibrate import apply_temperatures, fit_temperatures

MONTHS = "January February March April May June July August September October November December"


def parse_date(s: str) -> dt.date:
    s = s.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return dt.date.fromisoformat(s)
    names = MONTHS.split()
    if m := re.fullmatch(r"(\w+) (\d+), (\d{4})", s):
        return dt.date(int(m[3]), names.index(m[1]) + 1, int(m[2]))
    m = re.fullmatch(r"(\d+) (\w+) (\d{4})", s)
    return dt.date(int(m[3]), names.index(m[2]) + 1, int(m[1]))


def gold(q):
    return next(c.description for c in q.candidates if q.target_distribution[c.id] == 1.0)


def qmap(s):
    return {q.id: q for q in s.questions}


def is_yes(q):
    return q.target_distribution[q.candidates[-1].id] == 1.0


@pytest.mark.parametrize("kind", ["temporal", "numeric", "policy"])
def test_generation_is_deterministic_and_well_formed(kind):
    a, b = generate(kind, 50, 3, {"family": "x"}), generate(kind, 50, 3, {"family": "x"})
    assert [s.state for s in a] == [s.state for s in b]
    assert generate(kind, 5, 4, {})[0].state != a[0].state
    for s in a:
        assert s.metadata["family"] == "x" and s.metadata["generator"] == kind
        for q in s.questions:
            assert sum(q.target_distribution.values()) == pytest.approx(1.0)
            assert len({c.description for c in q.candidates}) == len(q.candidates)


def test_noul_labels_are_balanced():
    for kind in ("temporal", "numeric", "policy"):
        ys = [
            is_yes(q) for s in generate(kind, 400, 0, {}) for q in s.questions if q.type == "noul"
        ]
        assert 0.3 < sum(ys) / len(ys) < 0.7, kind


def test_temporal_labels_recomputed():
    for s in generate("temporal", 300, 1, {}):
        f = dict(line.split(": ", 1) for line in s.state.splitlines() if ": " in line)
        order, delivered, today = (parse_date(f[k]) for k in ("Order placed", "Delivered", "Today"))
        if m := re.fullmatch(r"(\d+) days after the order was placed", f["Shipped"]):
            shipped = order + dt.timedelta(days=int(m[1]))
        else:
            shipped = parse_date(f["Shipped"])
        promise = int(re.search(r"ship within (\d+) days", s.state)[1])
        window = int(re.search(r"within (\d+) days of delivery", s.state)[1])
        q = qmap(s)
        assert is_yes(q["shipped_on_time"]) == ((shipped - order).days <= promise)
        assert is_yes(q["return_open"]) == ((today - delivered).days <= window)
        assert int(gold(q["days_to_delivery"]).split()[0]) == (delivered - order).days
        assert gold(q["order_weekday"]) == order.strftime("%A")


def test_numeric_final_total_recomputed():
    for s in generate("numeric", 300, 2, {}):
        items = re.findall(r"- .+?: (\d+) x ([\d,]+\.\d\d)", s.state)
        sub = sum(int(n) * Decimal(p.replace(",", "")) for n, p in items)
        disc = int(re.search(r"Discount: (\d+)%", s.state)[1])
        tax = int(re.search(r"Sales tax: (\d+)%", s.state)[1])
        ship = Decimal(re.search(r"Shipping: ([\d,.]+) ", s.state)[1].replace(",", ""))
        budget = Decimal(re.search(r"Approved budget: ([\d,.]+)", s.state)[1].replace(",", ""))
        total = (sub * (100 - disc) / 100 * (100 + tax) / 100 + ship).quantize(
            Decimal("0.01"), ROUND_HALF_UP
        )
        q = qmap(s)
        assert is_yes(q["over_budget"]) == (total > budget)
        if "final_total" in q:
            assert gold(q["final_total"]) == f"{total:,.2f}"


RULES = {
    "limit": 500, "days": 30, "tenure": 6, "approval_over": 150, "exempt_under": 40,
    "banned": ["alcohol", "fines"], "meal_cap": 60,
}  # fmt: skip
OK = {
    "amount": 100, "receipt": True, "days_ago": 10, "category": "software", "tenure": 12,
    "approved": None, "role": "staff", "trip_id": None, "attendees": None,
}  # fmt: skip


@pytest.mark.parametrize(
    "change,fails",
    [
        ({}, []),
        ({"amount": 501, "approved": True}, ["limit"]),
        ({"amount": None}, ["limit"]),  # not established
        ({"receipt": None}, ["receipt"]),
        ({"role": "executive", "amount": 39, "receipt": False}, []),  # exemption
        ({"role": "executive", "amount": 40, "receipt": False}, ["receipt"]),
        ({"days_ago": 31}, ["age"]),
        ({"days_ago": 30}, []),
        ({"category": "fines"}, ["category"]),
        ({"tenure": 5}, ["tenure"]),
        ({"amount": 151}, ["approval"]),
        ({"amount": 151, "approved": False, "days_ago": 99}, ["age", "approval"]),
        ({"category": "business travel"}, ["travel"]),
        ({"category": "business travel", "trip_id": True}, []),
        ({"category": "client meals", "attendees": 2, "amount": 120}, []),
        ({"category": "client meals", "attendees": 2, "amount": 121, "approved": True}, ["meals"]),
        ({"category": "client meals"}, ["meals"]),  # attendee count missing
    ],
)
def test_evaluate_policy(change, fails):
    assert evaluate_policy(RULES, {**OK, **change}) == fails


def test_policy_questions_are_consistent():
    for s in generate("policy", 300, 5, {}):
        q = qmap(s)
        first = gold(q["first_failure"])
        assert is_yes(q["permitted"]) == first.startswith("None")
        for k, v in q.items():
            if k.startswith("meets_") and not is_yes(v):
                # a failed clause cannot come before the first failure
                assert not first.startswith("None")
                sec = int(re.search(r"Section (\d+)", v.instruction)[1])
                assert int(re.search(r"Section (\d+)", first)[1]) <= sec


def test_length_bucketed_temperatures_fit_and_apply():
    logits, targets, types, lengths = [], [], [], []
    for i in range(400):
        long = i % 2 == 1
        g = i % 3
        x = [0.0, 0.0, 0.0]
        # short prompts: always right; long prompts: confidently wrong half the time
        x[g if (not long or i % 4 == 1) else (g + 1) % 3] = 4.0
        logits.append(x)
        targets.append([float(j == g) for j in range(3)])
        types.append("choice")
        lengths.append(2000 if long else 100)
    temps = fit_temperatures(logits, targets, types, lengths, long_threshold=1024)
    assert set(temps) == {"choice", "choice@short", "choice@long"}
    assert temps["choice@long"] > temps["choice"] > temps["choice@short"]

    class Stub:
        temperature = torch.ones(3, 2)

    apply_temperatures(Stub, {"choice": 2.0, "choice@long": 5.0, "noul": 1.5})
    assert Stub.temperature.tolist() == [[1.5, 1.5], [2.0, 5.0], [1.0, 1.0]]
