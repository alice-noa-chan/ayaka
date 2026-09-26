"""Reference labels recomputed from the rendered evidence, without generator oracles."""

import calendar
import datetime as dt
import re
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import pytest

from ayaka.data.loaders import load_spec_samples
from ayaka.data.synthetic import generate
from ayaka.prompt import canonical_order
from ayaka.training.run import DEFAULT_SPECS, split_eval


def gold(q):
    return next(c.description for c in q.candidates if q.target_distribution[c.id] == 1.0)


@pytest.mark.parametrize("kind", ["long_rules", "calendar", "probability"])
def test_hard_sources_are_deterministic_registered_and_grouped(kind):
    name = f"synth_{kind}"
    assert name in DEFAULT_SPECS
    samples, manifest = load_spec_samples(name, limit=40, seed=7)
    assert samples == generate(
        kind,
        40,
        7,
        {
            "language": "en",
            "source": name,
            "task_family": manifest.task_family,
            "evidence_state": "intact",
        },
    )
    assert manifest.revision == "v2"
    assert manifest.source_url.endswith("hard_synthetic.py")
    assert samples != load_spec_samples(name, limit=40, seed=8)[0]
    pools = {(manifest.task_family, "en"): samples[:]}
    held = split_eval(pools, 10, seed=0)
    assert held
    assert {s.metadata["split_group"] for s in held}.isdisjoint(
        s.metadata["split_group"] for cell in pools.values() for s in cell
    )


@pytest.mark.parametrize("seed", [0, 11])
def test_long_rules_recomputed_from_register_revision_fx_and_ledger(seed):
    tiers, windows, elevated_seen = Counter(), set(), set()
    for sample in generate("long_rules", 200, seed, {}):
        state = sample.state
        fields = dict(
            re.findall(
                r"^(Trading name|Purchase order|Invoice date|Currency|Amount): (.+)$", state, re.M
            )
        )
        current = dt.date.fromisoformat(fields["Invoice date"])
        vendors = re.findall(r"^Vendor (\S+) \| (.+) \| (\w+)$", state, re.M)
        vendor, _, country = next(v for v in vendors if v[1] == fields["Trading name"])
        risk = "STANDARD" if country in {"DE", "FR", "NL"} else "ELEVATED"
        elevated_seen.add(risk)
        revisions = [
            (dt.date.fromisoformat(day), int(days))
            for day, days in re.findall(r"Window version \d+: effective (\S+); days (\d+)", state)
        ]
        window = max((day, days) for day, days in revisions if day <= current)[1]
        windows.add(window)
        rates = {
            (day, currency): Decimal(rate)
            for day, currency, rate in re.findall(r"^FX (\S+) \| (\w+) \| ([\d.]+)$", state, re.M)
        }

        def converted(day, currency, amount, rates=rates):
            return (Decimal(amount) / rates[day, currency]).quantize(Decimal("0.01"), ROUND_HALF_UP)

        total = converted(fields["Invoice date"], fields["Currency"], fields["Amount"])
        for vid, order, day, currency, amount in re.findall(
            r"^Invoice \S+ \| (\S+) \| (\S+) \| (\S+) \| (\w+) \| ([\d.]+)$", state, re.M
        ):
            age = (current - dt.date.fromisoformat(day)).days
            if vid == vendor and order == fields["Purchase order"] and 0 < age <= window:
                total += converted(day, currency, amount)
        matrix = re.findall(
            r"Tier \d+ \(([^)]+)\): STANDARD <= (\d+) EUR; ELEVATED <= (\d+) EUR", state
        )
        tier = next(
            (
                name
                for name, standard, elevated in matrix
                if total <= Decimal(standard if risk == "STANDARD" else elevated)
            ),
            "executive committee",
        )
        tiers[tier] += 1
        q = {q.id: q for q in sample.questions}
        assert gold(q["aggregate"]) == f"{total:.2f} EUR"
        assert gold(q["routing"]) == tier
        assert q["senior"].target_distribution["true"] == float(
            tier in {"finance director", "executive committee"}
        )
    assert len(tiers) == 4
    assert windows == {14, 30}
    assert elevated_seen == {"STANDARD", "ELEVATED"}


def test_calendar_oracle_in_utc_and_rendered_answer_position_diversity():
    positions, boundaries, leap_days = Counter(), set(), 0
    for sample in generate("calendar", 1000, 2, {}):
        fields = dict(
            re.findall(
                r"^(Activation date|Term|Issuer fixed offset|Customer fixed offset|Submission in customer clock): (.+)$",
                sample.state,
                re.M,
            )
        )
        start = dt.date.fromisoformat(fields["Activation date"])
        year, month = start.year, start.month
        for _ in range(int(fields["Term"].split()[0])):
            month += 1
            if month == 13:
                year, month = year + 1, 1
        expiry = dt.date(year, month, min(start.day, calendar.monthrange(year, month)[1]))
        leap_days += expiry.month == 2 and expiry.day == 29
        issuer = int(fields["Issuer fixed offset"].removeprefix("UTC"))
        customer = int(fields["Customer fixed offset"].removeprefix("UTC"))
        cutoff_utc = dt.datetime.combine(expiry + dt.timedelta(days=1), dt.time()) - dt.timedelta(
            hours=issuer
        )
        arrival_utc = dt.datetime.fromisoformat(
            fields["Submission in customer clock"]
        ) - dt.timedelta(hours=customer)
        minutes = int((arrival_utc - cutoff_utc).total_seconds() / 60)
        boundaries.add(minutes)
        bucket = 0 if minutes < 0 else 1 if minutes <= 60 else 2 if minutes <= 1440 else 3
        q = {q.id: q for q in sample.questions}
        assert q["on_time"].target_distribution["true"] == float(arrival_utc < cutoff_utc)
        assert gold(q["expiry"]) == expiry.isoformat()
        assert q["lateness"].target_distribution[f"s{bucket}"] == 1.0
        descriptions = [c.description for c in q["expiry"].candidates]
        positions[canonical_order(descriptions).index(descriptions.index(expiry.isoformat()))] += 1
    assert leap_days > 0
    assert {0, 60, 61, 1440, 1441} <= boundaries
    assert set(positions) == {0, 1, 2, 3}


def test_probabilities_recomputed_with_sequential_draws_and_population_counts():
    variants, sides = Counter(), Counter()
    for sample in generate("probability", 400, 9, {}):
        state = sample.state
        if "Units:" in state:
            variants["sampling"] += 1
            n = int(re.search(r"Units: (\d+)", state)[1])
            k = int(re.search(r"Defective units: (\d+)", state)[1])
            draws = int(re.search(r"Random sample size: (\d+)", state)[1])
            all_good = Fraction(1)
            for j in range(draws):
                all_good *= Fraction(n - k - j, n - j)
            expected = 1 - all_good
        else:
            variants["conditional"] += 1
            prior = int(re.search(r"prevalence: (\d+)%", state)[1])
            tpr = int(re.search(r"Sensitivity: (\d+)%", state)[1])
            fpr = int(re.search(r"False-alarm rate: (\d+)%", state)[1])
            incident_alarms, normal_alarms = prior * tpr, (100 - prior) * fpr
            expected = Fraction(incident_alarms, incident_alarms + normal_alarms)
        target = sample.questions[0].target_distribution
        assert target["true"] == pytest.approx(float(expected), abs=1e-12)
        assert 0 < target["true"] < 1
        sides[target["true"] > 0.5] += 1
    assert sides == {True: 200, False: 200}
    assert set(variants) == {"sampling", "conditional"}
