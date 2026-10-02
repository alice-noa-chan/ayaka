"""Visible-only shortcut baselines and targeted rejection of known data defects."""

from collections import Counter, defaultdict
from datetime import datetime

from ..eval.v2 import summarize, typed_row
from ..primitives import QuestionSpec


def shortcut_audit(samples, *, reject=True):
    groups, losses = defaultdict(list), defaultdict(list)
    seen = set()
    for sample in samples:
        lineage = sample.metadata["source_lineage"]
        if lineage in seen:
            continue  # translations do not triple the apparent sample size
        seen.add(lineage)
        family = sample.metadata["task_family"]
        for question in sample.questions:
            target = [question.target_distribution.get(c.id, 0) for c in question.candidates]
            spec = QuestionSpec(
                question.type,
                question.instruction,
                [c.description for c in question.candidates],
                [c.ordinal for c in question.candidates] if question.type == "score" else None,
            )
            row = typed_row(spec, [1 / len(target)] * len(target), target)
            row.update(type=question.type, latency_s=0, reasoning_tokens=0)
            losses[question.type].append(row)
            if question.type == "choice" and max(target) == 1:
                numbers = [int(c.id) for c in question.candidates]
                gold = numbers[target.index(1)]
                ordered = sorted(numbers)
                groups[f"rank/{family}"].append((ordered.index(gold), len(ordered)))
                groups[f"position/{family}"].append((target.index(1), len(target)))
            elif question.type == "noul" and max(target) == 1:
                truth = question.target_distribution.get("true", 0) == 1
                facts = sample.metadata["oracle_facts"]
                year = (
                    datetime.fromisoformat(facts["local"]).year
                    if "local" in facts
                    else facts["year"]
                )
                groups[f"year_parity/{family}"].append(int(truth == bool(year % 2)))
    report, failures = {}, []
    for key, values in groups.items():
        if key.startswith("year_parity"):
            accuracy = max(sum(values), len(values) - sum(values)) / len(values)
            report[key] = {"cases": len(values), "best_parity_accuracy": accuracy}
            if len(values) >= 50 and accuracy > 0.70:
                failures.append(key)
        else:
            counts = Counter(position for position, _ in values)
            hits = sum(position == size // 2 for position, size in values)
            report[key] = {
                "cases": len(values),
                "histogram": dict(counts),
                "median_accuracy": hits / len(values),
            }
            # Targeted regression: five-option invoice gold must not be the median.
            if key == "rank/numeric" and len(values) >= 50 and hits / len(values) > 0.40:
                failures.append(key)
    result = {
        "passed": not failures,
        "failures": failures,
        "baselines": report,
        "uniform": {kind: summarize(rows) for kind, rows in losses.items()},
        "scope": "Targeted median/year shortcut checks and visible-only priors; not a proof that all shortcuts are absent.",
    }
    if reject and failures:
        raise ValueError(f"known recovery shortcuts detected: {failures}")
    return result
