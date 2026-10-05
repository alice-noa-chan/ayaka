"""Offline edits and audits for the existing, independently parsed authored grammar.

No natural human row, freeform paraphrase, model, teacher or held-out discovery is
supported. Weights in the receipt describe component-balanced sampling; this
module does not apply them to a trainer or establish a quality improvement.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ..eval.read_artifact import fingerprint
from .direct_verification import WRAPPERS, verify_authored_gold
from .schema import Sample
from .source_groups import connected_groups, identities

VERSION = "ayaka-authored-counterfactual-1"
OPERATIONS = {
    "year": "leap",
    "event_date": "rule_revision",
    "override": "exception",
    "credential": "exception",
    "amount": "rounding",
    "timezone_offset_minutes": "timezone",
    "weekday_end": "business",
    "bag_counts": "probability",
    "completed_checks": "rubric",
    "candidate_order": None,
}
RELATIONS = {"flip", "preserve"}


def _identity(sample):
    value = sample.metadata.get("source_example_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("counterfactual rows require a nonempty source example identity")
    return value


def _gold(sample):
    if not isinstance(sample, Sample) or not isinstance(sample.metadata, dict):
        raise ValueError("an authored Sample and provenance object are required")
    Sample.from_json(sample.to_json())  # revalidate candidate/target invariants after mutations
    if len(sample.questions) != 1:
        raise ValueError("counterfactual rows require exactly one authored question")
    question = sample.questions[0]
    gold = verify_authored_gold(sample, question)
    stored = {c.id: question.target_distribution.get(c.id, 0.0) for c in question.candidates}
    if stored != gold:
        raise ValueError("stored authored gold disagrees with independent evidence parsing")
    return gold


def _iso_date(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError("date edits require an exact ISO date")
    date.fromisoformat(value)
    return value


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("integer edit is outside its declared bounds")
    return value


def _replace(facts, pattern, value):
    result, count = re.subn(pattern, lambda _: str(value), facts)
    if count != 1:
        raise ValueError("edit does not match exactly one declared fact field")
    return result


def _edit_facts(facts, operation, value):
    if operation == "year":
        return _replace(facts, r"(?<=Year )[0-9]{4}(?=\.)", f"{_integer(value, 1, 9999):04}")
    if operation == "event_date":
        return _replace(facts, r"(?<=The event is on )[0-9-]+(?=\.)", _iso_date(value))
    if operation == "override":
        if type(value) is not bool:
            raise ValueError("override edit must be boolean")
        return _replace(facts, r"(?<=override present: )(?:True|False)(?=;)", value)
    if operation == "credential":
        return _replace(
            facts, r"(?<=credential level: )[0-9]+(?=\.)", _integer(value, 0, 1_000_000)
        )
    if operation == "amount":
        if (
            not isinstance(value, str)
            or len(value) > 100
            or not re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", value)
        ):
            raise ValueError("amount edit must be bounded exact decimal text")
        return _replace(facts, r"(?<=Amount )-?[0-9]+(?:\.[0-9]+)?(?=\. Round)", value)
    if operation == "timezone_offset_minutes":
        minutes = _integer(value, -1439, 1439)
        match = re.fullmatch(
            r"Timestamp (\S+)\. Convert to UTC and report the day of month\.", facts
        )
        local = datetime.fromisoformat(match[1])
        edited = local.replace(tzinfo=timezone(timedelta(minutes=minutes)))
        return facts.replace(match[1], edited.isoformat(), 1)
    if operation == "weekday_end":
        return _replace(facts, r"(?<=through )[0-9-]+(?=, inclusive)", _iso_date(value))
    if operation == "bag_counts":
        if not isinstance(value, dict) or set(value) != {"red", "blue"}:
            raise ValueError("bag edit requires exactly red and blue counts")
        red = _integer(value["red"], 0, 1_000_000)
        blue = _integer(value["blue"], 0, 1_000_000)
        return _replace(
            facts, r"(?<=holds )[0-9]+ red and [0-9]+ blue(?= balls)", f"{red} red and {blue} blue"
        )
    if operation == "completed_checks":
        if (
            not isinstance(value, list)
            or len(value) > 64
            or any(not isinstance(v, str) or not v or len(v) > 100 for v in value)
            or len(set(value)) != len(value)
        ):
            raise ValueError("completed checks require bounded unique text")
        return _replace(facts, r"(?<=Completed checks: )\[.*\](?=\. Report)", repr(value))
    raise ValueError("unsupported authored fact edit")


def make_counterfactual(original, operation, value, *, relation):
    """Keep criteria fixed, recompute gold, and reject an incorrect flip declaration.

    `preserve` means the question target is preserved; it is not a claim that
    every fact edit preserves the document's meaning. Candidate order edits do
    preserve the complete per-candidate meanings. Arbitrary text is unsupported.
    """
    if (
        not isinstance(operation, str)
        or operation not in OPERATIONS
        or not isinstance(relation, str)
        or relation not in RELATIONS
    ):
        raise ValueError("unsupported operation or target relation")
    original_gold = _gold(original)
    if "counterfactual" in original.metadata:
        raise ValueError("edit the original root, not an already augmented row")
    parent_id = _identity(original)
    family = original.metadata["task_family"]
    if OPERATIONS[operation] is not None and OPERATIONS[operation] != family:
        raise ValueError("operation belongs to a different authored task family")
    edited = copy.deepcopy(original)
    question = edited.questions[0]
    wrapper = re.fullmatch(WRAPPERS[original.metadata["split"]], original.state)
    facts = wrapper[1]
    if operation == "candidate_order":
        candidates = {c.id: c for c in question.candidates}
        if (
            not isinstance(value, list)
            or any(not isinstance(v, str) for v in value)
            or len(value) != len(candidates)
            or set(value) != set(candidates)
        ):
            raise ValueError("candidate order must be an exact permutation")
        question.candidates = [candidates[v] for v in value]
    else:
        facts = _edit_facts(facts, operation, value)
        edited.state = original.state[: wrapper.start(1)] + facts + original.state[wrapper.end(1) :]
        edited.metadata["case_facts_sha256"] = hashlib.sha256(facts.encode()).hexdigest()
    if edited.state == original.state and question.candidates == original.questions[0].candidates:
        raise ValueError("counterfactual operation must actually change the input")
    fresh = verify_authored_gold(edited, question)
    changed = fresh != original_gold  # candidate ID -> probability, independent of display order
    if changed != (relation == "flip"):
        raise ValueError("declared target relation disagrees with independently recomputed gold")
    question.target_distribution = fresh
    recipe = {
        "version": VERSION,
        "parent_id": parent_id,
        "parent_sha256": fingerprint(original.to_json()),
        "operation": operation,
        "value": copy.deepcopy(value),
        "relation": relation,
    }
    edited.metadata["source_example_id"] = f"{parent_id}/cf/{fingerprint(recipe)}"
    edited.metadata["derived_from"] = sorted(identities(original))
    edited.metadata["counterfactual"] = recipe
    for key in (
        "verified_traces",
        "trace_validator",
        "teacher",
        "teacher_probs",
        "proposal_supervision",
    ):
        edited.metadata.pop(key, None)  # changed facts cannot retain a parent trace/teacher
    return edited


def audit_counterfactuals(samples, *, required_operations=None, max_component_rows=None):
    """Audit the complete closure before any quota/split or training publication."""
    samples = list(samples)
    if not samples:
        raise ValueError("an empty counterfactual cohort cannot pass")
    required = {} if required_operations is None else required_operations
    if not isinstance(required, dict) or any(
        op not in OPERATIONS or type(count) is not int or count < 1
        for op, count in required.items()
    ):
        raise ValueError("operation minima must be positive counts of supported edits")
    if max_component_rows is not None:
        _integer(max_component_rows, 1, 1_000_000)
    rows = {}
    for sample in samples:
        _gold(sample)
        identity = _identity(sample)
        if identity in rows:
            raise ValueError("counterfactual source identities must be unique")
        rows[identity] = sample
    counts, cells, roots = Counter(), Counter(), Counter()
    for sample in rows.values():
        recipe = sample.metadata.get("counterfactual")
        if "counterfactual" not in sample.metadata:
            continue
        if (
            not isinstance(recipe, dict)
            or set(recipe)
            != {"version", "parent_id", "parent_sha256", "operation", "value", "relation"}
            or recipe["version"] != VERSION
        ):
            raise ValueError("unsupported counterfactual provenance")
        if (
            not isinstance(recipe["parent_id"], str)
            or not isinstance(recipe["parent_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", recipe["parent_sha256"])
        ):
            raise ValueError("counterfactual root identity and digest are invalid")
        parent = rows.get(recipe["parent_id"])
        if parent is None or "counterfactual" in parent.metadata:
            raise ValueError("every derived row requires its original root in the audit")
        if parent.metadata["split"] != sample.metadata["split"]:
            raise ValueError("counterfactual parent and child cannot cross split roles")
        expected = make_counterfactual(
            parent, recipe["operation"], recipe["value"], relation=recipe["relation"]
        )
        if fingerprint(expected.to_json()) != fingerprint(sample.to_json()):
            raise ValueError("counterfactual payload differs from its bound controlled edit")
        op, relation, kind = recipe["operation"], recipe["relation"], sample.questions[0].type
        counts[op] += 1
        roots[recipe["parent_id"]] += 1
        gold = _gold(sample)
        label = next(c for c, mass in gold.items() if mass == 1)
        cells[(op, relation, kind, label)] += 1
    if not counts:
        raise ValueError("the audit requires at least one derived counterfactual")
    if any(counts[op] < minimum for op, minimum in required.items()):
        raise ValueError("counterfactual cohort falls short of declared operation minima")
    # The verified facts, unlike their split-specific wrappers, identify the
    # same underlying case. Close this alias on copies, retaining every original
    # lineage bridge. Do not modify shared grouping semantics or source rows.
    grouping_rows = [
        replace(
            sample,
            metadata={
                **sample.metadata,
                "lineage_ids": sorted(
                    identities(sample) | {"authored-facts/" + sample.metadata["case_facts_sha256"]}
                ),
            },
        )
        for sample in samples
    ]
    memberships, groups = connected_groups(grouping_rows)
    components = defaultdict(list)
    for index, group in enumerate(memberships):
        components[group].append(index)
    weights = {}
    for indices in components.values():
        if len({samples[i].metadata["split"] for i in indices}) != 1:
            raise ValueError("connected counterfactual aliases leak across split roles")
        if max_component_rows is not None and len(indices) > max_component_rows:
            raise ValueError("counterfactual component exceeds its declared row cap")
        for index in indices:
            weights[_identity(samples[index])] = 1 / len(indices)
    return {
        "version": VERSION,
        "scope": "offline authored grammar audit; no model quality or launch attestation",
        "samples": len(samples),
        "derived_samples": sum(counts.values()),
        "components": len(groups),
        "max_component_rows": max(map(len, components.values())),
        "operation_counts": dict(sorted(counts.items())),
        "derived_per_root": dict(sorted(roots.items())),
        "label_cells": [
            {"operation": op, "relation": relation, "type": kind, "label": label, "rows": n}
            for (op, relation, kind, label), n in sorted(cells.items())
        ],
        "suggested_component_weights": dict(sorted(weights.items())),
        "weights_applied": False,
        "context_ready": False,
        "trace_supervision_ready": False,
        "dataset_sha256": fingerprint([rows[key].to_json() for key in sorted(rows)]),
        "promotable": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-component-rows", type=int)
    parser.add_argument("--require-operation", action="append", default=[], metavar="OP=COUNT")
    args = parser.parse_args(argv)
    raw = args.input.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != args.expected_sha256:
        raise ValueError("counterfactual input differs from its external byte anchor")
    samples = [Sample.from_json(json.loads(line)) for line in raw.splitlines() if line.strip()]
    minima = {}
    for entry in args.require_operation:
        op, separator, count = entry.partition("=")
        if not separator or op in minima or not count.isdecimal():
            raise ValueError("operation minima require unique OP=COUNT entries")
        minima[op] = int(count)
    report = audit_counterfactuals(
        samples, required_operations=minima, max_component_rows=args.max_component_rows
    )
    report["input_file_sha256"] = digest
    payload = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    with args.output.open("xb") as output:
        output.write(payload)
    return report


if __name__ == "__main__":
    main()
