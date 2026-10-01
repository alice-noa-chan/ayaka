"""Exact finite-domain partition labels, distinct from free-form semantic claims."""

import hashlib
import json
from itertools import combinations

from .reasoning_v2 import SPLITS
from .schema import Candidate, Question, Sample


def finite_partition_audit(universe, memberships, parent=None):
    if not isinstance(universe, list) or not universe or len(universe) > 256:
        raise ValueError("finite domain needs 1–256 explicit outcomes")
    if any(not isinstance(v, str) or not v for v in universe) or len(set(universe)) != len(
        universe
    ):
        raise ValueError("finite domain outcomes must be unique nonempty strings")
    domain = set(universe)
    if not isinstance(memberships, dict) or not memberships or len(memberships) > 8:
        raise ValueError("finite partition needs 1–8 buckets")
    groups = {}
    for key, values in memberships.items():
        if not isinstance(key, str) or not key or not isinstance(values, list) or not values:
            raise ValueError("buckets require ids and nonempty outcome lists")
        if (
            any(not isinstance(v, str) for v in values)
            or len(set(values)) != len(values)
            or set(values) - domain
        ):
            raise ValueError("bucket references duplicate or unknown outcomes")
        groups[key] = set(values)
    if parent is not None and (not isinstance(parent, list) or set(parent) - domain or not parent):
        raise ValueError("parent must be a nonempty subset of the declared domain")
    allowed = domain if parent is None else set(parent)
    union = set().union(*groups.values())
    pairs = list(combinations(groups, 2))
    return {
        "overlap_pairs": [[a, b] for a, b in pairs if groups[a] & groups[b]],
        "equivalent_pairs": [[a, b] for a, b in pairs if groups[a] == groups[b]],
        "outside_parent": sorted(union - allowed),
        "residual": sorted(allowed - union),
        "explicit_coverage": len(union & allowed) / len(allowed),
        "exhaustive": allowed <= union,
    }


def candidate_curriculum(split, count=32):
    if split not in SPLITS or type(count) is not int or count < 1:
        raise ValueError("invalid candidate split or count")
    variant = SPLITS.index(split)
    voices = (
        "Classify the displayed status",
        "Choose the recorded outcome",
        "Assign the shown category",
        "Determine the documented result",
        "Audit the observed state",
    )
    output = []
    for i in range(count):
        # Entire rule vocabulary/expressions are held out, including derived diagnostics.
        universe = [f"{split}-event-{i}-{j}" for j in range(4)]
        mode = "open" if i % 2 else "expand"
        parent = universe if mode == "open" else universe[1:]
        memberships = {"a": [parent[0]], "b": [parent[1]]}
        rows = [
            {"id": k, "description": "Status " + values[0], "excludes": "Every other status"}
            for k, values in memberships.items()
        ]
        question = {
            "type": "choice",
            "instructions": voices[variant] + ".",
            "candidate_generation": {
                "mode": mode,
                "scope": "One status in " + ", ".join(parent),
                "experimental": True,
                "max_new": 2,
                "max_tokens": 1024,
            },
        }
        if mode == "expand":
            question["criteria"] = {
                "original": "Status " + universe[0],
                "other": "Any of " + ", ".join(parent),
            }
            question["candidate_generation"]["other_id"] = "other"
        possible = universe[i % 4 : i % 4 + 1] if i % 3 else universe
        state = {"possible_statuses": possible, "domain": universe, "rule": voices[variant]}
        selected = next((k for k, values in memberships.items() if possible == values), "__other__")
        criteria = [
            Candidate(r["id"], r["description"] + "\nExcludes: " + r["excludes"]) for r in rows
        ]
        criteria.append(
            Candidate("__other__", "All remaining outcomes in the declared parent scope")
        )
        target = {c.id: 0.0 for c in criteria}
        relevant = [v for v in possible if v in parent]
        # Conditional ground truth; unknown evidence produces a distribution, not an invented observation.
        for v in relevant:
            key = next((k for k, values in memberships.items() if v in values), "__other__")
            target[key] += 1 / len(relevant)
        if not relevant:
            # The readout is conditional on the parent: include every parent outcome when evidence excludes it.
            target = {
                "a": 1 / len(parent),
                "b": 1 / len(parent),
                "__other__": (len(parent) - 2) / len(parent),
            }
        lineage = f"candidates-v1/{split}/{i}"
        metadata = {
            "source": "repository-authored",
            "license": "MIT",
            "split": split,
            "language": "en",
            "source_example_id": lineage,
            "source_lineage": lineage,
            "generator_template_id": f"candidates-v1/{split}",
            "rule_combination": f"{split}/{i}",
            "document_voice": voices[variant],
            "task_family": "candidate_partition",
            "proposal_supervision": {
                "validator": "finite-domain-v1",
                "universe": universe,
                "parent": parent,
                "memberships": memberships,
                "question": question,
                "rows": rows,
            },
            "partition_audit": finite_partition_audit(universe, memberships, parent),
            "information_sufficiency": len(possible) == 1 and bool(relevant),
            "selected_outcome": selected,
        }
        sample = Sample(
            state,
            [
                Question(
                    "q",
                    "choice",
                    question["instructions"] + " Condition on: " + ", ".join(parent),
                    criteria,
                    target,
                )
            ],
            metadata,
        )
        output.append(sample)
    return output


def partition_diagnostics(sample):
    """Independent Noul targets for overlap, completeness and evidence sufficiency."""
    annotation = sample.metadata["proposal_supervision"]
    domain, memberships, parent = (
        annotation["universe"],
        annotation["memberships"],
        annotation["parent"],
    )
    # Include exact equivalence, partial overlap and complete/incomplete coverage.
    index = int(sample.metadata["source_example_id"].rsplit("/", 1)[1])
    memberships = {k: list(v) for k, v in memberships.items()}
    if index % 4 == 1:
        memberships["b"] = list(memberships["a"])
    elif index % 4 == 2:
        memberships["b"] += memberships["a"]
    elif index % 4 == 3:
        memberships["c"] = parent[2:]
    audit = finite_partition_audit(domain, memberships, parent)
    questions = [
        Question.noul(
            "overlap", "Do any listed buckets share an outcome?", bool(audit["overlap_pairs"])
        ),
        Question.noul(
            "coverage",
            "Do the explicitly listed buckets cover every declared parent outcome? Ignore residual Other.",
            audit["exhaustive"],
        ),
        Question.noul(
            "sufficiency",
            "Does observed evidence uniquely identify an outcome inside the parent?",
            sample.metadata["information_sufficiency"],
        ),
    ]
    metadata = {
        k: v
        for k, v in sample.metadata.items()
        if k not in {"proposal_supervision", "partition_audit"}
    }
    metadata["task_family"] = "partition_diagnostics"
    state = {"evidence": sample.state, "memberships": memberships, "parent": parent}
    metadata["content_hash"] = hashlib.sha256(
        json.dumps(state, sort_keys=True).encode()
    ).hexdigest()
    return Sample(state, questions, metadata)
