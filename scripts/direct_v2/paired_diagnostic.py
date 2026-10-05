"""Diagnose saved calibration/dev reasoning pairs without fitting or training.

Raw question-weighted results describe only the observed subset. Source/type
coverage and connected-case bootstrap intervals expose regressions hidden by a
pooled mean; they do not select teachers, promote a model or attest GPU execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from ayaka.eval.read_artifact import fingerprint
from ayaka.swift.binding import SwiftReadIndex, validate_bound_reads
from ayaka.swift.losses import row_nll, target_distribution
from ayaka.swift.provenance import assert_roles_isolated, row_key
from ayaka.swift.router import validate_pairs

ROLES = ("calibration", "dev")
TYPES = ("choice", "noul", "score")
VERSION = "ayaka-paired-observation-diagnostic-1"


def _sha(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("expected anchors must be exact lowercase SHA256 values")
    return value


def _metrics(row):
    target, probs, labels = target_distribution(row), row["raw_probs"], row["labels"]
    result = {
        "nll": row_nll(row),
        "brier": math.fsum((probs[label] - target[label]) ** 2 for label in labels),
        # This diagnostic is neither Noul's abstention credit nor Score's metric.
        "hard_argmax_matches_gold": float(
            max(labels, key=probs.__getitem__) == max(labels, key=target.__getitem__)
        ),
    }
    if row["type"] == "score":
        values = [float(label) for label in labels]
        if any(not math.isfinite(value) for value in values) or any(
            a >= b for a, b in zip(values, values[1:], strict=False)
        ):
            raise ValueError("Score labels must be finite and strictly numerically ordered")
        expected = math.fsum(
            probs[label] * value for label, value in zip(labels, values, strict=True)
        )
        span = values[-1] - values[0]
        result["normalized_expected_absolute_error"] = math.fsum(
            target[label] * abs(expected - value) / span
            for label, value in zip(labels, values, strict=True)
        )
        p_cdf = t_cdf = 0.0
        errors = []
        for label in labels[:-1]:
            p_cdf += probs[label]
            t_cdf += target[label]
            errors.append((p_cdf - t_cdf) ** 2)
        result["rps"] = mean(errors)
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("diagnostic metrics exceed finite arithmetic")
    return result


def _relation_keys(row):
    keys = [
        ("state", fingerprint(row["binding"]["state"])),
        ("input", row["rendered_input_sha256"]),
    ]
    ancestry = (row["case_id"], row["cluster_id"], *row["lineage_ids"])
    if not all(isinstance(value, str) and value for value in ancestry):
        raise ValueError("connected-case intervals require explicit nonempty ancestry")
    # Case/cluster IDs retain their declared source scope. Explicit lineages
    # also join globally, conservatively merging coincident unqualified IDs.
    keys.extend(
        ("ancestry", value if ":" in value else f"{row['source']}:{value}") for value in ancestry
    )
    keys.extend(("ancestry", value) for value in row["lineage_ids"])
    return keys


def _clusters(rows):
    """Keep declared ancestry and exact shared evidence together, transitively.

    Construct on the entire direct inventory, including unobserved bridge rows,
    before taking completion/source/type subsets. IDs never enter the report.
    """
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, row in enumerate(rows):
        for key in _relation_keys(row):
            if key in owners:
                left, right = root(index), root(owners[key])
                parents[max(left, right)] = min(left, right)
            else:
                owners[key] = index
    return {row["id"]: root(index) for index, row in enumerate(rows)}


def _interval(items, key, iterations, seed):
    totals = defaultdict(list)
    for item in items:
        totals[item["cluster"]].append(item["reasoned"][key] - item["direct"][key])
    groups = [(math.fsum(values), len(values)) for _, values in sorted(totals.items())]
    if len(groups) < 2:
        return {
            "clusters": len(groups),
            "delta_ci95": None,
            "reason": "fewer_than_two_connected_cases",
        }
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        selected = rng.choices(groups, k=len(groups))
        deltas.append(
            math.fsum(value for value, _ in selected) / sum(count for _, count in selected)
        )
    deltas.sort()
    return {
        "clusters": len(groups),
        "delta_ci95": [
            deltas[math.floor(0.025 * (iterations - 1))],
            deltas[math.ceil(0.975 * (iterations - 1))],
        ],
    }


def _summary(items, *, iterations, seed):
    keys = sorted({key for item in items for key in item["direct"]})
    metrics = {}
    for key in keys:
        applicable = [item for item in items if key in item["direct"]]
        metrics[key] = {
            "pairs": len(applicable),
            "direct": mean(item["direct"][key] for item in applicable),
            "reasoned": mean(item["reasoned"][key] for item in applicable),
            "reasoned_minus_direct": mean(
                item["reasoned"][key] - item["direct"][key] for item in applicable
            ),
            **_interval(applicable, key, iterations, seed),
        }
    return {
        "pairs": len(items),
        "connected_cases": len({item["cluster"] for item in items}),
        "metrics": metrics,
        "nll_improved": sum(item["reasoned"]["nll"] < item["direct"]["nll"] for item in items),
        "nll_worsened": sum(item["reasoned"]["nll"] > item["direct"]["nll"] for item in items),
        "hard_argmax_wrong_to_correct": sum(
            not item["direct"]["hard_argmax_matches_gold"]
            and item["reasoned"]["hard_argmax_matches_gold"]
            for item in items
        ),
        "hard_argmax_correct_to_wrong": sum(
            item["direct"]["hard_argmax_matches_gold"]
            and not item["reasoned"]["hard_argmax_matches_gold"]
            for item in items
        ),
    }


def _coverage(rows, paired_ids):
    canonical = [row for row in rows if row["readout"] == "canonical_letter_raw"]
    observed = sum(row["id"] in paired_ids for row in canonical)
    return {
        "direct_rows": len(rows),
        "canonical_rows": len(canonical),
        "excluded_grouped_rows": len(rows) - len(canonical),
        "observed_pairs": observed,
        "canonical_unobserved": len(canonical) - observed,
        "canonical_observed_fraction": observed / len(canonical) if canonical else None,
    }


def diagnose_roles(reads, expected_cohorts, *, iterations=2000, seed=20261005):
    """Require anchored canonical cohorts and bound calibration/dev observations.

    This pure function returns content fingerprints for complete inventories;
    external byte anchors for excluded/paired rows are the caller's responsibility.
    The CLI additionally requires all four exact file SHA256 values.
    """
    if set(reads) != set(ROLES) or set(expected_cohorts) != set(ROLES):
        raise ValueError("require exactly calibration and dev roles")
    if type(iterations) is not int or not 100 <= iterations <= 10000 or type(seed) is not int:
        raise ValueError("bootstrap requires 100..10000 iterations and an integer seed")
    all_rows, canonicals, pairs = {}, {}, {}
    identities, runtimes, recipes = set(), set(), set()
    for role in ROLES:
        block = reads[role]
        if not isinstance(block, dict) or set(block) != {"direct", "paired"}:
            raise ValueError("each role requires only direct and paired row lists")
        rows, paired = block["direct"], block["paired"]
        if not isinstance(rows, list) or not rows or not isinstance(paired, list):
            raise ValueError("direct inventory must be nonempty; paired inventory may be empty")
        for row in rows + paired:
            if (
                row.get("split") != role
                or row.get("public") is not False
                or row.get("type") not in TYPES
            ):
                raise ValueError("require explicit non-public calibration/dev and known types")
            identities.add(
                tuple(
                    row.get(key)
                    for key in ("model", "revision", "tokenizer_revision", "prompt_variant")
                )
            )
        if any("reasoned_read" in row for row in rows):
            raise ValueError("direct inventory cannot already contain a reasoned observation")
        SwiftReadIndex(rows)
        canonical, excluded = [], []
        for row in rows:
            (canonical if row["readout"] == "canonical_letter_raw" else excluded).append(row)
        if not canonical or any(
            row["readout"] != "grouped_approx" or len(row["labels"]) <= 26 for row in excluded
        ):
            raise ValueError("exclude only explicitly declared >26-label grouped reads")
        canonical.sort(key=row_key)
        if fingerprint(canonical) != _sha(expected_cohorts[role]):
            raise ValueError("canonical sorted source/id cohort differs from external anchor")
        validate_bound_reads(canonical)
        pairs[role] = validate_pairs(canonical, paired, require_candidates=False)
        runtimes.update(row["binding"]["runtime_sha256"] for row in canonical)
        recipes.update(value["recipe_sha256"] for value in pairs[role].values())
        all_rows[role], canonicals[role] = sorted(rows, key=row_key), canonical
    if len(identities) != 1 or len(runtimes) != 1 or len(recipes) > 1:
        raise ValueError("diagnostic must compare one model/tokenizer/runtime/prompt/trace recipe")
    assert_roles_isolated(all_rows["calibration"], all_rows["dev"])
    relations = [{key for row in all_rows[role] for key in _relation_keys(row)} for role in ROLES]
    if relations[0] & relations[1]:
        raise ValueError("calibration/dev connected ancestry or exact evidence overlap")
    results = {}
    for role in ROLES:
        rows, paired = all_rows[role], pairs[role]
        clusters = _clusters(rows)
        items = []
        for row in canonicals[role]:
            nested = paired.get(row["id"])
            if nested is None:
                continue
            trace = nested["pass_inputs"][0]["messages"][1]["content"]
            final = {**row, **{key: nested[key] for key in ("raw_probs", "candidate_log_masses")}}
            items.append(
                {
                    "source": row["source"],
                    "type": row["type"],
                    "cluster": clusters[row["id"]],
                    "direct": _metrics(row),
                    "reasoned": _metrics(final),
                    "finish_reason": nested["finish_reason"],
                    "trace_tokens": nested["trace_tokens"],
                    "completed_nonempty_eos": nested["finish_reason"] == "eos"
                    and nested["trace_tokens"] > 0
                    and bool(trace.strip()),
                }
            )
        cohorts = {}
        for name, selected in (
            ("all_observed_pairs", items),
            ("completed_nonempty_eos", [item for item in items if item["completed_nonempty_eos"]]),
        ):
            cohorts[name] = {
                "all": _summary(selected, iterations=iterations, seed=seed),
                "by_type": {
                    kind: _summary(
                        [item for item in selected if item["type"] == kind],
                        iterations=iterations,
                        seed=seed,
                    )
                    for kind in TYPES
                },
                "by_source": {
                    source: _summary(
                        [item for item in selected if item["source"] == source],
                        iterations=iterations,
                        seed=seed,
                    )
                    for source in sorted({row["source"] for row in rows})
                },
            }
        results[role] = {
            "canonical_fingerprint": expected_cohorts[role],
            "inventory_fingerprints": {
                kind: fingerprint(sorted(reads[role][kind], key=row_key))
                for kind in ("direct", "paired")
            },
            "coverage": {
                "all": _coverage(rows, paired),
                "by_source": {
                    source: _coverage([row for row in rows if row["source"] == source], paired)
                    for source in sorted({row["source"] for row in rows})
                },
                "by_type": {
                    kind: _coverage([row for row in rows if row["type"] == kind], paired)
                    for kind in TYPES
                },
            },
            "finish_reasons": dict(
                sorted(Counter(item["finish_reason"] for item in items).items())
            ),
            "trace_tokens_all_observed": sum(item["trace_tokens"] for item in items),
            "completed_nonempty_eos_pairs": sum(item["completed_nonempty_eos"] for item in items),
            "cohorts": cohorts,
        }
    return {
        "version": VERSION,
        "status": "saved_observation_diagnostic_only",
        "results": results,
        "identity": dict(
            zip(
                ("model", "revision", "tokenizer_revision", "prompt_variant"),
                next(iter(identities)),
                strict=True,
            )
        ),
        "runtime_sha256": next(iter(runtimes)),
        "reasoning_recipe_sha256": sorted(recipes),
        "bootstrap": {
            "iterations": iterations,
            "seed": seed,
            "unit": "connected ancestry/exact-evidence case",
            "estimand": "question-weighted reasoned-minus-direct on observed subset",
            "interval": "percentile 95%; exploratory; no multiple-comparison adjustment",
            "completion_conditioning": "completed-EOS subset is conditional, not an all-request causal estimate",
        },
        "policy_fitted_or_changed": False,
        "training_data_created": False,
        "promotable": False,
        "execution_attested": False,
        "scope": "non-public historical calibration/dev only; no holdout, train selection, unseen-cohort, natural-policy, v1-superiority or official ranking claim",
    }


def _load(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha(expected):
        raise ValueError("actual saved read bytes differ from external file anchor")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ROLES:
        parser.add_argument(f"--expected-{role}-cohort-sha256", required=True)
        for kind in ("direct", "paired"):
            parser.add_argument(f"--{role}-{kind}", required=True, type=Path)
            parser.add_argument(f"--expected-{role}-{kind}-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261005)
    args = vars(parser.parse_args())
    out = args["out"]
    if out.exists() or not out.parent.is_dir():
        raise ValueError("output requires an existing parent and a fresh report path")
    source_root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__).resolve(), source_root / "ayaka/eval/read_artifact.py"]
    source_paths.extend(sorted((source_root / "ayaka/swift").glob("*.py")))
    source_anchors = {
        path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    reads, anchors = {}, {}
    for role in ROLES:
        reads[role] = {}
        for kind in ("direct", "paired"):
            name = f"{role}_{kind}"
            anchors[name] = {
                "path": str(args[name]),
                "sha256": _sha(args[f"expected_{name}_sha256"]),
            }
            reads[role][kind] = _load(args[name], anchors[name]["sha256"])
    report = diagnose_roles(
        reads,
        {role: args[f"expected_{role}_cohort_sha256"] for role in ROLES},
        iterations=args["iterations"],
        seed=args["seed"],
    )
    report["file_anchors"] = anchors
    after = {
        path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    if source_anchors != after:
        raise ValueError("diagnostic source changed during analysis; retry on frozen sources")
    report["source_sha256"] = source_anchors
    report["diagnostic_script_sha256"] = source_anchors["scripts/direct_v2/paired_diagnostic.py"]
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "report": str(out),
                "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "promotable": False,
            }
        )
    )


if __name__ == "__main__":
    main()
