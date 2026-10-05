"""Diagnose paired reasoning with calibration-only, path-specific temperatures.

This post-hoc saved-observation analysis changes no serving policy or teacher.
Fit both paths on the same completed calibration pairs, then evaluate fixed
temperatures on dev. Temperature cannot repair candidate ordering. Confidence
intervals condition on the fitted temperatures and observed trace availability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from ayaka.eval.read_artifact import fingerprint
from ayaka.swift.fit import fit_temperature, nll
from ayaka.swift.provenance import row_key

from .paired_diagnostic import (
    ROLES,
    TYPES,
    _clusters,
    _load,
    _metrics,
    _sha,
    _summary,
    diagnose_roles,
)

VERSION = "ayaka-paired-calibration-diagnostic-1"
ARMS = {
    "raw_paths": (False, False),
    "calibrated_direct_vs_raw_reasoned": (True, False),
    "raw_direct_vs_calibrated_reasoned": (False, True),
    "separately_calibrated_paths": (True, True),
}


def _completed(nested):
    trace = nested["pass_inputs"][0]["messages"][1]["content"]
    return nested["finish_reason"] == "eos" and nested["trace_tokens"] > 0 and bool(trace.strip())


def _observations(block):
    by_id = {row["id"]: row["reasoned_read"] for row in block["paired"]}
    result = []
    for direct in sorted(block["direct"], key=row_key):
        if direct["id"] not in by_id:
            continue
        nested = by_id[direct["id"]]
        reasoned = {
            **direct,
            **{key: nested[key] for key in ("raw_probs", "candidate_log_masses")},
        }
        result.append((direct, reasoned, _completed(nested)))
    return result


def _scaled_metrics(row, temperature):
    """Scale actual log masses, recovering underflowed mass without any floor."""
    peak = max(row["candidate_log_masses"].values())
    masses = {
        label: (mass - peak) / temperature for label, mass in row["candidate_log_masses"].items()
    }
    if any(not math.isfinite(value) for value in masses.values()):
        raise ValueError("temperature-scaled logit span exceeds finite arithmetic")
    values = {label: math.exp(value) for label, value in masses.items()}
    total = math.fsum(values.values())
    # A derived metric row is never exported as a new bound observation.
    return _metrics(
        {
            **row,
            "raw_probs": {label: value / total for label, value in values.items()},
            "candidate_log_masses": masses,
        }
    )


def _uniform_metrics(row):
    return _metrics(
        {
            **row,
            "raw_probs": dict.fromkeys(row["labels"], 1 / len(row["labels"])),
            "candidate_log_masses": dict.fromkeys(row["labels"], 0.0),
        }
    )


def _cohorts(items, sources, iterations, seed):
    result = {}
    for name, selected in (
        ("all_observed_pairs", items),
        ("completed_nonempty_eos", [item for item in items if item["completed_nonempty_eos"]]),
    ):
        result[name] = {
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
                for source in sources
            },
        }
    return result


def diagnose_calibration(reads, expected_cohorts, *, iterations=2000, seed=20261005):
    """Validate both inventories first; fit only completed non-public calibration.

    All source/type/arm comparisons are exploratory, not a selection gate.
    The pure function reports logical anchors; the CLI also requires byte anchors.
    """
    raw = diagnose_roles(reads, expected_cohorts, iterations=iterations, seed=seed)
    observed = {role: _observations(reads[role]) for role in ROLES}
    matched = [
        (direct, reasoned) for direct, reasoned, complete in observed["calibration"] if complete
    ]
    fit_clusters = _clusters(sorted(reads["calibration"]["direct"], key=row_key))
    fit_membership = fingerprint([(row["source"], row["id"]) for row, _ in matched])
    fits = {}
    for index, path in enumerate(("direct", "reasoned")):
        fits[path] = {}
        for kind in TYPES:
            rows = [pair[index] for pair in matched if pair[index]["type"] == kind]
            temperature = fit_temperature(rows)
            fits[path][kind] = {
                "temperature": temperature,
                "completed_pairs": len(rows),
                "connected_cases": len({fit_clusters[row["id"]] for row in rows}),
                "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
                "derived_fit_rows_sha256": fingerprint(rows),
                "nll_before": nll(rows, 1.0) if rows else None,
                "nll_after": nll(rows, temperature) if rows else None,
                "boundary": "lower"
                if temperature == 0.25
                else "upper"
                if temperature == 10
                else None,
                "status": "calibration_fitted"
                if rows
                else "no_completed_calibration_pairs_identity",
            }
    clusters = _clusters(sorted(reads["dev"]["direct"], key=row_key))
    summaries = {"raw_paths": raw["results"]["dev"]["cohorts"]}
    sources = sorted({row["source"] for row in reads["dev"]["direct"]})
    for arm, (scale_direct, scale_reasoned) in {**ARMS, "uniform_baseline": (False, False)}.items():
        if arm == "raw_paths":
            continue  # Reuse the validated baseline and its exact finite-bootstrap draws.
        items = []
        for direct, reasoned, complete in observed["dev"]:
            kind = direct["type"]
            items.append(
                {
                    "source": direct["source"],
                    "type": kind,
                    "cluster": clusters[direct["id"]],
                    "completed_nonempty_eos": complete,
                    "direct": _uniform_metrics(direct)
                    if arm == "uniform_baseline"
                    else _scaled_metrics(direct, fits["direct"][kind]["temperature"])
                    if scale_direct
                    else _metrics(direct),
                    "reasoned": _uniform_metrics(reasoned)
                    if arm == "uniform_baseline"
                    else _scaled_metrics(reasoned, fits["reasoned"][kind]["temperature"])
                    if scale_reasoned
                    else _metrics(reasoned),
                }
            )
        summaries[arm] = _cohorts(items, sources, iterations, seed)
    return {
        "version": VERSION,
        "status": "post_hoc_saved_observation_diagnostic_only",
        "identity": raw["identity"],
        "runtime_sha256": raw["runtime_sha256"],
        "reasoning_recipe_sha256": raw["reasoning_recipe_sha256"],
        "observation_provenance": {
            role: {
                key: raw["results"][role][key]
                for key in (
                    "canonical_fingerprint",
                    "inventory_fingerprints",
                    "coverage",
                    "finish_reasons",
                    "trace_tokens_all_observed",
                    "completed_nonempty_eos_pairs",
                )
            }
            for role in ROLES
        },
        "fit": {
            "role": "non_public_calibration",
            "subset": "same completed nonempty EOS paired IDs for both paths",
            "membership_sha256": fit_membership,
            "completed_pairs": len(matched),
            "objective": "question-weighted per-type gold NLL; no source-specific fit",
            "temperature_bounds": [0.25, 10.0],
            "implementation": "ayaka.swift.fit.fit_temperature",
            "convergence": "not attested by fitter; bounded golden-section objective search",
            "parameters": fits,
        },
        "dev_comparisons": summaries,
        "bootstrap": {
            **raw["bootstrap"],
            "conditioning": "fixed calibration temperatures; uncertainty of fitting not resampled",
        },
        "serving_policy_changed": False,
        "training_data_created": False,
        "promotable": False,
        "execution_attested": False,
        "temperature_effect": "argmax ordering preserved; Score expectations and Noul thresholds can change",
        "uniform_baseline": "equal candidate mass reference, not a fitted or serving policy",
        "scope": "post-hoc loss diagnosis; no serving gate, train benefit, causal trace effect, v1 superiority or ranking claim",
    }


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
    root = Path(__file__).resolve().parents[2]
    source_paths = [
        Path(__file__).resolve(),
        root / "scripts/direct_v2/paired_diagnostic.py",
        root / "ayaka/eval/read_artifact.py",
    ]
    source_paths.extend(sorted((root / "ayaka/swift").glob("*.py")))

    def sources():
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        }

    before = sources()
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
    report = diagnose_calibration(
        reads,
        {role: args[f"expected_{role}_cohort_sha256"] for role in ROLES},
        iterations=args["iterations"],
        seed=args["seed"],
    )
    if sources() != before:
        raise ValueError("diagnostic source changed during analysis; retry on frozen sources")
    report["file_anchors"] = anchors
    report["source_sha256"] = before
    report["diagnostic_script_sha256"] = before["scripts/direct_v2/paired_calibration.py"]
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
