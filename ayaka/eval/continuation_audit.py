"""Decompose saved continuation regressions without loading weights or opening test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from .v2 import paired_report, summarize, typed_row


def checked_rows(report):
    if report.get("complete") is not True or report.get("split") != "dev":
        raise ValueError("audit requires complete dev reports; test stays unopened")
    identity = report.get("model_id")
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise ValueError("report needs an exact checkpoint fingerprint")
    rows = report.get("rows", {}).get("off", [])
    if not rows:
        raise ValueError("audit needs nonempty reasoning-off rows")
    checked, seen = [], set()
    for original in rows:
        if not original.get("id") or original["id"] in seen:
            raise ValueError("question identities must be nonempty and unique")
        seen.add(original["id"])
        if original.get("model_id") != identity or original.get("split") != "dev":
            raise ValueError("row checkpoint/split binding differs from report")
        if original.get("route") != "direct" or any(
            original.get(key, 0) != 0 for key in ("budget", "reasoning_tokens", "generated_tokens")
        ):
            raise ValueError("off audit cannot contain generation or reasoned paths")
        if original.get("type") not in {"choice", "noul", "score"}:
            raise ValueError("unsupported decision type")
        if not original.get("cluster_id"):
            raise ValueError("audit requires underlying-case identities")
        probs, target = original["probs"], original["target"]
        if original["type"] == "noul" and len(probs) != 2:
            raise ValueError("Noul probabilities must be ordered false/true")
        ordinals = original.get("ordinals")
        if original["type"] == "score" and (
            not isinstance(ordinals, list)
            or len(ordinals) != len(probs)
            or len(set(ordinals)) != len(ordinals)
            or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in ordinals)
        ):
            raise ValueError("Score needs distinct finite ordinals aligned with candidates")
        spec = SimpleNamespace(type=original["type"], ordinals=ordinals)
        # Recompute from probabilities: stale cached metrics cannot hide a regression.
        row = {**original, **typed_row(spec, probs, target)}
        pred = max(range(len(probs)), key=probs.__getitem__)
        row["argmax_credit"] = target[pred]
        checked.append(row)
    return sorted(checked, key=lambda row: row["id"])


def changes(before, after):
    old, new = summarize(before), summarize(after)
    result = {
        "n": len(before),
        "independent_cases": len({row["cluster_id"] for row in before}),
        "before": old,
        "after": new,
        "cc_delta": new["cc_equal_types"] - old["cc_equal_types"],
        "by_type": {},
    }
    for kind in old["by_type"]:
        pairs = [(a, b) for a, b in zip(before, after, strict=True) if a["type"] == kind]
        a, b = old["by_type"][kind], new["by_type"][kind]
        values = {"n": len(pairs), "cc_delta": b["cc"] - a["cc"]}
        for metric in ("nll", "brier", "ece", *(("rps",) if kind == "score" else ())):
            values[f"{metric}_delta"] = b[metric] - a[metric]
        if kind == "score":
            delta = [a["nmae"] - b["nmae"] for a, b in pairs]
            values.update(
                improved=sum(v > 1e-6 for v in delta),
                worsened=sum(v < -1e-6 for v in delta),
                count_tolerance=1e-6,
            )
        else:
            delta = [b["correct"] - a["correct"] for a, b in pairs]
            values.update(fixed=sum(v > 0 for v in delta), broken=sum(v < 0 for v in delta))
        if kind == "noul":
            transitions = Counter()
            for left, right in pairs:
                state = lambda row: (  # noqa: E731
                    "abstain" if row["abstained"] else ("yes" if row["probs"][1] >= 0.8 else "no")
                )
                transitions[f"{state(left)}->{state(right)}"] += 1
            count = len(pairs)
            argmax_delta = sum(b["argmax_credit"] - a["argmax_credit"] for a, b in pairs) / count
            abstention_delta = (
                sum(
                    (b["argmax_credit"] - b["correct"]) - (a["argmax_credit"] - a["correct"])
                    for a, b in pairs
                )
                / count
            )
            values.update(
                transitions=dict(sorted(transitions.items())),
                argmax_credit_delta=argmax_delta,
                abstention_credit_penalty_delta=abstention_delta,
                thresholded_credit_delta=argmax_delta - abstention_delta,
                decomposition="thresholded credit = argmax credit - credit lost to abstention",
                nll_improved_but_cc_regressed=b["nll"] < a["nll"] and b["cc"] < a["cc"],
            )
        result["by_type"][kind] = values
    return result


def audit_continuation(parent, candidate, workload=None, history=None, *, replicates=2000):
    if type(replicates) is not int or replicates < 1:
        raise ValueError("bootstrap replicate count must be positive")
    before, after = checked_rows(parent), checked_rows(candidate)
    if [row["id"] for row in before] != [row["id"] for row in after]:
        raise ValueError("compare the complete identical cohort, never its intersection")
    fields = (
        "cluster_id",
        "type",
        "target",
        "ordinals",
        "language",
        "family",
        "modality",
        "partition",
    )
    for a, b in zip(before, after, strict=True):
        if any(a.get(key) != b.get(key) for key in fields) or a.get("tier", "standard") != b.get(
            "tier", "standard"
        ):
            raise ValueError("question metadata or case binding differs")
    result = {
        "version": 1,
        "scope": "saved dev observations; cannot identify a causal training mechanism",
        "parent_checkpoint_sha256": parent["model_id"],
        "candidate_checkpoint_sha256": candidate["model_id"],
        "execution": {"model_forwards": 0, "optimizer_steps": 0, "new_gpu_seconds": 0},
        "test_opened": False,
        "overall": changes(before, after),
        "paired": paired_report(before, after, replicates),
        "groups": {},
        "unresolved": [
            "data shift versus trace CE interference versus readout training",
            "independent natural-document transfer",
            "whether parent-distribution replay prevents the observed regressions",
        ],
    }
    for field in ("language", "family"):
        result["groups"][field] = {
            value: changes(
                [row for row in before if row[field] == value],
                [row for row in after if row[field] == value],
            )
            for value in sorted({row[field] for row in before})
        }
    if workload is not None:
        total = workload["total_rows"]
        if type(total) is not int or total < 1:
            raise ValueError("workload needs a positive complete row count")
        for values in workload["counts"].values():
            if (
                any(type(n) is not int or n < 0 for n in values.values())
                or sum(values.values()) != total
            ):
                raise ValueError("workload stratum counts must sum to every scheduled row")
        result["training_exposure"] = {
            "total_rows": total,
            "fractions": {
                field: {key: count / total for key, count in values.items()}
                for field, values in workload["counts"].items()
            },
            "unique_source_lineages": workload.get("unique_source_lineages"),
            "schedule_sha256": workload.get("schedule_sha256"),
        }
    if history is not None:
        if not history or [row["step"] for row in history] != list(range(1, len(history) + 1)):
            raise ValueError("history must cover the complete contiguous schedule")
        if workload is not None and len(history) != workload["steps"]:
            raise ValueError("history and workload schedule lengths differ")
        window = min(20, len(history))
        result["training_losses"] = {}
        for metric in ("total", "nll", "reasoning_ce", "pointer_nll", "brier", "rps"):
            if not all(metric in row for row in history):
                continue
            if any(not math.isfinite(row[metric]) for row in history):
                raise ValueError("training history contains nonfinite losses")
            result["training_losses"][metric] = {
                "first_window": sum(row[metric] for row in history[:window]) / window,
                "last_window": sum(row[metric] for row in history[-window:]) / window,
                "window_steps": window,
            }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError("audit requires a new output file; never overwrite receipts")
    paths = {key: getattr(args, key) for key in ("parent", "candidate", "workload", "history")}
    inputs = {key: path.read_bytes() for key, path in paths.items() if path is not None}
    result = audit_continuation(
        **{key: json.loads(value) for key, value in inputs.items()}, replicates=args.replicates
    )
    result["input_sha256"] = {
        key: hashlib.sha256(value).hexdigest() for key, value in inputs.items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"out": str(args.out), "cc_delta": result["overall"]["cc_delta"], **result["execution"]}
        )
    )
    return result


if __name__ == "__main__":
    main()
