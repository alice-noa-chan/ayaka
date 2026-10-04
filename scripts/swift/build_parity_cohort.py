"""Deterministically derive parity fixtures from non-public calibration data.

This is a tokenizer/kernel stress cohort, not an accuracy evaluation sample.
20/26-option and skew fixtures add explicitly wrong synthetic distractors;
two permutations preserve the same state, candidate descriptions and target.
Never reads dev, test, train or public inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.eval.read_artifact import fingerprint  # noqa: E402
from ayaka.swift.collect import iter_dataset  # noqa: E402
from scripts.swift.inputs import REPO, sha256  # noqa: E402

DEFAULT_SOURCE = REPO / "runs/v2-pretraining-20261002-ready/calibration.jsonl"
OUTPUT = Path(__file__).with_name("parity_cohort.jsonl")


def build(source=DEFAULT_SOURCE):
    source = Path(source)
    if source.name != "calibration.jsonl":
        raise ValueError("cohort source must be the declared calibration.jsonl")
    items = list(iter_dataset([source]))
    if any(item.public or item.split != "calibration" for item in items):
        raise ValueError("cohort source must be entirely non-public calibration")
    selected = {
        kind: min((i for i in items if i.question.type == kind), key=lambda i: i.id)
        for kind in ("choice", "noul", "score")
    }
    receipt = {"builder": "swift-parity-cohort-v1", "source_sha256": sha256(source)}
    rows = []
    for name, kind, count, skew in (
        ("binary", "noul", 2, False),
        ("twenty", "choice", 20, False),
        ("twentysix", "choice", 26, False),
        ("skew", "choice", 2, True),
        ("score", "score", 5, False),
    ):
        item = selected[kind]
        labels = ["no", "yes"] if kind == "noul" else [str(i) for i in range(count)]
        target = item.gold if isinstance(item.gold, str) else max(item.gold, key=item.gold.get)
        description = item.question.descriptions[item.question.labels.index(target)]
        descriptions = [description] + [
            f"Parity distractor {i}: contradicts the required answer." for i in range(1, count)
        ]
        if kind == "noul":
            descriptions = ["no", "yes"]
        instruction = item.question.instruction
        if skew:
            instruction += (
                "\nFixture rule: select option 0; option 1 contradicts this explicit rule."
            )
        rows.append(
            {
                "id": f"parity:{name}",
                "source": "swift-parity-calibration",
                "case_id": f"parity:{item.case_id}",
                "split": "calibration",
                "public": False,
                "state": item.state,
                "labels": labels,
                "expected": labels[0],
                "question": {
                    "type": kind,
                    "instructions": instruction,
                    "criteria": dict(zip(labels, descriptions, strict=True)),
                },
                "provenance": {
                    **receipt,
                    "source_item_id": item.id,
                    "synthetic_distractors": kind != "noul",
                    "skew_fixture": skew,
                },
            }
        )
    original = rows[2]
    for name, order in (
        ("reverse", list(reversed(original["labels"]))),
        ("rotate", original["labels"][7:] + original["labels"][:7]),
    ):
        row = json.loads(json.dumps(original))
        row["id"] = f"parity:twentysix-{name}"
        row["labels"] = order
        row["question"]["criteria"] = {
            label: original["question"]["criteria"][label] for label in order
        }
        row["provenance"]["permutation_of"] = original["id"]
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    rows = build(args.source)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Fixed {len(rows)} fixtures; fingerprint={fingerprint(rows)}")


if __name__ == "__main__":
    main()
