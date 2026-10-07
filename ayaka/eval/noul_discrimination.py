"""Noul discrimination diagnostic over saved reads; nothing is fitted.

Separates two reasons a Noul question is not credited under the fixed
0.2/0.8 band: the model cannot rank true above false (AUC near 0.5), or it
ranks well but its probabilities stay inside the band. Per group it reports
AUC, served credit, abstentions and the mean P(true) for each gold label;
for two systems on the same questions it also counts state transitions
(correct / wrong / abstain).

Inputs are either ``pretraining_v2`` reports (``rows[mode]`` with
``confidence`` and ``calibration_target``) or ``checkpoint_comparison`` JSONL
rows (``probs``, ``candidate_ids`` and ``target``). Only hard 0/1 targets
enter AUC and the state counts; soft-target questions are counted separately.

    python -m ayaka.eval.noul_discrimination --report parent=a.json --report pilot=b.json \\
        --group family --out noul-discrimination.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

LOW, HIGH = 0.2, 0.8
VERSION = "ayaka-noul-discrimination-1"


def noul_rows(path, mode="off"):
    """Load Noul rows as {id: {"p": P(true), "y": gold P(true), **metadata}}."""
    path = Path(path)
    if path.suffix == ".jsonl":
        raw = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    else:
        raw = json.loads(path.read_bytes())["rows"][mode]
    rows = {}
    for row in raw:
        if row["type"] != "noul":
            continue
        if "candidate_ids" in row and "confidence" not in row:
            index = row["candidate_ids"].index("true")
            p, y = row["probs"][index], row["target"][index]
        else:
            p, y = row["confidence"], row["calibration_target"]
        if row["id"] in rows:
            raise ValueError(f"duplicate Noul question {row['id']}")
        rows[row["id"]] = {**row, "p": float(p), "y": float(y)}
    if not rows:
        raise ValueError(f"no Noul rows in {path}")
    return rows


def state(row):
    """correct, wrong or abstain under the fixed band; hard targets only."""
    if LOW < row["p"] < HIGH:
        return "abstain"
    return "correct" if (row["p"] >= HIGH) == (row["y"] == 1) else "wrong"


def auc(rows):
    """P(random gold-true question scores above a random gold-false one); ties count half."""
    pos = [r["p"] for r in rows if r["y"] == 1]
    neg = [r["p"] for r in rows if r["y"] == 0]
    if not pos or not neg:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def summarize(rows):
    hard = [r for r in rows if r["y"] in (0.0, 1.0)]
    states = Counter(state(r) for r in hard)
    by_gold = {
        str(int(gold)): sum(r["p"] for r in hard if r["y"] == gold)
        / sum(1 for r in hard if r["y"] == gold)
        for gold in (0.0, 1.0)
        if any(r["y"] == gold for r in hard)
    }
    return {
        "questions": len(rows),
        "hard_target_questions": len(hard),
        "soft_target_questions": len(rows) - len(hard),
        "auc": auc(hard),
        "correct": states["correct"],
        "wrong": states["wrong"],
        "abstain": states["abstain"],
        "mean_p_true_by_gold": by_gold,
    }


def discrimination(systems, group=None):
    """systems: {name: rows}. All systems must cover the same question IDs."""
    names = list(systems)
    ids = set(systems[names[0]])
    if any(set(rows) != ids for rows in systems.values()):
        raise ValueError("every system must cover the same Noul questions")
    ordered = sorted(ids)

    def key(i):
        return "all" if group is None else str(systems[names[0]][i].get(group))

    members = defaultdict(list)
    for i in ordered:
        members[key(i)].append(i)
    report = {
        "version": VERSION,
        "band": [LOW, HIGH],
        "group": group,
        "groups": {
            value: {name: summarize([systems[name][i] for i in group_ids]) for name in names}
            for value, group_ids in sorted(members.items())
        },
        "fitted": False,
    }
    if len(names) == 2:
        before, after = systems[names[0]], systems[names[1]]
        hard = [i for i in ordered if before[i]["y"] in (0.0, 1.0)]
        report["transitions"] = {
            f"{a}->{b}": n
            for (a, b), n in sorted(
                Counter((state(before[i]), state(after[i])) for i in hard).items()
            )
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a named saved report; pass two to also count state transitions",
    )
    parser.add_argument("--mode", default="off", help="row group in pretraining_v2 reports")
    parser.add_argument("--group", help="row field to group by, e.g. family, source, language")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError("diagnostic reports must use a new output file")
    systems = {}
    for item in args.report:
        name, sep, path = item.partition("=")
        if not sep or not name or name in systems:
            raise ValueError("each --report needs a unique NAME=PATH")
        systems[name] = noul_rows(path, args.mode)
    report = discrimination(systems, args.group)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
