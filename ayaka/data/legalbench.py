"""LegalBench (nguha/legalbench) -> rule/policy-application decisions.

LegalBench tasks carry individual licenses. Only tasks listed in the
vendored ``legalbench_tasks.json`` are used: those whose task README
states CC BY 4.0 (or a more permissive license), each with the task's
own instruction text (``base_prompt.txt`` up to the first worked example,
also CC BY 4.0). Non-commercial tasks are never in that file.

Per task, rows become a noul when the label set is {Yes, No}, otherwise a
choice over the task's label set. A per-task cap keeps a few large tasks
from dominating.
"""

from __future__ import annotations

import json
import random
import re
from importlib import resources

from .schema import Candidate, Question, Sample, one_hot

REPO = "nguha/legalbench"
# identifier columns never reach the model: some encode the label
# (e.g. sara_entailment case ids end in "_pos"/"_neg")
ID_COLUMN = re.compile(r"(?i)(^|[\s_])(id|idx|index|case[\s_]?id)$")


def allowed_tasks() -> dict[str, dict]:
    path = resources.files("ayaka.data") / "legalbench_tasks.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def row_to_sample(
    row: dict, task: str, instruction: str, labels: list[str], metadata: dict
) -> Sample | None:
    answer = str(row.get("answer", "")).strip()
    if answer not in labels:
        return None
    fields = {
        k: str(v)
        for k, v in row.items()
        if k != "answer" and not ID_COLUMN.search(k) and v not in (None, "")
    }
    if not fields:
        return None
    state = next(iter(fields.values())) if len(fields) == 1 else fields
    if {lb.lower() for lb in labels} == {"yes", "no"}:
        q = Question.noul("q0", instruction, float(answer.lower() == "yes"))
    else:
        cands = [Candidate(f"l{i}", lb) for i, lb in enumerate(labels)]
        q = Question("q0", "choice", instruction, cands, one_hot(cands, f"l{labels.index(answer)}"))
    md = dict(metadata)
    md.update({"legalbench_task": task, "source_example_id": f"{task}:{row.get('index', '')}"})
    return Sample(state=state, questions=[q], metadata=md)


def load_legalbench(
    limit: int | None, seed: int, metadata: dict, per_task: int = 400
) -> list[Sample]:
    from datasets import concatenate_datasets, load_dataset

    rng = random.Random(seed)
    out: list[Sample] = []
    for task, info in sorted(allowed_tasks().items()):
        parts = []
        for split in ("train", "test"):
            try:
                parts.append(load_dataset(REPO, task, split=split))
            except Exception as e:  # a missing split must not kill the rest
                print(f"[legalbench] {task}/{split} skipped: {e}", flush=True)
        if not parts:
            continue
        ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        if "answer" not in ds.column_names:
            continue
        labels = sorted({str(a).strip() for a in ds["answer"]})
        if not 2 <= len(labels) <= 26:
            continue
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        kept = 0
        for i in idx:
            s = row_to_sample(dict(ds[i]), task, info["instruction"], labels, metadata)
            if s is not None:
                out.append(s)
                kept += 1
            if kept >= per_task:
                break
    rng.shuffle(out)
    return out[:limit] if limit is not None else out
