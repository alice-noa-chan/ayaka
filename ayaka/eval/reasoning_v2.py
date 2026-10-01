"""Serial paired evaluation; targets and family metadata never enter inference."""

import copy
import time

from ..primitives import QuestionSpec
from ..reasoning import ReasoningSettings
from ..training.batching import _noul_canonical
from .v2 import paired_report, summarize, typed_row


def evaluate_efforts(
    decision,
    samples,
    *,
    efforts=("low", "medium", "high"),
    deadline=None,
    verbose=False,
    include_auto=False,
    resume_rows=None,
):
    samples = list(samples)
    modes = {"off": ReasoningSettings(mode="off")}
    modes.update({effort: ReasoningSettings(mode="on", effort=effort) for effort in efforts})
    if include_auto:
        modes["auto"] = ReasoningSettings(mode="auto")
    if not set(resume_rows or {}) <= modes.keys():
        raise ValueError("resume modes must match this evaluation")
    rows = {mode: copy.deepcopy((resume_rows or {}).get(mode, [])) for mode in modes}
    expected = {
        f"{s.metadata.get('source_example_id')}/{q.id}": q for s in samples for q in s.questions
    }
    if len(expected) != sum(len(s.questions) for s in samples):
        raise ValueError("evaluation requires unique question IDs")
    known = {}
    for mode, existing in rows.items():
        known[mode] = {r["id"] for r in existing}
        if len(known[mode]) != len(existing) or not known[mode] <= expected.keys():
            raise ValueError("resume rows must have unique IDs from this evaluation")
        for row in existing:
            q = _noul_canonical(expected[row["id"]])
            target = [q.target_distribution.get(c.id, 0) for c in q.candidates]
            if (
                row["target"] != target
                or row["type"] != q.type
                or row["budget"] != modes[mode].budget
            ):
                raise ValueError("resume targets, types and budgets must match")
    complete = True
    for sample in samples:
        for question in sample.questions:
            question_id = f"{sample.metadata.get('source_example_id')}/{question.id}"
            new_call = False
            q = _noul_canonical(question)
            spec = QuestionSpec(
                q.type,
                q.instruction,
                [c.description for c in q.candidates],
                [c.ordinal for c in q.candidates] if q.type == "score" else None,
            )
            target = [q.target_distribution.get(c.id, 0) for c in q.candidates]
            for mode, settings in modes.items():
                if question_id in known[mode]:
                    continue
                if deadline is not None and time.monotonic() >= deadline:
                    complete = False
                    break
                start = time.perf_counter()
                result = decision.decide(sample.state, [spec], reasoning=[settings])[0]
                new_call = True
                row = typed_row(spec, result.probs, target)
                extra = result.extras["reasoning"]
                if mode == "off":
                    from ..routing import routing_features

                    row["routing_features"] = extra.get("routing_features") or routing_features(
                        sample.state, spec, result, decision.tok
                    )
                row.update(
                    id=question_id,
                    family=sample.metadata.get("task_family", "unknown"),
                    language=sample.metadata.get("language", "en"),
                    split=sample.metadata.get("split"),
                    tier=sample.metadata.get("tier", "standard"),
                    latency_s=time.perf_counter() - start,
                    reasoning_tokens=extra["generated_tokens"],
                    input_tokens=extra["input_tokens"],
                    route=extra["route"],
                    budget=extra["budget"],
                    finish_reason=extra["finish_reason"],
                    error=extra["error"],
                    probs=result.probs,
                    target=target,
                )
                rows[mode].append(row)
            if not complete:
                break
            if verbose and new_call and min(map(len, rows.values())) % 16 == 0:
                print(
                    f"[eval] questions={len(rows['off'])} modes={','.join(modes)} reasoning_tokens={sum(r['reasoning_tokens'] for rs in rows.values() for r in rs)}",
                    flush=True,
                )
        if not complete:
            break
    positions = {key: index for index, key in enumerate(expected)}
    rows = {mode: sorted(rs, key=lambda r: positions[r["id"]]) for mode, rs in rows.items()}
    reports = {
        mode: {**summarize(rs), "status": "complete" if complete else "incomplete"}
        for mode, rs in rows.items()
        if rs
    }
    pairs = {
        mode: paired_report(rows["off"], rs)
        for mode, rs in rows.items()
        if mode != "off" and rs and [r["id"] for r in rs] == [r["id"] for r in rows["off"]]
    }
    return {
        "status": "complete" if complete else "incomplete",
        "reports": reports,
        "paired": pairs,
        "rows": rows,
    }
