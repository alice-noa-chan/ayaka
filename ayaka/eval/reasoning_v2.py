"""Serial paired evaluation; targets and family metadata never enter inference."""

import time

from ..primitives import QuestionSpec
from ..reasoning import ReasoningSettings
from ..training.batching import _noul_canonical
from .v2 import paired_report, summarize, typed_row


def evaluate_efforts(decision, samples, *, efforts=("low", "medium", "high"), deadline=None):
    modes = {"off": ReasoningSettings(mode="off")}
    modes.update({effort: ReasoningSettings(mode="on", effort=effort) for effort in efforts})
    rows = {mode: [] for mode in modes}
    complete = True
    for sample in samples:
        for question in sample.questions:
            q = _noul_canonical(question)
            spec = QuestionSpec(
                q.type,
                q.instruction,
                [c.description for c in q.candidates],
                [c.ordinal for c in q.candidates] if q.type == "score" else None,
            )
            target = [q.target_distribution.get(c.id, 0) for c in q.candidates]
            for mode, settings in modes.items():
                if deadline is not None and time.monotonic() >= deadline:
                    complete = False
                    break
                start = time.perf_counter()
                result = decision.decide(sample.state, [spec], reasoning=[settings])[0]
                row = typed_row(spec, result.probs, target)
                extra = result.extras["reasoning"]
                row.update(
                    id=f"{sample.metadata.get('source_example_id')}/{q.id}",
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
        if not complete:
            break
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
