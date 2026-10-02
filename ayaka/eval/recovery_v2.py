"""Matched, source-clustered recovery diagnostics; oracle traces are privileged input."""

from __future__ import annotations

import copy
import math
from collections import Counter

import torch

from ..primitives import QuestionSpec
from ..reasoning import ReasoningSettings
from ..reasoning_pipeline import Trace, TraceFailure, readout_suffix
from ..training.batching import _noul_canonical
from ..training.scoped_calibration import ScopedCalibration
from .v2 import paired_report, summarize, typed_row


def row_spec(row):
    return QuestionSpec(
        row["type"], "", [str(i) for i in range(len(row["probs"]))], row.get("ordinals")
    )


def recalibrate_report(report, calibrations):
    """Apply separately fitted artifacts, never fit on dev/test probabilities."""
    if not report.get("complete") or report.get("split") not in {"dev", "test", "public"}:
        raise ValueError("application requires a complete dev/test/public report")
    result = copy.deepcopy(report)
    for rows in result["rows"].values():
        for row in rows:
            calibration = calibrations.get((row["modality"], row["partition"]))
            if calibration is None:
                continue
            calibration.validate_binding(row["model_id"], row["modality"], row["partition"])
            budget = row["budget"] if row["route"] != "direct" else 0
            probs = calibration.apply(row["probs"], row["type"], row["route"], budget)
            row.update(typed_row(row_spec(row), probs, row["target"]), probs=probs)
    result["summary"] = {mode: summarize(rows) for mode, rows in result["rows"].items()}
    # Existing grouped and paired summaries describe old probabilities.
    for key in ("groups", "image_text_pairs", "reasoning_changes"):
        result.pop(key, None)
    result["calibration"] = "reserved_split_scoped_temperature"
    result["calibration_input"] = (
        "log(max(probability, 1e-12)); saturated logits are not recoverable"
    )
    return result


def fit_report_calibrations(report):
    if not report.get("complete") or report.get("split") != "calibration":
        raise ValueError("fitting requires the complete reserved calibration split")
    rows = [row for group in report["rows"].values() for row in group]
    identities = {row["model_id"] for row in rows}
    if len(identities) != 1 or identities != {report["model_id"]}:
        raise ValueError("calibration must have one exact checkpoint identity")
    domains = {(row["modality"], row["partition"]) for row in rows}
    return {
        domain: ScopedCalibration.fit(
            [row for row in rows if (row["modality"], row["partition"]) == domain],
            report["model_id"],
            *domain,
        )
        for domain in domains
    }


@torch.inference_mode()
def verified_trace_readout(generator, state, spec, notes):
    """Teacher-force verified text, then use exactly the production cached readout."""
    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("verified trace must be nonempty")
    ids, payload = generator.prepare(generator.messages_for(state, spec))
    if payload is not None:
        raise ValueError("oracle readout currently supports text evidence only")
    tokens = generator.tok.encode(notes)
    eos = getattr(getattr(generator.tok, "hf", None), "eos_token_id", None)
    if eos is not None:
        tokens.append(eos)
    suffix = readout_suffix(generator.tok, spec, generator.model.cfg.max_label_candidates)
    if (
        len(tokens) > 1024
        or len(ids) + len(tokens) + len(suffix.suffix_ids) > generator.max_context
    ):
        raise ValueError("complete verified trace and readout do not fit")
    _, cache = generator.prefill(
        torch.tensor([ids + tokens], device=generator.model.embed_weight().device), None
    )
    trace = Trace(text=notes, input_ids=ids, token_ids=tokens, cache=cache)
    return generator.readout(trace, spec)


def evaluate_trace_diagnostics(decision, samples, budget=128):
    """Only examples with independently verified traces; never treat oracle as deployment."""
    rows = {"off": [], "generated": [], "oracle": []}
    examples, skipped, seen = [], Counter(), set()
    for sample in samples:
        if "media" in sample.metadata:
            skipped["image_oracle_unsupported"] += len(sample.questions)
            continue
        for question in sample.questions:
            notes = sample.metadata.get("verified_traces", {}).get(question.id)
            if not notes:
                skipped["no_verified_trace"] += 1
                continue
            q = _noul_canonical(question)
            identity = sample.metadata["source_example_id"] + "/" + q.id
            if identity in seen:
                raise ValueError("trace diagnostics require unique question identities")
            seen.add(identity)
            spec = QuestionSpec(
                q.type,
                q.instruction,
                [c.description for c in q.candidates],
                [c.ordinal for c in q.candidates] if q.type == "score" else None,
            )
            target = [q.target_distribution.get(c.id, 0) for c in q.candidates]
            direct = decision.decide(
                sample.state, [spec], reasoning=[ReasoningSettings(mode="off")]
            )[0]
            generator = decision.generator
            suffix = readout_suffix(generator.tok, spec, generator.model.cfg.max_label_candidates)
            failure = None
            try:
                trace = generator.generate_trace(
                    generator.messages_for(sample.state, spec),
                    budget,
                    reserve=len(suffix.suffix_ids),
                )
                if not trace.text.strip():
                    raise TraceFailure(Trace(error="empty trace", token_ids=trace.token_ids))
                generated = generator.readout(trace, spec)
            except TraceFailure as exc:
                trace, generated, failure = exc.trace, direct.probs, exc.trace.error
            oracle = verified_trace_readout(generator, sample.state, spec, notes)
            for mode, probabilities in [
                ("off", direct.probs),
                ("generated", generated),
                ("oracle", oracle),
            ]:
                row = typed_row(spec, probabilities, target)
                row.update(
                    id=identity,
                    cluster_id=sample.metadata["source_lineage"],
                    language=sample.metadata["language"],
                    family=sample.metadata["task_family"],
                    split=sample.metadata["split"],
                    probs=probabilities,
                    target=target,
                    ordinals=spec.ordinals,
                    reasoning_tokens=trace.generated_tokens if mode == "generated" else 0,
                )
                rows[mode].append(row)
            examples.append(
                {
                    "id": identity,
                    "generated_trace": trace.text,
                    "verified_trace": notes,
                    "failure": failure,
                    "finish_reason": trace.finish_reason,
                }
            )
    return {
        "complete": True,
        "scope": "privileged verified-trace diagnosis, not deployable model accuracy",
        "rows": rows,
        "examples": examples,
        "skipped": dict(skipped),
        "summary": {mode: summarize(group) for mode, group in rows.items() if group},
        "paired": {
            mode: paired_report(rows["off"], group)
            for mode, group in rows.items()
            if mode != "off" and group
        },
    }


def promotion_screen(
    baseline, candidate, baseline_mode="off", candidate_mode="off", *, replicates=2000
):
    """Predeclared dev screen. A passing screen still requires a fresh final test."""
    if any(
        not report.get("complete") or report.get("split") != "dev"
        for report in (baseline, candidate)
    ):
        raise ValueError("screening requires complete dev reports, never test or partial runs")
    a, b = baseline["rows"][baseline_mode], candidate["rows"][candidate_mode]
    if not a or len(a) != len(b):
        raise ValueError("screening requires nonempty matched questions")
    for left, right in zip(a, b, strict=True):
        for key in ("id", "cluster_id", "type", "target", "ordinals", "language", "family"):
            if left.get(key) != right.get(key):
                raise ValueError(f"unmatched screening field: {key}")
    sa, sb, paired = summarize(a), summarize(b), paired_report(a, b, replicates)
    failures = []
    if paired["independent_cases"] < 200:
        failures.append("fewer_than_200_independent_dev_cases")
    if sb["cc_equal_types"] - sa["cc_equal_types"] < 5:
        failures.append("overall_gain_below_5_points")
    if paired["cc_delta_95ci"][0] <= 0:
        failures.append("overall_gain_not_supported_by_paired_95ci")
    for kind in ("choice", "noul", "score"):
        old, new = sa["by_type"].get(kind), sb["by_type"].get(kind)
        if old is None or new is None:
            failures.append(f"missing_type:{kind}")
            continue
        if new["cc"] < old["cc"] - 1:
            failures.append(f"type_regression:{kind}")
        for metric in ("nll", "brier", *(("rps",) if kind == "score" else ())):
            if not math.isfinite(new[metric]) or new[metric] > old[metric] + 1e-8:
                failures.append(f"probability_regression:{kind}/{metric}")
    for language in ("en", "ko", "ja"):
        old = [row for row in a if row["language"] == language]
        new = [row for row in b if row["language"] == language]
        if not old:
            failures.append(f"missing_language:{language}")
        elif summarize(new)["cc_equal_types"] < summarize(old)["cc_equal_types"] - 1:
            failures.append(f"language_regression:{language}")
    numeric = {
        "month_end",
        "leap",
        "business",
        "timezone",
        "rounding",
        "calendar",
        "temporal_numeric",
    }
    old = [row for row in a if row["family"] in numeric]
    new = [row for row in b if row["family"] in numeric]
    if not old:
        failures.append("missing_temporal_numeric_evidence")
    elif summarize(new)["cc_equal_types"] - summarize(old)["cc_equal_types"] < 5:
        failures.append("temporal_numeric_gain_below_5_points")
    return {
        "screen_passed": not failures,
        "failures": failures,
        "baseline": sa,
        "candidate": sb,
        "paired": paired,
        "final_test_required": True,
        "official_composite": None,
    }
