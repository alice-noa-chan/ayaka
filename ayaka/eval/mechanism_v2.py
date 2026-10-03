"""Frozen-weight diagnostics; oracle/context interventions are not deployment scores."""

import argparse
import copy
import gc
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import replace
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from time import monotonic

import torch

from ..checkpoint import load_checkpoint
from ..collate import EncodedQuestion, encode_decision, suffix_rows
from ..data.recovery_holdout import POLICIES
from ..data.schema import Sample
from ..model.electra import PRIMITIVE_INDEX, length_bucket
from ..model.ragged import ragged_softmax
from ..primitives import QuestionSpec
from ..prompt import render_prefix
from ..reasoning_pipeline import Trace, TraceGenerator, controlled_decision, readout_suffix
from ..tokenization import HFTokenizer
from ..training.batching import _noul_canonical
from ..training.prepare_v2 import canonical, sha256
from ..training.scoped_calibration import checkpoint_fingerprint
from .v2 import paired_report, summarize, typed_row

HEADS = ("lm", "pointer", "hybrid")
CONDITIONS = ("direct", "empty", "generated", "oracle", "distractor")
SEED = 20261003


def reference_notes(sample):
    """Recompute from visible facts; never copy a target into the diagnostic trace."""
    import calendar

    rule = sample.metadata["semantic_rule"]
    if sample.state["policy"] != POLICIES["en"][rule]:
        raise ValueError("oracle requires the exact declared visible rule")
    r = sample.state["record"]
    if rule == "inclusive_business":
        start = date.fromisoformat(r["start"])
        holidays = {date.fromisoformat(x) for x in r["holidays"]}
        eligible = []
        current = start
        while len(eligible) < r["required"]:
            if current.weekday() not in r["weekend_weekdays"] and current not in holidays:
                eligible.append(current)
            current += timedelta(days=1)
        deadline = eligible[-1]
        delivered = date.fromisoformat(r["delivery"])
        value = max(0, (delivered - deadline).days)
        text = (
            f"Counting inclusively, the eligible dates are {', '.join(map(str, eligible))}. "
            f"The deadline is {deadline}. Delivery is {delivered}; "
            f"nonnegative calendar days late = {value}."
        )
    elif rule == "month_clipping":
        start = date.fromisoformat(r["start"])
        year, month0 = divmod(start.year * 12 + start.month - 1 + r["months"], 12)
        limit = calendar.monthrange(year, month0 + 1)[1]
        value = min(start.day, limit)
        text = (
            f"Shifting {start} by {r['months']} months gives year {year}, month {month0 + 1}. "
            f"That month has {limit} days; clipping {start.day} gives day {value}."
        )
    elif rule in {"taxable_shipping", "serial_discount"}:
        items = r["unit_cents"] * r["quantity"]

        def rounded(n):
            return int(n.quantize(Decimal(1), rounding=ROUND_HALF_UP))

        first = rounded(Decimal(items) * (100 - r["discount_percent"]) / 100)
        if rule == "taxable_shipping":
            base = first + r["shipping_cents"]
            value = rounded(Decimal(base) * (100 + r["tax_percent"]) / 100)
            text = (
                f"Items cost {items} cents. The first discount rounds to {first}. "
                f"Adding taxable shipping gives {base}; applying tax and half-up rounding "
                f"gives {value} cents."
            )
        else:
            second = rounded(Decimal(first) * (100 - r["second_discount_percent"]) / 100)
            taxed = rounded(Decimal(second) * (100 + r["tax_percent"]) / 100)
            value = taxed + r["shipping_cents"]
            text = (
                f"Items cost {items} cents. Sequential discounts round to {first}, then "
                f"{second}. Tax rounds to {taxed}. Adding untaxed shipping gives {value} cents."
            )
    elif rule == "absolute_cancellation":
        active = not r["cancelled"]
        affordable = r["cost"] <= r["limit"]
        qualified = r["override"] or r["credential"] >= r["required"]
        value = int(active and affordable and qualified)
        text = (
            f"Not cancelled = {active}; cost within limit = {affordable}; "
            f"override or sufficient credential = {qualified}. "
            f"All three are required; approval value = {value}."
        )
    elif rule == "missing_fair_approval":
        value = {"0": 1 - r["prior_true"], "1": r["prior_true"]}
        text = (
            "The flag is unobserved and no additional evidence is supplied. "
            f"The stated prior gives P(approval=1)={value['1']}, P(approval=0)={value['0']}."
        )
    else:
        raise ValueError("unsupported reference rule")
    if value != sample.metadata["oracle_facts"]["value"]:
        raise ValueError("visible-fact recomputation disagrees with heldout oracle")
    for question in sample.questions:
        if question.type in {"choice", "score"}:
            expected = {
                c.id: value.get(c.id, 0) if isinstance(value, dict) else float(int(c.id) == value)
                for c in question.candidates
            }
        else:
            asked = int(question.instruction.removeprefix("Is the result ").removesuffix("?"))
            truth = value.get(str(asked), 0) if isinstance(value, dict) else float(asked == value)
            expected = {"false": 1 - truth, "true": truth}
        if question.target_distribution != expected:
            raise ValueError("prepared targets disagree with visible-fact recomputation")
    return text


def prepare_cohort(path, per_rule=3):
    if type(per_rule) is not int or not 1 <= per_rule <= 3:
        raise ValueError("diagnostic permits one to three independent cases per rule")
    groups = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        sample = Sample.from_json(json.loads(line))
        m = sample.metadata
        if m.get("source") == "ayaka-independent-recovery-1" and m.get("language") == "en":
            if m.get("split") != "dev" or not m.get("evaluation_only"):
                raise ValueError("mechanism selection is restricted to evaluation-only dev")
            groups[m["semantic_rule"]].append(sample)
    if set(groups) != set(POLICIES["en"]):
        raise ValueError("all six heldout rules are required")
    records = []
    for rule in sorted(groups):
        ordered = sorted(
            groups[rule],
            key=lambda s: hashlib.sha256(
                f"{SEED}/{s.metadata['source_lineage']}".encode()
            ).digest(),
        )
        if len(ordered) < per_rule:
            raise ValueError("incomplete rule stratum")
        selected = ordered[:per_rule]
        notes = [reference_notes(s) for s in selected]
        for i, sample in enumerate(selected):
            records.append(
                {
                    "sample": sample.to_json(),
                    "oracle": notes[i],
                    "distractor": notes[(i + 1) % len(notes)]
                    if len(notes) > 1
                    else "No derivation.",
                    "distractor_source": selected[(i + 1) % len(selected)].metadata[
                        "source_example_id"
                    ]
                    if len(selected) > 1
                    else None,
                }
            )
    lineages = [r["sample"]["metadata"]["source_lineage"] for r in records]
    if len(lineages) != len(set(lineages)):
        raise ValueError("cohort must contain distinct independent cases")
    # Cases with identical results/traces can make distractors uninformative;
    # preserve this fact rather than choosing replacements after model scoring.
    return records


@torch.inference_mode()
def score_heads(model, batch, cache):
    """One encoding/learned head evaluation, with native LM and hybrid projections."""
    if not bool(batch.has_label.all()):
        raise ValueError("matched LM/head diagnostic requires label-supported candidate sets")
    old = model.cfg
    try:
        model.cfg = replace(old, readout="pointer")
        out = model(batch, past_key_values=cache, apply_temperature=False)
    finally:
        model.cfg = old
    bucket = length_bucket(
        batch.seq_len, model.cfg.long_prompt_tokens, batch.answer_pos.numel(), out.logits.device
    )
    gate = model.gate[batch.primitive[batch.cand_question], bucket[batch.cand_question]]
    logits = {
        "lm": out.label_logits,
        "pointer": out.pointer_logits,
        "hybrid": out.label_logits + gate * out.pointer_logits,
    }
    return {head: ragged_softmax(values, out.cand_cu).tolist() for head, values in logits.items()}


@torch.inference_mode()
def score_trace(generator, trace, spec):
    suffix = readout_suffix(generator.tok, spec, generator.model.cfg.max_label_candidates)
    item = EncodedQuestion(trace.input_ids + trace.token_ids, suffix, PRIMITIVE_INDEX[spec.type])
    batch = suffix_rows([item], len(item.prefix_ids), generator.tok.pad_id).to(
        generator.model.embed_weight().device
    )
    return score_heads(generator.model, batch, copy.deepcopy(trace.cache))


@torch.inference_mode()
def forced_trace(generator, state, spec, text):
    ids, payload = generator.prepare(generator.messages_for(state, spec))
    if payload is not None:
        raise ValueError("text-only intervention required")
    tokens = generator.tok.encode(text)
    eos = getattr(getattr(generator.tok, "hf", None), "eos_token_id", None)
    if eos is not None:
        tokens.append(eos)
    suffix = readout_suffix(generator.tok, spec, generator.model.cfg.max_label_candidates)
    if len(tokens) > 512 or len(ids) + len(tokens) + len(suffix.suffix_ids) > generator.max_context:
        raise ValueError("complete intervention does not fit; refuse truncation")
    _, cache = generator.prefill(
        torch.tensor([ids + tokens], device=generator.model.embed_weight().device), None
    )
    return Trace(
        text=text, input_ids=ids, token_ids=tokens, cache=cache, finish_reason="teacher_forced"
    )


def interaction_interval(cells, condition, replicates=2000):
    """Paired difference-in-differences: LM reasoning gain minus hybrid reasoning gain."""
    arms = [
        cells[f"{c}/{h}"]
        for c, h in [(condition, "lm"), ("empty", "lm"), (condition, "hybrid"), ("empty", "hybrid")]
    ]
    fields = ("id", "cluster_id", "type", "target", "ordinals")
    identities = [[tuple(r.get(k) for k in fields) for r in a] for a in arms]
    if not arms[0] or any(a != identities[0] for a in identities):
        raise ValueError("interaction arms must have identical question identities")
    clusters = defaultdict(list)
    for i, row in enumerate(arms[0]):
        clusters[row["cluster_id"]].append(i)
    groups = list(clusters.values())
    rng, values = random.Random(15), []

    def statistic(indices):
        scores = [summarize([a[i] for i in indices])["cc_equal_types"] for a in arms]
        return scores[0] - scores[1] - scores[2] + scores[3]

    for _ in range(replicates):
        values.append(statistic([i for _ in groups for i in rng.choice(groups)]))
    from .v2 import percentile

    return {
        "definition": "LM reasoning gain minus hybrid reasoning gain, both relative to empty closed context",
        "cc_difference_in_differences": statistic(list(range(len(arms[0])))),
        "ci95": [percentile(values, 0.025), percentile(values, 0.975)],
        "independent_cases": len(groups),
        "bootstrap_replicates": replicates,
        "seed": 15,
    }


@torch.inference_mode()
def evaluate_frozen(decision, records, budget=512, progress=None, deadline=None):
    if budget != 512:
        raise ValueError("use the predeclared 512-token diagnostic budget")
    model, tok = decision.model, decision.tok
    model.eval()
    generator = TraceGenerator(model, tok, apply_temperature=False)
    cells = {f"{condition}/{head}": [] for condition in CONDITIONS for head in HEADS}
    diagnostics = []
    for case_index, record in enumerate(records):
        sample = Sample.from_json(record["sample"])
        if record["oracle"] != reference_notes(sample):
            raise ValueError("oracle intervention differs from verified visible facts")
        for question in sample.questions:
            if deadline is not None and monotonic() >= deadline:
                raise TimeoutError("diagnostic deadline reached; cohort remains incomplete")
            q = _noul_canonical(question)
            spec = QuestionSpec(
                q.type,
                q.instruction,
                [c.description for c in q.candidates],
                [c.ordinal for c in q.candidates] if q.type == "score" else None,
            )
            target = [q.target_distribution.get(c.id, 0) for c in q.candidates]
            prefix, items = encode_decision(
                sample.state,
                [spec.view()],
                tok,
                decision.max_seq_len,
                model.cfg.max_label_candidates,
            )
            if (
                prefix != render_prefix(sample.state, tok)
                or len(prefix) + len(items[0].rendered.suffix_ids) > decision.max_seq_len
            ):
                raise ValueError("direct context would truncate; refuse an unmatched intervention")
            cache = decision.original._prefix_cache(prefix, model.embed_weight().device)
            batch = suffix_rows(items, len(prefix), tok.pad_id).to(model.embed_weight().device)
            probabilities = {"direct": score_heads(model, batch, copy.deepcopy(cache))}
            # Check the current production readout using the exact same prefix cache.
            output = model(batch, past_key_values=copy.deepcopy(cache), apply_temperature=False)
            parity = max(
                abs(a - b)
                for a, b in zip(
                    probabilities["direct"]["hybrid"],
                    ragged_softmax(output.logits, output.cand_cu).tolist(),
                    strict=True,
                )
            )
            if parity > 1e-6:
                raise ValueError("diagnostic hybrid output differs from production")
            del cache, output
            trace_info = {}
            for condition in CONDITIONS[1:]:
                start = monotonic()
                if condition == "generated":
                    suffix = readout_suffix(tok, spec, model.cfg.max_label_candidates)
                    trace = generator.generate_trace(
                        generator.messages_for(sample.state, spec),
                        budget,
                        reserve=len(suffix.suffix_ids),
                    )
                    if not trace.text.strip():
                        raise ValueError(
                            "empty generated trace; refuse to hide fallback in an ablation"
                        )
                else:
                    text = "" if condition == "empty" else record[condition]
                    trace = forced_trace(generator, sample.state, spec, text)
                probabilities[condition] = score_trace(generator, trace, spec)
                trace_info[condition] = {
                    "tokens": trace.generated_tokens,
                    "finish_reason": trace.finish_reason,
                    "seconds": monotonic() - start,
                    "trace_sha256": sha256(trace.text.encode()),
                    "text": trace.text,
                }
                del trace
            identity = sample.metadata["source_example_id"] + "/" + q.id
            for condition, heads in probabilities.items():
                for head, probs in heads.items():
                    row = typed_row(spec, probs, target)
                    row.update(
                        id=identity,
                        cluster_id=sample.metadata["source_lineage"],
                        language="en",
                        family=sample.metadata["task_family"],
                        ordinals=spec.ordinals,
                        probs=probs,
                        target=target,
                        condition=condition,
                        head=head,
                        reasoning_tokens=trace_info.get(condition, {}).get("tokens", 0)
                        if condition == "generated"
                        else 0,
                    )
                    cells[f"{condition}/{head}"].append(row)
            diagnostics.append(
                {
                    "id": identity,
                    "production_parity_max_abs": parity,
                    "trace": trace_info,
                    "distractor_identical_to_oracle": record["oracle"] == record["distractor"],
                }
            )
            if progress:
                progress(
                    {
                        "case": case_index + 1,
                        "question": q.id,
                        "total_cases": len(records),
                        "completed_questions": len(diagnostics),
                        "generated": trace_info["generated"],
                    }
                )
    return {
        "complete": True,
        "scope": "frozen English dev mechanism diagnosis; oracle/distractor are privileged interventions, not deployment performance",
        "budget": budget,
        "cohort_sha256": sha256(canonical(records)),
        "rows": cells,
        "summary": {key: summarize(rows) for key, rows in cells.items()},
        "paired": {
            f"{c}/{h}": paired_report(cells[f"empty/{h}"], cells[f"{c}/{h}"])
            for c in ("generated", "oracle", "distractor")
            for h in HEADS
        },
        "paired_direct": {
            f"{c}/{h}": paired_report(cells[f"direct/{h}"], cells[f"{c}/{h}"])
            for c in CONDITIONS[1:]
            for h in HEADS
        },
        "interaction": {c: interaction_interval(cells, c) for c in ("generated", "oracle")},
        "diagnostics": diagnostics,
        "optimizer_steps": 0,
        "calibration_fitted": False,
        "weights_selected": False,
        "official_composite": None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=3600)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.out.exists() or sha256(canonical(plan["records"])) != plan["cohort_sha256"]:
        raise ValueError("require a new output and an intact frozen cohort")
    args.out.mkdir(parents=True)
    deadline = monotonic() + args.seconds
    torch.set_num_threads(4)
    for name, path in [("parent", args.parent), ("pilot", args.pilot)]:
        identity = checkpoint_fingerprint(path)
        if identity != plan["checkpoints"][name]:
            raise ValueError("checkpoint differs from the frozen plan")
        model = load_checkpoint(str(path), device="cuda", dtype=torch.bfloat16, merge=False).eval()
        decision = controlled_decision(model, HFTokenizer.for_config(model.cfg))
        report = evaluate_frozen(
            decision,
            plan["records"],
            deadline=deadline,
            progress=lambda p, checkpoint=name: print(
                json.dumps({"checkpoint": checkpoint, **p}), flush=True
            ),
        )
        report["model_id"] = identity
        (args.out / (name + ".json")).write_bytes(canonical(report) + b"\n")
        if checkpoint_fingerprint(path) != identity:
            raise ValueError("frozen checkpoint changed during evaluation")
        del model, decision, report
        gc.collect()
        torch.cuda.empty_cache()
    (args.out / "complete.json").write_bytes(
        canonical(
            {
                "complete": True,
                "optimizer_steps": 0,
                "full_training_started": False,
                "test_evaluated": False,
            }
        )
        + b"\n"
    )


if __name__ == "__main__":
    main()
