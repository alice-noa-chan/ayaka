"""Evidence-route evaluation: one GPU pass per split, policy selection offline.

``collect`` runs every question once through the plain decision readout and,
where the calculation gate passes, once through plan generation, validation
and the quotes / executed readouts. Rows keep every variant, so the policy grid
(readout, fusion weight, computed-predicate confidence, baseline cutoff) is
scored afterwards without another GPU pass. Gold labels are used for scoring
only; the model sees state, instruction and option descriptions.

    python -m ayaka.eval.evidence_eval collect --ckpt CKPT --split public --out public.json
    python -m ayaka.eval.evidence_eval collect --ckpt CKPT --split jsonl --data dev.jsonl --out dev.json
    python -m ayaka.eval.evidence_eval collect --ckpt CKPT --split jev_open_test --limit 250 --out openjev.json
    python -m ayaka.eval.evidence_eval select --dev dev.json openjev.json --apply public.json test.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time

from ..evidence import (
    EvidenceError,
    augmented_state,
    needs_calculation,
    reasoned_state,
    reasoning_messages,
)
from ..evidence_ids import id_messages, parse_id_plan, recover_id_evidence, validate_id_plan
from ..evidence_pipeline import decision_request
from ..evidence_policy import EvidencePolicy, fuse_probabilities
from .jevbench import TIERS, _ece, data_dir, load_jsonl, record_to_item, speed_axis

# ---------------------------------------------------------------- records


def sample_to_records(sample, prefix: str) -> list[dict]:
    """Canonical Samples -> JevBench-style records (gold kept for scoring)."""
    records = []
    for q in sample.questions:
        gold = max(q.target_distribution, key=q.target_distribution.get)
        if q.type == "noul":
            by_id = {c.id: c.description for c in q.candidates}
            labels = ["false", "true"]
            criteria = {"false": by_id["false"], "true": by_id["true"]}
        elif q.type == "score":
            ordered = sorted(q.candidates, key=lambda c: c.ordinal)
            labels = [str(c.ordinal) for c in ordered]
            criteria = [c.description for c in ordered]
            gold = str(next(c.ordinal for c in ordered if c.id == gold))
        else:
            labels = [c.id for c in q.candidates]
            criteria = {c.id: c.description for c in q.candidates}
        records.append(
            {
                "id": f"{prefix}-{sample.metadata.get('source_example_id', '')}-{q.id}",
                "state": sample.state,
                "question": {"type": q.type, "instructions": q.instruction, "criteria": criteria},
                "labels": labels,
                "expected": gold,
                "family": sample.metadata.get("task_family", "unknown"),
            }
        )
    return records


def load_records(split: str, data: str = "", limit: int = 0, seed: int = 0) -> list[dict]:
    if split == "public":
        records = [
            dict(r, tier=t)
            for t in TIERS
            for r in load_jsonl(os.path.join(data or data_dir(), f"{t}.jsonl"))
        ]
    elif split == "jsonl":
        records = [dict(r, tier=r.get("tier") or "jsonl") for r in load_jsonl(data)]
    else:
        from ..data.decontam import Decontaminator
        from ..data.loaders import load_spec_samples

        samples, _ = load_spec_samples(split, limit=limit or None, dedup=False, seed=seed)
        samples, _ = Decontaminator.from_jevbench().filter(samples)
        records = [dict(r, tier=split) for s in samples for r in sample_to_records(s, split)]
    return records[: limit or None]


# ---------------------------------------------------------------- collection


def _probs(result, labels):
    return dict(zip(labels, result.probs, strict=True))


def collect(
    decision,
    generator,
    records,
    batch_size: int = 8,
    device=None,
    log=print,
    reasoner=None,
) -> list:
    """Baseline + (gated) evidence variants for every record.

    ``generator`` (ID plans) and ``reasoner`` (free-form worked steps) are each
    optional; a missing one skips its variants.
    """

    rows, pending = [], []
    for rec in records:
        item = record_to_item(rec)
        tic = time.perf_counter()
        base = decision.decide(item.state, [item.spec], device=device)[0]
        row = {
            "id": rec.get("id", ""),
            "tier": rec.get("tier", ""),
            "family": rec.get("family", "unknown"),
            "type": item.spec.type,
            "labels": item.labels,
            "expected": item.expected,
            "baseline": _probs(base, item.labels),
            "baseline_s": time.perf_counter() - tic,
            "gate": False,
        }
        request = decision_request(item.state, item.spec)
        if needs_calculation(request):
            row["gate"] = True
            pending.append((row, item, request))
        rows.append(row)
    log(f"[evidence] {len(rows)} questions, {len(pending)} pass the calculation gate")

    pending.sort(key=lambda x: len(str(x[2]["state"])))
    if generator is not None:
        _batched(pending, generator, id_messages, batch_size, log, "extracted", decision, device)
    if reasoner is not None:
        _batched(
            pending, reasoner, reasoning_messages, batch_size, log, "reasoned", decision, device
        )
    return rows


def _batched(pending, gen, messages, batch_size, log, what, decision, device):
    import torch

    finish = _finish if what == "extracted" else _finish_reasoning

    def run(batch):
        tic = time.perf_counter()
        texts = gen.generate([messages(req) for _, _, req in batch])
        per = (time.perf_counter() - tic) / len(batch)
        return [(t, per) for t in texts]

    i, size = 0, batch_size
    while i < len(pending):
        batch = pending[i : i + size]
        try:
            outs = run(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if size > 1:
                size //= 2
                log(f"[evidence] OOM -> extraction batch {size}")
                continue
            outs = [("", 0.0)]
            batch[0][0]["error"] = "OOM"
        except EvidenceError as exc:  # context too long for every member
            outs = [("", 0.0)] * len(batch)
            for row, _, _ in batch:
                row["error"] = str(exc)
        for (row, item, request), (raw, secs) in zip(batch, outs, strict=True):
            finish(decision, row, item, request, raw, secs, device)
        i += len(batch)
        if i % 40 < len(batch):
            log(f"[evidence] {what} {i}/{len(pending)}")


def _finish_reasoning(decision, row, item, request, raw, secs, device):
    row.update(reasoning=raw[:4000], reasoning_s=secs)
    if not raw.strip():
        return
    tic = time.perf_counter()
    state = reasoned_state(item.state, raw)
    row["reasoned"] = _probs(decision.decide(state, [item.spec], device=device)[0], item.labels)
    row["reasoned_readout_s"] = time.perf_counter() - tic


def _finish(decision, row, item, request, raw, secs, device):
    row.update(raw_plan=raw[:4000], extraction_s=secs, verified=False, recovered=False)
    verified = None
    if raw:
        try:
            verified = validate_id_plan(item.state, parse_id_plan(raw))
        except (EvidenceError, ValueError) as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            try:
                verified = recover_id_evidence(item.state, raw)
                row["recovered"] = True
            except EvidenceError:
                verified = None
    if verified is None:
        return
    row["verified"] = True
    row["calculations"] = verified["calculations"]
    if item.spec.type == "noul":
        value = verified["calculations"].get("question_holds", {}).get("result")
        if isinstance(value, bool):
            row["computed"] = item.labels[1 if value else 0]
    tic = time.perf_counter()
    for name, calcs in (("quotes", False), ("executed", True)):
        state = augmented_state(item.state, verified, calculations=calcs)
        row[name] = _probs(decision.decide(state, [item.spec], device=device)[0], item.labels)
    row["readout_s"] = (time.perf_counter() - tic) / 2


# ---------------------------------------------------------------- policies


def policy_grid():
    for readout, weight, conf, cutoff in itertools.product(
        ("quotes", "executed"), (0.5, 1.0), (None, 0.8, 0.95), (0.8, 0.9, 0.99, 1.0)
    ):
        yield EvidencePolicy(
            readout=readout,
            weight=weight,
            boolean_confidence=conf,
            baseline_cutoff=cutoff,
            recover_ids=True,
            gate="calculation",
        )
    for weight, cutoff in itertools.product((0.5, 1.0), (0.8, 0.9, 0.99, 1.0)):
        yield EvidencePolicy(
            readout="reasoned", weight=weight, baseline_cutoff=cutoff, gate="calculation"
        )
    yield EvidencePolicy(readout="baseline")


def available(rows, policy) -> bool:
    """A policy is scored only on rows that were collected with its variant."""
    key = {"reasoned": "reasoning", "quotes": "raw_plan", "executed": "raw_plan"}
    need = key.get(policy.readout)
    return need is None or all(need in r for r in rows if r["gate"])


def apply_policy(row, policy: EvidencePolicy):
    """-> (probabilities, routed, latency_s) for one stored row."""
    base = row["baseline"]
    routed = (
        policy.readout != "baseline"
        and row["gate"]
        and max(base.values()) <= policy.baseline_cutoff
    )
    latency = row["baseline_s"]
    if not routed:
        return base, False, latency
    if policy.readout == "reasoned":
        latency += row.get("reasoning_s", 0.0) + row.get("reasoned_readout_s", 0.0)
        aux = row.get("reasoned")
        probs, _ = fuse_probabilities(base, aux or base, policy, usable=aux is not None)
        return probs, True, latency
    latency += row.get("extraction_s", 0.0)
    usable = row.get("verified", False)
    if usable and (policy.recover_ids or not row.get("recovered")):
        latency += row.get("readout_s", 0.0)
        aux = row[policy.readout]
    else:
        usable, aux = False, base
    probs, _ = fuse_probabilities(
        base, aux, policy, computed_label=row.get("computed"), usable=usable
    )
    return probs, True, latency


def score(rows, policy) -> dict:
    correct, nll, confs, oks, lat, routed = 0, 0.0, [], [], [], 0
    for row in rows:
        probs, was_routed, secs = apply_policy(row, policy)
        pred = max(probs, key=probs.get)
        ok = pred == row["expected"]
        correct += ok
        nll -= math.log(max(probs[row["expected"]], 1e-12))
        confs.append(probs[pred])
        oks.append(ok)
        lat.append(secs)
        routed += was_routed
    n = max(len(rows), 1)
    ordered = sorted(lat)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / n,
        "nll": nll / n,
        "ece": _ece(confs, oks),
        "routed": routed,
        "latency_p50_s": ordered[len(ordered) // 2] if ordered else 0.0,
        "latency_p95_s": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        if ordered
        else 0.0,
    }


def select(dev_rows) -> tuple[EvidencePolicy, list]:
    """Most dev correct, then lower NLL, then fewer routed questions."""
    table = [(p, score(dev_rows, p)) for p in policy_grid() if available(dev_rows, p)]
    table.sort(key=lambda x: (-x[1]["correct"], x[1]["nll"], x[1]["routed"]))
    return table[0][0], table


def report(rows, policy) -> dict:
    out = {}
    groups = {}
    for row in rows:
        groups.setdefault(row["tier"], []).append(row)
    for tier, members in sorted(groups.items()):
        base, chosen = score(members, EvidencePolicy(readout="baseline")), score(members, policy)
        fams = {}
        for row in members:
            probs, _, _ = apply_policy(row, policy)
            b = max(row["baseline"], key=row["baseline"].get) == row["expected"]
            p = max(probs, key=probs.get) == row["expected"]
            f = fams.setdefault(row["family"], [0, 0, 0])
            f[0] += b
            f[1] += p
            f[2] += 1
        gated = [r for r in members if r["gate"]]
        out[tier] = {
            "baseline": base,
            "policy": chosen,
            "gated": len(gated),
            "valid_plans": sum(bool(r.get("verified") and not r.get("recovered")) for r in gated),
            "recovered_plans": sum(bool(r.get("recovered")) for r in gated),
            "families": {k: f"{b}->{p}/{n}" for k, (b, p, n) in sorted(fams.items())},
            "speed_axis_policy": speed_axis(chosen["latency_p50_s"], chosen["latency_p95_s"]),
        }
    return out


# ---------------------------------------------------------------- CLI


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ayaka.eval.evidence_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--split", required=True, help="public | jsonl | <dataset spec>")
    c.add_argument("--data", default="")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--max-new-tokens", type=int, default=768)
    c.add_argument("--variants", default="plans", help="comma list of: plans (ID plans), reasoning")
    c.add_argument("--reason-tokens", type=int, default=384)
    c.add_argument(
        "--reasoner-adapter",
        default="off",
        choices=["off", "on"],
        help="worked steps from the base model (off) or with the decision LoRA (on)",
    )
    c.add_argument("--out", required=True)
    s = sub.add_parser("select")
    s.add_argument("--dev", nargs="+", required=True)
    s.add_argument("--extra", nargs="*", default=[], help="more variants for the same rows")
    s.add_argument("--apply", nargs="*", default=[])
    s.add_argument("--out", default="")
    args = ap.parse_args(argv)

    if args.cmd == "collect":
        import torch

        from ..checkpoint import load_checkpoint
        from ..evidence_generation import PlanGenerator
        from ..primitives import Decision
        from ..tokenization import HFTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        tic = time.perf_counter()
        model = load_checkpoint(args.ckpt, device=device, dtype=dtype, merge=False).eval()
        tok = HFTokenizer.for_config(model.cfg)
        print(f"[evidence] model loaded in {time.perf_counter() - tic:.0f}s", flush=True)
        records = load_records(args.split, args.data, args.limit)
        variants = set(args.variants.split(","))
        with torch.inference_mode():
            rows = collect(
                Decision(model, tok),
                PlanGenerator(model, tok, max_new_tokens=args.max_new_tokens)
                if "plans" in variants
                else None,
                records,
                batch_size=args.batch_size,
                log=lambda m: print(m, flush=True),
                reasoner=PlanGenerator(
                    model,
                    tok,
                    max_new_tokens=args.reason_tokens,
                    stop_when=None,
                    adapter=args.reasoner_adapter,
                )
                if "reasoning" in variants
                else None,
            )
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"split": args.split, "data": args.data, "rows": rows}, f, ensure_ascii=False)
        return rows

    extra = {}
    for p in args.extra:
        with open(p, encoding="utf-8") as f:
            for r in json.load(f)["rows"]:
                extra[(r["tier"], r["id"])] = r

    def rows_of(paths):
        out = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                for r in json.load(f)["rows"]:
                    more = extra.get((r["tier"], r["id"]), {})
                    out.append({**more, **r} if more else r)
        return out

    dev = rows_of(args.dev)
    policy, table = select(dev)
    result = {
        "selected": policy.to_dict(),
        "dev": report(dev, policy),
        "dev_top5": [(p.to_dict(), m) for p, m in table[:5]],
        "applied": {p: report(rows_of([p]), policy) for p in args.apply},
    }
    print(json.dumps(result, indent=1, ensure_ascii=False)[:6000])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, ensure_ascii=False)
    return result


if __name__ == "__main__":
    main()
