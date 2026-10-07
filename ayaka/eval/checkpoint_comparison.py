"""Matched development evaluation of published v1 and trained v2 checkpoints.

The published v1 keeps its frozen gated worked-steps route. Trained v2 is read
through its saved serving input contract, with reasoning off or forced medium.
No model is trained, no temperature is fitted, and no private test is opened.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path

from ..data.schema import Sample
from ..evidence_pipeline import FROZEN_REASONING_POLICY
from ..reasoning import ReasoningSettings
from .matched_contract import BACKBONE_REVISION, CHECKPOINT_REPO, CHECKPOINT_REVISION
from .pretraining_v2 import cohort_fingerprint
from .quality_hierarchy import RULE, _canonical_questions, _comparison, _model_id
from .read_artifact import fingerprint
from .v2 import summarize, typed_row

VERSION = "ayaka-trained-checkpoint-comparison-dev-1"
SYSTEMS = ("v1_on", "v2_off", "v2_on")


def _json_ordinal(value):
    """Retain exact numeric Score levels when schema loading creates Decimals."""
    if not isinstance(value, Decimal):
        return value
    if value == value.to_integral_value():
        return int(value)
    number = float(value)
    if not math.isfinite(number) or Decimal(str(number)) != value:
        raise ValueError("Score ordinal cannot be represented exactly in comparison JSON")
    return number


def questions(samples):
    """Retain original candidate IDs and complete canonical development inputs."""
    expected = _canonical_questions(samples)
    prepared = []
    for sample in samples:
        for question in sample.questions:
            key = sample.metadata["source_example_id"] + "/" + question.id
            spec, fields = expected[key]
            ordinals = (
                [_json_ordinal(value) for value in spec.ordinals]
                if spec.ordinals is not None
                else None
            )
            fields = {**fields, "ordinals": ordinals}
            spec = replace(spec, candidate_ids=fields["candidate_ids"], ordinals=ordinals)
            binding = fingerprint({"state": sample.state, "question": asdict(spec), **fields})
            prepared.append(
                (
                    sample.state,
                    spec,
                    {
                        **fields,
                        "input_sha256": binding,
                        "state_sha256": fingerprint(sample.state),
                        "question_view_sha256": fingerprint([spec.view().__dict__]),
                    },
                )
            )
    return prepared


def make_protocol(samples, v1_model_id, v2_model_id):
    questions(samples)
    ids = {"v1": _model_id(v1_model_id), "v2": _model_id(v2_model_id)}
    if ids["v1"] == ids["v2"]:
        raise ValueError("v1 and v2 require distinct checkpoint identities")
    return {
        "version": VERSION,
        "split": "dev",
        "cohort_sha256": cohort_fingerprint(samples),
        "model_ids": ids,
        "backbone_revision": BACKBONE_REVISION,
        "v1": {
            "checkpoint_repo": CHECKPOINT_REPO,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "policy": asdict(FROZEN_REASONING_POLICY),
            "reasoner_adapter": "off",
            "max_new_tokens": 384,
            "content_token_accounting": "decoded worked-step tokens; final EOS excluded",
        },
        "v2": {
            "off": ReasoningSettings(mode="off").as_dict(),
            "on": ReasoningSettings(mode="on", effort="medium").as_dict(),
            "temperatures": "saved checkpoint temperatures; no evaluation fit",
            "reasoned_calibration": "not independently validated",
        },
        "scope": "matched development comparison; not an official JevBench rank",
        "promotable": False,
        "official_composite": None,
    }


def validate_protocol(protocol, samples):
    expected = make_protocol(samples, protocol["model_ids"]["v1"], protocol["model_ids"]["v2"])
    if fingerprint(expected) != fingerprint(protocol):
        raise ValueError("comparison protocol differs from its fixed development recipe")


def _checked_row(row, spec, fields, protocol, system):
    if any(row.get(name) != value for name, value in fields.items()):
        raise ValueError("comparison row differs from canonical evidence, targets or order")
    model_id = protocol["model_ids"]["v1" if system == "v1_on" else "v2"]
    if (
        row.get("protocol_sha256") != fingerprint(protocol)
        or row.get("model_id") != model_id
        or row.get("system") != system
    ):
        raise ValueError("comparison row belongs to a different checkpoint or protocol")
    tokens = row.get("reasoning_tokens")
    budget = 0 if system == "v2_off" else 384
    routes = {"baseline", "reasoned"} if system == "v1_on" else {"reasoned", "fallback"}
    if system == "v2_off":
        routes = {"direct"}
    if (
        type(tokens) is not int
        or not 0 <= tokens <= budget
        or type(row.get("budget")) is not int
        or row["budget"] != budget
        or row.get("route") not in routes
        or (row["route"] == "reasoned" and (tokens == 0 or row.get("error") is not None))
        or (
            system == "v2_on"
            and row["route"] == "reasoned"
            and row.get("finish_reason") not in {"eos", "length"}
        )
        or not isinstance(row.get("latency_s"), (float, int))
        or not math.isfinite(row["latency_s"])
        or row["latency_s"] < 0
    ):
        raise ValueError("invalid route, reasoning usage or latency")
    if system == "v1_on":
        contexts = row.get("checked_contexts", [])
        if not contexts or contexts[0].get("state_sha256") != fields["state_sha256"]:
            raise ValueError("v1 requires checked original-input context")
        for context in contexts:
            counts = context.get("input_tokens", [])
            if (
                context.get("context_limit") != 8192
                or context.get("questions_sha256") != fields["question_view_sha256"]
                or len(counts) != 1
                or type(counts[0]) is not int
                or not 0 < counts[0] <= 8192
            ):
                raise ValueError("v1 context was clipped or used a different question")
    return {**row, **typed_row(spec, row["probs"], fields["target"])}


def checked_rows(rows, samples, protocol, system, *, complete=True):
    validate_protocol(protocol, samples)
    if system not in SYSTEMS:
        raise ValueError("unknown comparison system")
    expected = {fields["id"]: (spec, fields) for _, spec, fields in questions(samples)}
    seen, checked = set(), []
    for row in rows:
        key = row.get("id")
        if key in seen or key not in expected:
            raise ValueError("duplicate or unknown comparison question")
        spec, fields = expected[key]
        checked.append(_checked_row(row, spec, fields, protocol, system))
        seen.add(key)
    if complete and seen != set(expected):
        raise ValueError("each comparison system must cover every development question")
    return sorted(checked, key=lambda row: row["id"])


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def collect_rows(predict, samples, protocol, system, output):
    """Resume only matching rows and flush each complete observation."""
    output = Path(output)
    existing = read_rows(output) if output.exists() else []
    checked = checked_rows(existing, samples, protocol, system, complete=False)
    done = {row["id"] for row in checked}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for state, spec, fields in questions(samples):
            if fields["id"] in done:
                continue
            start = time.perf_counter()
            observed = predict(state, spec)
            row = {
                **observed,
                **fields,
                "system": system,
                "model_id": protocol["model_ids"]["v1" if system == "v1_on" else "v2"],
                "protocol_sha256": fingerprint(protocol),
                "budget": 0 if system == "v2_off" else 384,
                "latency_s": time.perf_counter() - start,
            }
            _checked_row(row, spec, fields, protocol, system)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            done.add(fields["id"])
    return checked_rows(read_rows(output), samples, protocol, system)


def compare(rows, samples, protocol, *, replicates=2000):
    if type(replicates) is not int or replicates < 1:
        raise ValueError("bootstrap repetitions must be positive")
    systems = {name: checked_rows(rows[name], samples, protocol, name) for name in SYSTEMS}
    comparisons = {
        "v2_off_over_v1_on": _comparison(
            systems["v1_on"], systems["v2_off"], major_gain=True, replicates=replicates
        ),
        "v2_on_over_v2_off": _comparison(
            systems["v2_off"], systems["v2_on"], major_gain=False, replicates=replicates
        ),
    }
    completed_reasoning = sum(row["route"] == "reasoned" for row in systems["v2_on"])
    return {
        "version": VERSION,
        "protocol": protocol,
        "rule": dict(RULE),
        "systems": {name: summarize(value) for name, value in systems.items()},
        "comparisons": comparisons,
        "routes": {
            name: {
                route: sum(row["route"] == route for row in value)
                for route in sorted({r["route"] for r in value})
            }
            for name, value in systems.items()
        },
        "screen_passed": completed_reasoning > 0
        and all(c["screen_passed"] for c in comparisons.values()),
        "v2_reasoning_completed": completed_reasoning,
        "official_composite": None,
        "promotable": False,
        "independent_test_required": True,
        "scope": protocol["scope"],
    }


class ContentTokenCounter:
    """Observe the unchanged v1 generator's final decoded token list."""

    def __init__(self, inner):
        self.inner, self.tokens = inner, 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def decode(self, ids):
        self.tokens = len(ids)
        return self.inner.decode(ids)


def predictor(model, tok, system):
    if system == "v1_on":
        from ..evidence_pipeline import reasoning_decision
        from .matched_execution import FullContextDecision

        decision = reasoning_decision(model, tok, 8192)
        checked = FullContextDecision(decision.original)
        decision.original = checked
        counter = ContentTokenCounter(decision.extractor.tok)
        decision.extractor.tok = counter

        def predict(state, spec):
            checked.contexts.clear()
            counter.tokens = 0
            result = decision.decide(state, [spec])[0]
            evidence = result.extras["evidence"]
            return {
                "probs": result.probs,
                "route": evidence["route"],
                "error": evidence["error"],
                "reasoning_tokens": counter.tokens,
                "checked_contexts": list(checked.contexts),
            }

        return predict
    from ..reasoning_pipeline import controlled_decision

    decision = controlled_decision(model, tok, 8192)
    setting = ReasoningSettings(mode="off" if system == "v2_off" else "on", effort="medium")

    def predict(state, spec):
        result = decision.decide(state, [spec], reasoning=[setting])[0]
        reasoning = result.extras["reasoning"]
        return {
            "probs": result.probs,
            "route": reasoning["route"],
            "error": reasoning["error"],
            "reasoning_tokens": reasoning["generated_tokens"],
            "finish_reason": reasoning["finish_reason"],
            "input_tokens": reasoning["input_tokens"],
        }

    return predict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "compare"))
    for name in ("samples", "protocol", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--system", choices=SYSTEMS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seconds", type=float, default=14400)
    parser.add_argument("--replicates", type=int, default=2000)
    for system in SYSTEMS:
        parser.add_argument(f"--{system.replace('_', '-')}", type=Path)
    args = parser.parse_args(argv)
    samples = [
        Sample.from_json(json.loads(line))
        for line in args.samples.read_bytes().splitlines()
        if line.strip()
    ]
    protocol = json.loads(args.protocol.read_bytes())
    validate_protocol(protocol, samples)
    if args.action == "compare":
        if args.out.exists():
            raise ValueError("comparison summaries must use a new output file")
        report = compare(
            {name: read_rows(getattr(args, name)) for name in SYSTEMS},
            samples,
            protocol,
            replicates=args.replicates,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return
    if args.system is None or args.checkpoint is None or args.backbone is None:
        raise ValueError("collection requires system, checkpoint and local backbone paths")
    destination = args.out.resolve()
    if (
        destination in {args.samples.resolve(), args.protocol.resolve()}
        or destination.is_relative_to(args.checkpoint.resolve())
        or destination.is_relative_to(args.backbone.resolve())
    ):
        raise ValueError("comparison output must not modify checkpoint or input files")
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 14400:
        raise ValueError("collection requires a positive process deadline up to four hours")
    from ..training.scoped_calibration import checkpoint_fingerprint

    wanted = protocol["model_ids"]["v1" if args.system == "v1_on" else "v2"]
    if checkpoint_fingerprint(args.checkpoint) != wanted:
        raise ValueError("checkpoint differs from its external identity")
    if args.system == "v1_on":
        from .matched_execution import checkpoint_config

        checkpoint_config(args.checkpoint)
    import torch
    from transformers import AutoTokenizer

    from ..checkpoint import load_checkpoint
    from ..tokenization import HFTokenizer
    from ..training.run_v2 import job_deadline

    with job_deadline(args.max_seconds):
        model = load_checkpoint(
            str(args.checkpoint),
            device=args.device,
            dtype=torch.bfloat16,
            merge=False,
            backbone_path=str(args.backbone),
            local_files_only=True,
            strict_loading=True,
        )
        if model.cfg.backbone_revision != BACKBONE_REVISION:
            raise ValueError("checkpoint uses a different native backbone revision")
        tok = HFTokenizer(
            AutoTokenizer.from_pretrained(str(args.backbone), local_files_only=True),
            model.cfg.backbone,
        )
        rows = collect_rows(
            predictor(model, tok, args.system), samples, protocol, args.system, args.out
        )
        if checkpoint_fingerprint(args.checkpoint) != wanted:
            raise ValueError("checkpoint files changed during collection")
    print(
        json.dumps(
            {
                "system": args.system,
                "questions": len(rows),
                "complete": True,
                "optimizer_updates": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
