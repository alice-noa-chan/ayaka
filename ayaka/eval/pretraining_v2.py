"""Separate native-image, paired-text and candidate diagnostics before promotion."""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path

from ..data.candidate_v2 import finite_partition_audit
from ..primitives import QuestionSpec
from ..reasoning import ReasoningSettings
from ..training.batching import _noul_canonical
from ..training.prepare_v2 import canonical, sha256, validate_bundle
from .v2 import paired_report, summarize, typed_row


def evaluate_tracks(decision, samples, modes=("off", "low", "medium", "high"), *, deadline=None):
    settings = {
        mode: ReasoningSettings(
            mode="off" if mode == "off" else ("auto" if mode == "auto" else "on"),
            effort=mode if mode in {"low", "medium", "high"} else "medium",
        )
        for mode in modes
    }
    if (
        not modes
        or len(set(modes)) != len(modes)
        or set(modes) - {"off", "auto", "low", "medium", "high"}
    ):
        raise ValueError("evaluation modes must be unique off/auto/low/medium/high values")
    rows, complete, seen = {mode: [] for mode in modes}, True, set()
    for sample in samples:
        metadata = sample.metadata
        state = (
            decision.prepare_media(sample.state, metadata["media"])
            if "media" in metadata
            else sample.state
        )
        for question in sample.questions:
            q = _noul_canonical(question)
            identity = metadata["source_example_id"] + "/" + q.id
            if identity in seen:
                raise ValueError("evaluation requires unique question ids")
            seen.add(identity)
            spec = QuestionSpec(
                q.type,
                q.instruction,
                [c.description for c in q.candidates],
                [c.ordinal for c in q.candidates] if q.type == "score" else None,
            )
            target = [q.target_distribution.get(c.id, 0) for c in q.candidates]
            for mode, setting in settings.items():
                if deadline is not None and time.monotonic() >= deadline:
                    complete = False
                    break
                start = time.perf_counter()
                result = decision.decide(state, [spec], reasoning=[setting])[0]
                diagnostic = result.extras.get("reasoning", {})
                row = typed_row(spec, result.probs, target)
                if mode == "off":
                    from ..routing import routing_features

                    row["routing_features"] = routing_features(
                        sample.state, spec, result, decision.tok, setting.budget
                    )
                row.update(
                    id=identity,
                    cluster_id=metadata["source_lineage"],
                    pair_id=metadata["source_lineage"] + "/" + metadata["language"] + "/" + q.id,
                    split=metadata["split"],
                    language=metadata["language"],
                    modality=metadata.get("modality", "text"),
                    family=metadata["task_family"],
                    partition="generated_finite" if "proposal_supervision" in metadata else "fixed",
                    probs=result.probs,
                    target=target,
                    budget=setting.budget,
                    route=diagnostic.get("route", "direct"),
                    generated_tokens=diagnostic.get("generated_tokens", 0),
                    reasoning_tokens=diagnostic.get("generated_tokens", 0),
                    input_tokens=diagnostic.get("input_tokens", 0),
                    latency_s=time.perf_counter() - start,
                    finish_reason=diagnostic.get("finish_reason", "direct"),
                )
                if mode == "off" and row["generated_tokens"]:
                    raise ValueError("off evaluation unexpectedly generated tokens")
                rows[mode].append(row)
            if not complete:
                break
        if not complete:
            break
    report = {
        "complete": complete,
        "rows": rows,
        "official_composite": None,
        "scope": "repository-authored pretraining tracks; not JevBench sealed-inclusive",
    }
    report["summary"] = {mode: summarize(group) for mode, group in rows.items() if group}
    report["groups"] = {}
    report["image_text_pairs"] = {}
    for mode, group in rows.items():
        strata, pairs = defaultdict(list), defaultdict(dict)
        for row in group:
            strata[
                f"{row['modality']}/{row['partition']}/{row['language']}/{row['family']}"
            ].append(row)
            pairs[row["pair_id"]][row["modality"]] = row
        report["groups"][mode] = {key: summarize(value) for key, value in strata.items()}
        joined = [pair for pair in pairs.values() if set(pair) == {"image", "text"}]
        report["image_text_pairs"][mode] = {
            "n": len(joined),
            "mean_image_minus_text_nll": sum(p["image"]["nll"] - p["text"]["nll"] for p in joined)
            / len(joined)
            if joined
            else None,
            "image_wrong_text_correct": sum(
                p["image"]["correct"] == 0 and p["text"]["correct"] == 1 for p in joined
            ),
            "both_wrong": sum(
                p["image"]["correct"] == 0 and p["text"]["correct"] == 0 for p in joined
            ),
            "interpretation": "paired diagnostics; not causal proof of perception versus reasoning errors",
        }
    report["reasoning_changes"] = {
        mode: paired_report(
            [row for row in rows["off"] if row["id"] in {r["id"] for r in group}],
            [row for row in group if row["id"] in {r["id"] for r in rows["off"]}],
        )
        for mode, group in rows.items()
        if mode != "off" and "off" in rows and group
    }
    return report


def evaluate_proposals(service, samples, *, deadline=None):
    """Exact semantic audit only for the declared status grammar, never arbitrary prose."""
    rows = []
    for sample in samples:
        annotation = sample.metadata.get("proposal_supervision")
        if not annotation:
            continue
        if deadline is not None and time.monotonic() >= deadline:
            return {"complete": False, "rows": rows}
        output = service.handle({"state": sample.state, "questions": {"q": annotation["question"]}})
        metadata = output["candidate_generation"]["q"]
        row = {
            "id": sample.metadata["source_example_id"],
            "split": sample.metadata["split"],
            "status": metadata["status"],
            "usage": output["usage"],
            "semantics": "unverified",
        }
        memberships, domain = {}, annotation["universe"]
        for proposal in metadata.get("proposals", []):
            matches = [
                value
                for value in domain
                if proposal["description"].strip() in {value, "Status " + value}
            ]
            if len(matches) != 1:
                break
            memberships[proposal["id"]] = matches
        else:
            if memberships and metadata["status"] == "completed":
                row.update(
                    semantics="exact_declared_status_grammar",
                    audit=finite_partition_audit(domain, memberships, annotation["parent"]),
                )
        rows.append(row)
    return {
        "complete": True,
        "rows": rows,
        "scope": "bounded finite-domain proposal evaluation; free-form descriptions remain unverified",
    }


def main(argv=None):
    import torch

    from ..multimodal import load_image_decision
    from ..serve import DecisionService
    from ..training.run_v2 import job_deadline, source_matches
    from ..training.scoped_calibration import checkpoint_fingerprint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--split", choices=("router_train", "dev", "calibration", "test"), default="dev"
    )
    parser.add_argument("--modes", nargs="+", default=["off", "low", "medium", "high"])
    parser.add_argument("--proposals", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-evaluation-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.max_evaluation_seconds)
        or not 0 < args.max_evaluation_seconds <= 28800
        or Path(args.out).exists()
    ):
        raise ValueError("evaluation needs a positive deadline and a new output file")
    manifest, splits = validate_bundle(args.bundle)
    if not source_matches(manifest):
        raise ValueError("source files changed after bundle preparation; prepare a new bundle")
    # The hard cap includes pretrained loading and an in-flight generation, not
    # only the gaps between questions. An interrupted report cannot be promoted.
    with job_deadline(args.max_evaluation_seconds):
        deadline = time.monotonic() + args.max_evaluation_seconds
        decision = load_image_decision(args.checkpoint, device=args.device, dtype=torch.bfloat16)
        report = evaluate_tracks(decision, splits[args.split], args.modes, deadline=deadline)
        model_id = checkpoint_fingerprint(args.checkpoint)
        for rows in report["rows"].values():
            for row in rows:
                row["model_id"] = model_id
        if args.proposals:
            report["proposals"] = evaluate_proposals(
                DecisionService(decision, "ayaka-v2"), splits[args.split], deadline=deadline
            )
        report.update(
            split=args.split,
            bundle_manifest_sha256=sha256((Path(args.bundle) / "manifest.json").read_bytes()),
            source_code_revision=manifest["code_revision"],
            calibration="not_fitted",
            router="not_promoted",
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_bytes(canonical(report) + b"\n")
    return report


if __name__ == "__main__":
    main()
