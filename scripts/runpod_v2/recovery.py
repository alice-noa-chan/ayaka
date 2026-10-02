"""One predeclared, fixed continuation pilot and matched recovery measurements."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import torch

from ayaka.data.schema import Sample
from ayaka.eval.pretraining_v2 import evaluate_tracks
from ayaka.eval.recovery_v2 import (
    evaluate_trace_diagnostics,
    fit_report_calibrations,
    promotion_screen,
    recalibrate_report,
)
from ayaka.training.prepare_v2 import canonical, validate_bundle
from ayaka.training.scoped_calibration import checkpoint_fingerprint


def selection(samples, per_type=64, rich_cases=64):
    """Select before inference, keeping translations grouped in rich diagnostics."""
    rng, selected, used = random.Random(20261002), [], set()
    for kind in ("choice", "noul", "score"):
        for language, fraction in (("en", 0.625), ("ko", 0.1875), ("ja", 0.1875)):
            pool = [
                (s, q)
                for s in samples
                if s.metadata["language"] == language
                and not s.metadata["source_example_id"].startswith("recovery-1/")
                for q in s.questions
                if q.type == kind
            ]
            rng.shuffle(pool)
            count, required = 0, int(per_type * fraction)
            for sample, question in pool:
                lineage = sample.metadata["source_lineage"]
                if lineage in used:
                    continue
                raw = sample.to_json()
                raw["questions"] = [q for q in raw["questions"] if q["id"] == question.id]
                selected.append(Sample.from_json(raw))
                used.add(lineage)
                count += 1
                if count == required:
                    break
            if count != required:
                raise ValueError("insufficient independent selection stratum")
    selected.extend(
        s
        for s in samples
        if s.metadata["source_example_id"].startswith("recovery-1/")
        and s.metadata["oracle_facts"]["index"] < rich_cases
    )
    return selected


def raw_temperatures(decision):
    decision.text.original.apply_temperature = False
    decision.text.generator.apply_temperature = False
    decision.images.original.apply_temperature = False
    decision.images.generator.apply_temperature = False


class ForcedDecision:
    def __init__(self, decision, effort):
        self.decision, self.effort = decision, effort

    def decide(self, state, questions, device=None):
        from ayaka.reasoning import ReasoningSettings

        setting = ReasoningSettings(
            mode="off" if self.effort == "off" else "on",
            effort=self.effort if self.effort != "off" else "medium",
        )
        return self.decision.decide(
            state, questions, device=device, reasoning=[setting] * len(questions)
        )


def public_report(decision, effort):
    from ayaka.eval.jevbench import run_jevbench

    report = run_jevbench(
        decision.model, decision.tok, verbose=False, decision=ForcedDecision(decision, effort)
    )
    report.update(
        complete=True,
        scope="bundled historical public proxy; no sealed score",
        mode=effort,
        official_composite=None,
    )
    return report


def measure(checkpoint, name, dev, calibration, out, *, public_modes=("off",), diagnostics=True):
    from ayaka.multimodal import load_image_decision

    start = time.monotonic()
    print(json.dumps({"stage": "measure", "checkpoint": name}), flush=True)
    decision = load_image_decision(str(checkpoint), device="cuda", dtype=torch.bfloat16)
    identity = checkpoint_fingerprint(checkpoint)

    def save(filename, report, split=None):
        report.update(model_id=identity, weights_selected=False)
        if split:
            report["split"] = split
        for rows in report.get("rows", {}).values():
            for row in rows:
                row["model_id"] = identity
        (out / f"{name}-{filename}.json").write_bytes(canonical(report) + b"\n")
        return report

    # Preserve the released v1 temperatures as a separately labelled baseline.
    if name == "v1":
        save("published-dev", evaluate_tracks(decision, dev, ("off",)), "dev")
        save("published-public", public_report(decision, "off"), "public")
        from transformers import AutoTokenizer

        from ayaka.checkpoint import load_checkpoint
        from ayaka.primitives import Decision
        from ayaka.tokenization import HFTokenizer

        legacy = load_checkpoint(str(checkpoint), device="cuda", dtype=torch.bfloat16, merge=False)
        tokenizer = HFTokenizer(
            AutoTokenizer.from_pretrained(
                legacy.cfg.backbone, revision=legacy.cfg.backbone_revision
            ),
            legacy.cfg.backbone,
        )
        text = Decision(legacy, tokenizer)
        parity_samples = [s for s in dev if "media" not in s.metadata][:18]
        differences = []
        for sample in parity_samples:
            from ayaka.primitives import QuestionSpec
            from ayaka.reasoning import ReasoningSettings
            from ayaka.training.batching import _noul_canonical

            for question in sample.questions:
                q = _noul_canonical(question)
                spec = QuestionSpec(
                    q.type,
                    q.instruction,
                    [c.description for c in q.candidates],
                    [c.ordinal for c in q.candidates] if q.type == "score" else None,
                )
                a = text.decide(sample.state, [spec])[0].probs
                b = decision.decide(
                    sample.state, [spec], reasoning=[ReasoningSettings(mode="off")]
                )[0].probs
                differences.append(max(abs(x - y) for x, y in zip(a, b, strict=True)))
        parity = {
            "questions": len(differences),
            "max_probability_delta": max(differences),
            "tolerance": 1e-5,
            "passed": max(differences) <= 1e-5,
        }
        save("native-parity", parity)
        del legacy, text
        gc.collect()
        torch.cuda.empty_cache()
        if not parity["passed"]:
            raise ValueError("pretrained v1/native parity failed; refuse continuation")
    raw_temperatures(decision)
    cal = evaluate_tracks(decision, calibration, ("off",))
    reasoning_calibration = selection(calibration, 32, 8)
    cal["rows"]["low"] = evaluate_tracks(decision, reasoning_calibration, ("low",))["rows"]["low"]
    cal = save("calibration", cal, "calibration")
    fits = fit_report_calibrations(cal)
    for domain, artifact in fits.items():
        artifact.save(out / f"{name}-temperature-{'-'.join(domain)}.json")
    raw = evaluate_tracks(decision, dev, ("off",))
    reasoning_dev = selection(dev, 32, 8)
    secondary = evaluate_tracks(decision, reasoning_dev, ("low", "high"))
    raw["rows"].update(secondary["rows"])
    raw["summary"].update(secondary["summary"])
    raw["reasoning_scope"] = "secondary preselected subset; high calibration is not fitted"
    raw = save("raw-dev", raw, "dev")
    calibrated = save("calibrated-dev", recalibrate_report(raw, fits), "dev")
    if diagnostics:
        rich = [
            s
            for s in dev
            if s.metadata["source_example_id"].startswith("recovery-1/")
            and s.metadata["oracle_facts"]["index"] < 6
        ]
        save("trace-diagnosis", evaluate_trace_diagnostics(decision, rich), "dev")
    for mode in public_modes:
        save(f"public-{mode}", public_report(decision, mode), "public")
    if name == "v1":
        # LM-only baseline is limited to supported letter sets; 60-label pointer
        # questions cannot be called zero-shot when the trained head is retained.
        cfg = decision.model.cfg
        decision.model.cfg = replace(cfg, readout="lm")
        subset = []
        for sample in dev:
            raw_sample = sample.to_json()
            raw_sample["questions"] = [
                q for q in raw_sample["questions"] if len(q["candidates"]) <= 26
            ]
            if raw_sample["questions"] and "media" not in sample.metadata:
                subset.append(Sample.from_json(raw_sample))
        with decision.model.backbone.disable_adapter():
            baseline = evaluate_tracks(decision, subset, ("off",))
        baseline["scope"] = "native original LM, no decision LoRA; text <=26 candidates only"
        save("pretrained-lm-dev", baseline, "dev")
        decision.model.cfg = cfg
    print(
        json.dumps(
            {
                "stage": "measured",
                "checkpoint": name,
                "seconds": time.monotonic() - start,
                "raw": raw["summary"],
                "calibrated": calibrated["summary"],
            }
        ),
        flush=True,
    )
    del decision
    gc.collect()
    torch.cuda.empty_cache()
    return calibrated, fits


def main(argv=None):
    from ayaka.training.run_v2 import main as train

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    _, splits = validate_bundle(args.kit / "recovery-bundle")
    dev, calibration = selection(splits["dev"]), selection(splits["calibration"], 64, 32)
    selection_receipt = {
        "dev_ids": [s.metadata["source_example_id"] for s in dev],
        "calibration_ids": [s.metadata["source_example_id"] for s in calibration],
        "primary_comparison": "v1 calibrated off vs pilot calibrated off",
        "reasoning_comparison": "secondary diagnostics, not primary promotion evidence",
        "planned_steps": 200,
        "test_access": "only after dev primary screen passes",
    }
    (args.out / "selection.json").write_bytes(canonical(selection_receipt) + b"\n")
    baseline, _ = measure(args.kit / "v1-checkpoint", "v1", dev, calibration, args.out)
    current, _ = measure(
        args.kit / "current-checkpoint",
        "current-v2",
        dev,
        calibration,
        args.out,
        public_modes=("off", "low", "high"),
    )
    (args.out / "current-screen.json").write_bytes(
        canonical(promotion_screen(baseline, current)) + b"\n"
    )
    print(json.dumps({"stage": "fixed_pilot", "steps": 200}), flush=True)
    train(
        [
            "--bundle",
            str(args.kit / "recovery-bundle"),
            "--init-checkpoint",
            str(args.kit / "v1-checkpoint"),
            "--execute",
            "--steps",
            "200",
            "--max-train-seconds",
            "2400",
            "--checkpoint-every",
            "200",
            "--out",
            str(args.out / "pilot"),
        ]
    )
    candidate, fits = measure(args.out / "pilot/checkpoint", "pilot", dev, calibration, args.out)
    screen = promotion_screen(baseline, candidate)
    (args.out / "pilot-screen.json").write_bytes(canonical(screen) + b"\n")
    if screen["screen_passed"]:
        from ayaka.multimodal import load_image_decision

        decision = load_image_decision(
            str(args.out / "pilot/checkpoint"), device="cuda", dtype=torch.bfloat16
        )
        raw_temperatures(decision)
        final = evaluate_tracks(decision, splits["test"], ("off",))
        final.update(split="test", model_id=checkpoint_fingerprint(args.out / "pilot/checkpoint"))
        for rows in final["rows"].values():
            for row in rows:
                row["model_id"] = final["model_id"]
        (args.out / "fresh-authored-test.json").write_bytes(
            canonical(recalibrate_report(final, fits)) + b"\n"
        )
    result = {
        "complete": True,
        "planned_steps": 200,
        "screen_passed": screen["screen_passed"],
        "full_training_started": False,
        "test_evaluated": screen["screen_passed"],
        "official_composite": None,
        "release_promoted": False,
    }
    (args.out / "complete.json").write_bytes(canonical(result) + b"\n")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    main()
