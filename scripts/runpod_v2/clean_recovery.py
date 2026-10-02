"""Audited clean continuation: CPU plan by default, fixed pilot only on execute."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from ayaka.eval.pretraining_v2 import evaluate_tracks
from ayaka.eval.recovery_v2 import fit_report_calibrations, promotion_screen, recalibrate_report
from ayaka.training.prepare_v2 import canonical, sha256, validate_bundle
from ayaka.training.run_v2 import source_matches
from ayaka.training.scoped_calibration import checkpoint_fingerprint
from scripts.runpod_v2.recovery import raw_temperatures


def validate_clean_bundle(bundle, parent):
    manifest, splits = validate_bundle(bundle)
    if manifest.get("recovery_policy") != "clean-3" or not source_matches(manifest):
        raise ValueError(
            "clean execution requires the exact newly prepared source and clean-3 data"
        )
    audit_path = Path(bundle) / "shortcut-audit.json"
    if sha256(audit_path.read_bytes()) != manifest["shortcut_audit_sha256"]:
        raise ValueError("shortcut audit checksum mismatch")
    audits = json.loads(audit_path.read_text(encoding="utf-8"))
    if not all(audits[s]["passed"] for s in splits):
        raise ValueError("clean dataset failed shortcut audit")
    recipe = json.loads((Path(bundle) / "training_config.json").read_text(encoding="utf-8"))
    if recipe["initial_checkpoint_sha256"] != checkpoint_fingerprint(parent):
        raise ValueError("clean continuation requires the pinned original parent")
    if any(s.metadata.get("evaluation_only") for s in splits["train"]):
        raise ValueError("holdout leaked into training")
    return manifest, splits


def clean_screen(
    parent_raw, pilot_raw, parent_calibrated, pilot_calibrated, parent_published, *, replicates=2000
):
    if len(
        {
            r.get("cohort_sha256")
            for r in (parent_raw, pilot_raw, parent_calibrated, pilot_calibrated, parent_published)
        }
    ) != 1 or not parent_raw.get("cohort_sha256"):
        raise ValueError("clean comparisons require identical content-bound cohorts")
    raw = promotion_screen(parent_raw, pilot_raw, replicates=replicates)
    calibrated = promotion_screen(parent_calibrated, pilot_calibrated, replicates=replicates)
    published = promotion_screen(parent_published, pilot_calibrated, replicates=replicates)
    return {
        "screen_passed": all(r["screen_passed"] for r in (raw, calibrated, published)),
        "raw": raw,
        "calibrated": calibrated,
        "published_parent": published,
        "official_composite": None,
        "full_training_authorized": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "execute"), default="plan", nargs="?")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--training-window-seconds", type=int, default=3600)
    parser.add_argument("--evaluation-window-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    _, splits = validate_clean_bundle(args.bundle, args.parent)
    if args.out.exists() or min(args.training_window_seconds, args.evaluation_window_seconds) <= 0:
        raise ValueError("require new output and positive explicit budgets")
    from ayaka.training.run_v2 import main as train

    if args.action == "plan":
        return train(
            [
                "--bundle",
                str(args.bundle),
                "--init-checkpoint",
                str(args.parent),
                "--plan-only",
                "--planned-steps",
                "200",
                "--out",
                str(args.out),
            ]
        )
    from ayaka.multimodal import load_image_decision

    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    # Evaluation budget accumulates across both models; loading and training use
    # the explicit separate training window. An external Pod envelope is required.
    remaining = args.evaluation_window_seconds

    def evaluate(model, identity, name, samples, split, modes=("off",)):
        nonlocal remaining
        start = time.monotonic()
        result = evaluate_tracks(model, samples, modes, deadline=start + remaining)
        remaining -= time.monotonic() - start
        result.update(model_id=identity, split=split, weights_selected=False)
        for rows in result["rows"].values():
            for row in rows:
                row["model_id"] = identity
        (args.out / (name + ".json")).write_bytes(canonical(result) + b"\n")
        if not result["complete"] or remaining <= 0:
            raise TimeoutError("evaluation incomplete; refuse promotion or final-test selection")
        return result

    parent_id = checkpoint_fingerprint(args.parent)
    model = load_image_decision(str(args.parent), device="cuda", dtype=torch.bfloat16)
    published = evaluate(model, parent_id, "parent-published-dev", splits["dev"], "dev")
    raw_temperatures(model)
    parent_cal = evaluate(
        model, parent_id, "parent-calibration", splits["calibration"], "calibration"
    )
    parent_raw = evaluate(model, parent_id, "parent-raw-dev", splits["dev"], "dev")
    fits = fit_report_calibrations(parent_cal)
    parent_fitted = recalibrate_report(parent_raw, fits)
    (args.out / "parent-calibrated-dev.json").write_bytes(canonical(parent_fitted) + b"\n")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    train(
        [
            "--bundle",
            str(args.bundle),
            "--init-checkpoint",
            str(args.parent),
            "--execute",
            "--steps",
            "200",
            "--max-train-seconds",
            str(args.training_window_seconds),
            "--checkpoint-every",
            "200",
            "--out",
            str(args.out / "pilot"),
        ]
    )
    checkpoint = args.out / "pilot/checkpoint"
    if json.loads((checkpoint / "complete.json").read_text(encoding="utf-8")) != {
        "steps": 200,
        "complete": True,
    }:
        raise ValueError("clean evaluation requires all 200 optimizer steps")
    pilot_id = checkpoint_fingerprint(checkpoint)
    model = load_image_decision(str(checkpoint), device="cuda", dtype=torch.bfloat16)
    raw_temperatures(model)
    pilot_cal = evaluate(model, pilot_id, "pilot-calibration", splits["calibration"], "calibration")
    pilot_raw = evaluate(model, pilot_id, "pilot-raw-dev", splits["dev"], "dev")
    pilot_fitted = recalibrate_report(pilot_raw, fit_report_calibrations(pilot_cal))
    (args.out / "pilot-calibrated-dev.json").write_bytes(canonical(pilot_fitted) + b"\n")
    screen = clean_screen(parent_raw, pilot_raw, parent_fitted, pilot_fitted, published)
    (args.out / "screen.json").write_bytes(canonical(screen) + b"\n")
    # Full high budgets are retained in a predeclared tiny diagnostic cohort;
    # neither effort nor weights are selected from these diagnostic results.
    probe = [
        s
        for s in splits["dev"]
        if s.metadata["source"] == "ayaka-independent-recovery-1"
        and "/0/en" in s.metadata["source_example_id"]
    ]
    evaluate(model, pilot_id, "pilot-forced-high", probe, "dev", ("high",))
    final = (
        evaluate(model, pilot_id, "pilot-test", splits["test"], "test")
        if screen["screen_passed"]
        else None
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if final is not None:
        model = load_image_decision(str(args.parent), device="cuda", dtype=torch.bfloat16)
        raw_temperatures(model)
        baseline = evaluate(model, parent_id, "parent-test", splits["test"], "test")
        from ayaka.eval.v2 import paired_report

        (args.out / "test-comparison.json").write_bytes(
            canonical(
                {
                    "baseline": baseline["summary"],
                    "candidate": final["summary"],
                    "paired": paired_report(baseline["rows"]["off"], final["rows"]["off"]),
                    "official_composite": None,
                }
            )
            + b"\n"
        )
    result = {
        "complete": True,
        "training_steps": 200,
        "screen_passed": screen["screen_passed"],
        "test_evaluated": final is not None,
        "release_promoted": False,
        "full_training_started": False,
        "official_composite": None,
    }
    (args.out / "complete.json").write_bytes(canonical(result) + b"\n")
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
