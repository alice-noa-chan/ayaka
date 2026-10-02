"""Complete the fixed pilot after verified diagnostics, without repeating surveys."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from ayaka.eval.pretraining_v2 import cohort_fingerprint, evaluate_tracks
from ayaka.eval.recovery_v2 import (
    evaluate_trace_diagnostics,
    fit_report_calibrations,
    promotion_screen,
    recalibrate_report,
)
from ayaka.training.prepare_v2 import canonical, validate_bundle
from ayaka.training.scoped_calibration import checkpoint_fingerprint
from scripts.runpod_v2.recovery import (
    high_diagnostic_selection,
    public_report,
    question_ids,
    raw_temperatures,
    selection,
    verify_fresh_test,
)


def validate_parent_reports(baseline, original, parent_id, dev):
    expected = question_ids(dev)
    for report in (baseline, original):
        if (
            report.get("model_id") != parent_id
            or not report.get("complete")
            or report.get("split") != "dev"
            or report.get("cohort_sha256") != cohort_fingerprint(dev)
            or [r["id"] for r in report.get("rows", {}).get("off", [])] != expected
        ):
            raise ValueError(
                "previous parent report must match the immutable checkpoint and dev cohort"
            )


def main(argv=None):
    from ayaka.multimodal import load_image_decision
    from ayaka.training.run_v2 import main as train

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    _, splits = validate_bundle(args.kit / "recovery-bundle")
    dev, calibration = selection(splits["dev"]), selection(splits["calibration"], 64, 32)
    previous = args.kit / "previous-recovery"
    baseline = json.loads((previous / "v1-calibrated-dev.json").read_text())
    original = json.loads((previous / "v1-raw-dev.json").read_text())
    parent_id = checkpoint_fingerprint(args.kit / "v1-checkpoint")
    validate_parent_reports(baseline, original, parent_id, dev)
    # Confirm the new host reproduces the stored native parent before updates.
    parent = load_image_decision(
        str(args.kit / "v1-checkpoint"), device="cuda", dtype=torch.bfloat16
    )
    raw_temperatures(parent)
    probe = evaluate_tracks(parent, [s for s in dev if "media" not in s.metadata][:18], ("off",))
    stored = {r["id"]: r for r in original["rows"]["off"]}
    delta = max(
        abs(p - q)
        for row in probe["rows"]["off"]
        for p, q in zip(row["probs"], stored[row["id"]]["probs"], strict=True)
    )
    (args.out / "parent-host-parity.json").write_bytes(
        canonical(
            {
                "passed": delta <= 1e-5,
                "max_probability_delta": delta,
                "questions": len(probe["rows"]["off"]),
                "optimizer_steps": 0,
            }
        )
        + b"\n"
    )
    if delta > 1e-5:
        raise ValueError("new host differs from stored parent; refuse matched continuation")
    del parent
    gc.collect()
    torch.cuda.empty_cache()
    available = min(3600, int(args.deadline_epoch - time.time() - 600))
    if available <= 0:
        raise ValueError("no complete training window remains after reserving QA and delivery")
    print(
        json.dumps({"stage": "fixed_pilot", "steps": 200, "training_window_seconds": available}),
        flush=True,
    )
    train(
        [
            "--bundle",
            str(args.kit / "recovery-bundle"),
            "--init-checkpoint",
            str(args.kit / "v1-checkpoint"),
            "--profile-reference",
            str(args.kit / "profile-reference"),
            "--execute",
            "--steps",
            "200",
            "--max-train-seconds",
            str(available),
            "--checkpoint-every",
            "200",
            "--out",
            str(args.out / "pilot"),
        ]
    )

    identity = checkpoint_fingerprint(args.out / "pilot/checkpoint")

    def save(name, report, split):
        report.update(model_id=identity, split=split, weights_selected=False)
        for rows in report.get("rows", {}).values():
            for row in rows:
                row["model_id"] = identity
        (args.out / f"pilot-{name}.json").write_bytes(canonical(report) + b"\n")
        return report

    candidate = load_image_decision(
        str(args.out / "pilot/checkpoint"), device="cuda", dtype=torch.bfloat16
    )
    raw_temperatures(candidate)
    cal = save("calibration", evaluate_tracks(candidate, calibration, ("off",)), "calibration")
    fits = fit_report_calibrations(cal)
    for domain, artifact in fits.items():
        artifact.save(args.out / f"pilot-temperature-{'-'.join(domain)}.json")
    raw = evaluate_tracks(candidate, dev, ("off",))
    secondary = evaluate_tracks(candidate, high_diagnostic_selection(dev), ("high",))
    raw["rows"].update(secondary["rows"])
    raw["summary"].update(secondary["summary"])
    raw["reasoning_scope"] = (
        "pilot low evaluated only in matched 54-question trace diagnosis; high=3 diagnostics with 1024-token budgets"
    )
    raw = save("raw-dev", raw, "dev")
    calibrated = save("calibrated-dev", recalibrate_report(raw, fits), "dev")
    rich = [
        s
        for s in dev
        if s.metadata["source_example_id"].startswith("recovery-1/")
        and s.metadata["oracle_facts"]["index"] < 6
    ]
    save("trace-diagnosis", evaluate_trace_diagnostics(candidate, rich), "dev")
    save("public-off", public_report(candidate, "off"), "public")
    screen = promotion_screen(baseline, calibrated)
    (args.out / "pilot-screen.json").write_bytes(canonical(screen) + b"\n")
    for path in previous.glob("*.json"):
        if path.name.startswith(("v1-", "current-")):
            (args.out / path.name).write_bytes(path.read_bytes())
    if screen["screen_passed"]:
        final = save("fresh-raw-test", evaluate_tracks(candidate, splits["test"], ("off",)), "test")
        (args.out / "fresh-authored-test.json").write_bytes(
            canonical(recalibrate_report(final, fits)) + b"\n"
        )
    del candidate
    gc.collect()
    torch.cuda.empty_cache()
    if screen["screen_passed"]:
        verify_fresh_test(args.kit, args.out)
    result = {
        "complete": True,
        "planned_steps": 200,
        "screen_passed": screen["screen_passed"],
        "full_training_started": False,
        "test_evaluated": screen["screen_passed"],
        "official_composite": None,
        "release_promoted": False,
        "previous_diagnostics_reused": True,
    }
    (args.out / "complete.json").write_bytes(canonical(result) + b"\n")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    main()
