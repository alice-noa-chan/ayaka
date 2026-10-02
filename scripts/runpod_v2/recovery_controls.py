"""Post-training integrity control; independent menus/truth, no weight selection."""

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import torch

from ayaka.eval.pretraining_v2 import evaluate_tracks
from ayaka.eval.v2 import paired_report, summarize
from ayaka.training.prepare_v2 import canonical
from ayaka.training.scoped_calibration import checkpoint_fingerprint
from scripts.runpod_v2.recovery import raw_temperatures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corrected-data-source", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args(argv)
    complete = json.loads((args.out / "pilot/checkpoint/complete.json").read_text())
    if complete != {"steps": 200, "complete": True}:
        raise ValueError("integrity control requires the entire fixed training schedule")
    spec = importlib.util.spec_from_file_location(
        "ayaka.data.recovery_control_v2", args.corrected_data_source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = module.recovery_curriculum("dev", 64)
    for sample in samples:
        sample.metadata["source_example_id"] += "/integrity-2"
    probe = [
        s
        for s in samples
        if s.metadata["language"] == "en" and s.metadata["oracle_facts"]["index"] < 3
    ]
    from ayaka.multimodal import load_image_decision

    torch.set_num_threads(4)
    reports = {}
    for name, checkpoint in (
        ("parent", args.kit / "v1-checkpoint"),
        ("pilot", args.out / "pilot/checkpoint"),
    ):
        print(json.dumps({"stage": "integrity_control", "model": name, "cases": 64}), flush=True)
        decision = load_image_decision(str(checkpoint), device="cuda", dtype=torch.bfloat16)
        raw_temperatures(decision)
        report = evaluate_tracks(decision, samples, ("off",))
        report["low_diagnostic"] = evaluate_tracks(decision, probe, ("low",))
        report.update(
            split="dev", model_id=checkpoint_fingerprint(checkpoint), weights_selected=False
        )
        reports[name] = report
        (args.out / f"integrity-{name}.json").write_bytes(canonical(report) + b"\n")
        del decision
        gc.collect()
        torch.cuda.empty_cache()
    a, b = reports["parent"]["rows"]["off"], reports["pilot"]["rows"]["off"]
    result = {
        "complete": True,
        "scope": "64 existing dev cases with independent numeric menus and randomized Noul truth; robustness, not an independent final test",
        "baseline": summarize(a),
        "candidate": summarize(b),
        "paired": paired_report(a, b),
        "official_composite": None,
        "release_promoted": False,
        "original_dataset_integrity": "gold-centered invoice menus and year-parity Noul truth invalidate stand-alone improvement claims",
    }
    original_gate = json.loads((args.out / "pilot-screen.json").read_text())["screen_passed"]
    proper = all(
        result["candidate"]["by_type"][kind][metric]
        <= result["baseline"]["by_type"][kind][metric] + 1e-8
        for kind in ("choice", "noul", "score")
        for metric in ("nll", "brier", *(("rps",) if kind == "score" else ()))
    )
    control_gate = (
        original_gate
        and proper
        and result["candidate"]["cc_equal_types"] - result["baseline"]["cc_equal_types"] >= 5
        and result["paired"]["cc_delta_95ci"][0] > 0
    )
    result["preliminary_control_gate_passed"] = control_gate
    result["corrected_test_evaluated"] = False
    if control_gate and args.deadline_epoch - time.time() > 780:
        final = module.recovery_curriculum("test", 256)
        final_reports = {}
        for name, checkpoint in (
            ("parent", args.kit / "v1-checkpoint"),
            ("pilot", args.out / "pilot/checkpoint"),
        ):
            decision = load_image_decision(str(checkpoint), device="cuda", dtype=torch.bfloat16)
            raw_temperatures(decision)
            report = evaluate_tracks(decision, final, ("off",))
            report.update(
                split="test", model_id=checkpoint_fingerprint(checkpoint), weights_selected=False
            )
            final_reports[name] = report
            (args.out / f"corrected-test-{name}.json").write_bytes(canonical(report) + b"\n")
            del decision
            gc.collect()
            torch.cuda.empty_cache()
        old, new = final_reports["parent"]["rows"]["off"], final_reports["pilot"]["rows"]["off"]
        result["corrected_test_evaluated"] = True
        result["corrected_test"] = {
            "baseline": summarize(old),
            "candidate": summarize(new),
            "paired": paired_report(old, new),
            "scope": "same held-out case facts with corrected menus/truth, no parameter or mode selection",
        }
    (args.out / "integrity-comparison.json").write_bytes(canonical(result) + b"\n")
    outcome = json.loads((args.out / "complete.json").read_text())
    outcome.update(
        dataset_integrity_control_complete=True,
        large_improvement_claim_permitted=False,
        original_fresh_test_biased=bool(outcome.get("test_evaluated", False)),
    )
    (args.out / "complete.json").write_bytes(canonical(outcome) + b"\n")
    print(
        json.dumps(
            {
                "integrity_control_complete": True,
                "parent_cc": result["baseline"]["cc_equal_types"],
                "pilot_cc": result["candidate"]["cc_equal_types"],
                "ci": result["paired"]["cc_delta_95ci"],
            }
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
