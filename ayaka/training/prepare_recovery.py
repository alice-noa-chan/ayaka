"""Prepare a fixed 200-step continuation experiment without executing updates."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from ..checkpoint import load_config
from ..data.recovery_v2 import recovery_curriculum
from .prepare_v2 import (
    SPLITS,
    VERSION,
    audit_splits,
    canonical,
    inspect_model,
    prepared_items,
    sha256,
    validate_bundle,
)
from .scoped_calibration import checkpoint_fingerprint
from .workload import describe_rows, finite_workload


def clean_splits(previous):
    from ..data.recovery_audit import shortcut_audit
    from ..data.recovery_holdout import independent_holdout

    result, audits = {}, {}
    for split in SPLITS:
        natural = [s for s in previous[split] if s.metadata.get("data_kind") == "natural"]
        if split in {"dev", "test"}:
            authored = independent_holdout(split, 240)
        else:
            authored = recovery_curriculum(
                split, {"train": 512, "router_train": 64, "calibration": 128}[split], generation=3
            )
        audits[split] = shortcut_audit(authored)
        # Never reuse the previously inspected natural or authored final test.
        result[split] = authored if split == "test" else natural + authored
    return result, audits


def prepare_recovery(source_bundle, checkpoint, out, *, training_window_seconds=2400, clean=False):
    if type(training_window_seconds) is not int or not 1 <= training_window_seconds <= 28800:
        raise ValueError("training window must be a positive integer within eight hours")
    root = Path(out)
    if root.exists():
        raise ValueError("recovery preparation requires a new directory")
    old_manifest, splits = validate_bundle(source_bundle)
    cfg = replace(
        load_config(checkpoint),
        name="ayaka-v2-recovery-pilot",
        version=2,
        reasoning_defaults={"mode": "auto", "effort": "medium"},
    )
    identity = checkpoint_fingerprint(checkpoint)
    shortcut_checks = None
    if clean:
        splits, shortcut_checks = clean_splits(splits)
    else:
        counts = {"train": 512, "router_train": 64, "dev": 256, "calibration": 128, "test": 256}
        for split, count in counts.items():
            new = recovery_curriculum(split, count)
            splits[split] = new if split == "test" else splits[split] + new
    audited = audit_splits(splits)
    model_audit, tok, backend = inspect_model(cfg, offline=True)
    context, inventory = {}, []
    for split, samples in splits.items():
        longest, rows = 0, 0
        for sample in samples:
            items = prepared_items(sample, tok, cfg, backend)
            longest = max(longest, *(item.length for item in items))
            rows += len(items)
            if split == "train":
                inventory.append(describe_rows(sample, items))
        context[split] = {"prepared_rows": rows, "max_tokens": longest}
        print(json.dumps({"split": split, **context[split]}), flush=True)
    model_audit.update(
        train_inventory=inventory,
        context_audit=context,
        initialization="checkpoint_continuation",
        parent_sha256=identity,
    )
    recipe = json.loads((Path(source_bundle) / "training_config.json").read_text())
    recipe.update(
        model=asdict(cfg),
        initialization="checkpoint_continuation",
        initial_checkpoint_sha256=identity,
        completion_target_seconds=training_window_seconds,
    )
    recipe["training"].update(
        lr=2e-5, head_lr=5e-5, micro_batch_tokens=4096, grad_checkpointing=True, seed=20261002
    )
    root.mkdir(parents=True)
    for split, samples in splits.items():
        (root / f"{split}.jsonl").write_bytes(
            b"\n".join(canonical(s.to_json()) for s in samples) + b"\n"
        )
    (root / "training_config.json").write_bytes(canonical(recipe) + b"\n")
    (root / "model_preflight.json").write_bytes(canonical(model_audit) + b"\n")
    package = Path(__file__).resolve().parents[1]
    files = [
        *(f"{split}.jsonl" for split in SPLITS),
        "training_config.json",
        "model_preflight.json",
    ]
    manifest = {
        **old_manifest,
        "counts": audited,
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(path.relative_to(package)).replace("\\", "/"): sha256(path.read_bytes())
            for path in sorted(package.rglob("*.py"))
        },
        "files": {name: sha256((root / name).read_bytes()) for name in files},
        "version": VERSION,
        "parent_bundle_sha256": sha256((Path(source_bundle) / "manifest.json").read_bytes()),
        "data_scope": "natural rehearsal plus generation-3 corrected recovery; separately authored rule-combination dev/test"
        if clean
        else "natural rehearsal plus multistage authored recovery; new authored-only final test",
        "recovery_policy": "clean-3" if clean else "legacy",
        "optimizer_steps_executed": 0,
    }
    if clean:
        (root / "shortcut-audit.json").write_bytes(canonical(shortcut_checks) + b"\n")
        manifest["shortcut_audit_sha256"] = sha256((root / "shortcut-audit.json").read_bytes())
    (root / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    validate_bundle(root)
    workload = finite_workload(
        inventory,
        200,
        recipe["training"]["questions_per_step"],
        recipe["training"]["seed"],
        recipe["language_sampling"],
    )
    (root / "workload-200.json").write_bytes(canonical(workload) + b"\n")
    (root / "experiment.json").write_bytes(
        canonical(
            {
                "planned_steps": 200,
                "schedule_sha256": workload["schedule_sha256"],
                "parent_checkpoint_sha256": identity,
                "overall_dev_gain_minimum_points": 5,
                "paired_ci_lower_minimum": 0,
                "temporal_numeric_gain_minimum_points": 5,
                "max_type_language_regression_points": 1,
                "no_probability_regression": True,
                "final_test": "independent-recovery-1 authored cases, no training traces; shared dev/test rule families, not official JevBench or natural generalization"
                if clean
                else "new authored-only cases, not an official JevBench or full natural-language promotion",
                "raw_improvement_required": clean,
                "calibrated_improvement_required": clean,
                "published_parent_nonregression_required": clean,
                "old_profiles_or_parent_reports_reusable": False
                if clean
                else "only exact content-bound inputs",
                "full_training_authorized_by_screen": False,
            }
        )
        + b"\n"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--training-window-seconds", type=int, default=2400)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="exclude old authored data and prepare fresh independent rule-combination evaluation",
    )
    args = parser.parse_args(argv)
    return prepare_recovery(
        args.source_bundle,
        args.checkpoint,
        args.out,
        training_window_seconds=args.training_window_seconds,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
