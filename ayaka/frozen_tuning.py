"""Serve a checkpoint with its frozen held-out tuning fit.

``ayaka.eval.heldout_tuning --frozen-out`` writes one JSON file with
everything the final gate measured besides the weights:

- the adapter mode the reads used (merged or unmerged LoRA);
- path temperatures fitted on the calibration cohort;
- the router fitted on router_train and promoted on dev;
- the serving policy chosen on router_train.

``frozen_decision`` rebuilds that exact decision for serving. A checkpoint
that ships ``frozen_tuning.json`` next to its weights is served this way by
default, so a downloaded model behaves as it did in the gate without extra
flags.

``package`` builds that release folder from a trained checkpoint and its
frozen fit::

    python -m ayaka.frozen_tuning package --checkpoint results/train/checkpoint \\
        --frozen results/frozen.json --out release/ayaka-v2-large
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

FROZEN_FILE = "frozen_tuning.json"
# The held-out reads ran every question with an 8192-token prompt budget.
READ_MAX_SEQ_LEN = 8192
# Checkpoint files a release ships. head.pt is the legacy pickle of the same
# head; releases carry head.safetensors only.
RELEASE_FILES = (
    "ayaka_config.json",
    "electra_config.json",
    "head.safetensors",
    "meta.json",
    "noul_calibration.json",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
)
OPTIONAL_FILES = ("electra_config.json", "noul_calibration.json")


def load_frozen(path):
    """Read and validate a frozen tuning file."""
    from .eval.final_gate import validate_frozen

    with open(path, encoding="utf-8") as stream:
        return validate_frozen(json.load(stream))


def find_frozen(folder):
    """Return the frozen tuning file shipped in a model folder, or None."""
    if folder is None:
        return None
    path = Path(folder) / FROZEN_FILE
    return path if path.is_file() else None


def always_types(policy):
    """Primitive types a policy always reasons on."""
    # "noul_always..." and "noul+choice_always..." both start with Noul.
    kinds = ("noul",) if policy.startswith("noul") else ()
    return kinds + (("choice",) if "choice_always" in policy else ())


def frozen_decision(model, tok, frozen, max_seq_len=None):
    """The decision the final gate measured: temperatures, router and policy."""
    from .reasoning_pipeline import controlled_decision
    from .routing import BenefitRouter
    from .training.path_calibration import PathCalibration

    router = BenefitRouter(**frozen["router"]) if "router" in frozen["policy"] else None
    return controlled_decision(
        model,
        tok,
        max_seq_len=max_seq_len or READ_MAX_SEQ_LEN,
        router=router,
        calibration=PathCalibration(frozen["path_temperatures"]),
        always_types=always_types(frozen["policy"]),
        # The temperatures were fitted on reads that kept the automatic
        # direct correction, so it stays on beneath them.
        direct_correction=True,
    )


def package(checkpoint, frozen_path, out):
    """Copy a checkpoint's release files and its frozen fit into a new folder.

    Files are copied byte for byte: the Noul calibration is bound to the
    hash of the config, head and adapter files, and is re-verified here.
    """
    checkpoint, out = Path(checkpoint), Path(out)
    if out.exists():
        raise ValueError("choose a new release folder; existing releases must remain intact")
    frozen = load_frozen(frozen_path)
    missing = [
        name
        for name in RELEASE_FILES
        if name not in OPTIONAL_FILES and not (checkpoint / name).is_file()
    ]
    if missing:
        raise ValueError(f"checkpoint is missing release files: {missing}")
    out.mkdir(parents=True)
    copied = []
    for name in RELEASE_FILES:
        source = checkpoint / name
        if source.is_file():
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, out / name)
            copied.append(name)
    (out / FROZEN_FILE).write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    _verify_calibration(out)
    return {"out": str(out), "files": copied + [FROZEN_FILE], "policy": frozen["policy"]}


def _verify_calibration(folder):
    path = folder / "noul_calibration.json"
    if not path.is_file():
        return
    from .checkpoint import load_config, read_contract
    from .training.noul_calibration import NoulCalibration
    from .training.scoped_calibration import checkpoint_fingerprint

    contract = read_contract(str(folder), load_config(str(folder)))
    NoulCalibration.load(
        str(path),
        checkpoint_fingerprint(folder, include_calibration=False),
        input_recipe_sha256=contract["input_recipe_sha256"] if contract else None,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("package", help="build a release folder with the frozen fit")
    pack.add_argument("--checkpoint", required=True, type=Path)
    pack.add_argument("--frozen", required=True, type=Path)
    pack.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = package(args.checkpoint, args.frozen, args.out)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
