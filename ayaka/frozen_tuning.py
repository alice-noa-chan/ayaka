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
"""

from __future__ import annotations

import json
from pathlib import Path

FROZEN_FILE = "frozen_tuning.json"
# The held-out reads ran every question with an 8192-token prompt budget.
READ_MAX_SEQ_LEN = 8192


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
