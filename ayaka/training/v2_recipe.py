"""The v2 training recipe validated by the final run of 2026-10-10.

``docs/experiments/V2_FINAL_RUN_RESULT_2026-10-10.md`` records the run. It
needed settings that used to be passed by hand. This module holds them so
the training CLIs apply them by default:

- the model configs: LM readout, 4096-token training rows and 8192-token
  serving prompts on every size. Each size keeps the backbone, LoRA rank
  and head dimensions of its v1 counterpart (large: Gemma 4 12B, rank 64;
  base: E4B, rank 32; small: E2B, rank 32). Only large was trained in the
  final run; base and small apply the same recipe unvalidated;
- StrategyQA as the extra human source of corpus plan 3 whenever ContractNLI
  enables plan 2 or later;
- the learning rates that trained without divergence (LoRA 3e-5, head
  1.5e-4). The earlier 1e-4 / 5e-4 diverged by step 500 on this corpus;
- checkpoint selection on the bundle's router_train split every 250 steps,
  with patience 3, so the exported adapter is the best read, not the last
  step;
- the token-rate training forecast for credit admission.

The corpus quotas, epochs and rows per step live in
``docs/experiments/direct_final_corpus_settings.json``. Native attention
without Liger stays the default of the corpus and bundle preparers.

Usage::

    python -m ayaka.training.v2_recipe model-config --out direct-config.json
    python -m ayaka.training.v2_recipe model-config --size small --out small-config.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ..config import AYAKA_BASE, AYAKA_LARGE, AYAKA_SMALL, AyakaConfig

LORA_LR = 3e-5
HEAD_LR = 1.5e-4
SELECT_EVERY = 250
SELECT_PATIENCE = 3
SELECT_SPLIT = "router_train"
FORECAST_BASIS = "token_rate"
PLAN3_EXTRA_NATURAL = ("strategyqa",)


def _v2(v1, name):
    return AyakaConfig(
        **{
            **asdict(v1),
            "name": name,
            "readout": "lm",
            "version": 2,
            "max_seq_len": 4096,
            "serve_max_seq_len": 8192,
            "long_prompt_tokens": 1024,
            "input_contract_required": False,
        }
    )


LARGE_MODEL = _v2(AYAKA_LARGE, "ayaka-v2-final-direct")
BASE_MODEL = _v2(AYAKA_BASE, "ayaka-v2-base-direct")
SMALL_MODEL = _v2(AYAKA_SMALL, "ayaka-v2-small-direct")
MODELS = {"large": LARGE_MODEL, "base": BASE_MODEL, "small": SMALL_MODEL}


def add_extra_natural_arguments(parser, choices):
    """Add --extra-natural (repeatable) and --no-extra-natural to a CLI."""
    parser.add_argument(
        "--extra-natural",
        action="append",
        choices=choices,
        default=None,
        help="extra human source of corpus plan3; default with --contractnli-train: "
        + ", ".join(PLAN3_EXTRA_NATURAL),
    )
    parser.add_argument(
        "--no-extra-natural",
        action="store_true",
        help="build plan2 without the default extra human sources",
    )


def extra_natural(args):
    """The extra human sources a CLI run uses, with the recipe default.

    The default applies only to full regeneration with ContractNLI, which is
    what enables corpus plan2 and plan3. Explicit sources always win.
    """
    if args.extra_natural is not None:
        if args.no_extra_natural:
            raise ValueError("--extra-natural and --no-extra-natural are exclusive")
        return tuple(args.extra_natural)
    if args.no_extra_natural or getattr(args, "contractnli_train", None) is None:
        return ()
    return PLAN3_EXTRA_NATURAL


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser("model-config", help="write the validated model config JSON")
    model.add_argument("--size", choices=sorted(MODELS), default="large")
    model.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("choose a new output; existing configs must remain intact")
    config = asdict(MODELS[args.size])
    args.out.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "name": config["name"]}))


if __name__ == "__main__":
    main()
