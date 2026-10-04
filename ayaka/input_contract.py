"""Checkpoint-bound direct inputs, shared by training reload and serving.

Legacy checkpoints retain their segmented inputs. A declared Swift checkpoint
must keep its exact renderer and tokenizer; missing or inconsistent metadata
is an error rather than a request to fall back to the legacy prompt.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .eval.read_artifact import fingerprint
from .input_errors import ContextLimitError
from .jev_api import ValidationError
from .training.swift_direct import (
    encode_direct_sample,
    input_serving_recipe,
    normalize_input_encoding,
    swift_question,
)
from .training.tokenizer_identity import scoped_tokenizer_preparation

FIELDS = ("input_encoding", "input_recipe", "input_recipe_sha256")


def metadata_contract(meta, cfg):
    if not isinstance(meta, dict):
        raise ValueError("checkpoint metadata must be an object")
    if not any(key in meta for key in FIELDS):
        if cfg.input_contract_required:
            raise ValueError("checkpoint requires its missing direct input contract")
        return None
    if not all(key in meta for key in FIELDS):
        raise ValueError("checkpoint direct input contract is incomplete")
    encoding = normalize_input_encoding(meta["input_encoding"])
    recipe = meta["input_recipe"]
    if not isinstance(recipe, dict) or fingerprint(recipe) != meta["input_recipe_sha256"]:
        raise ValueError("checkpoint input recipe hash differs")
    if encoding["encoder"] == "swift_canonical":
        from .swift.readers import READOUT
        from .training.swift_direct import VERSION

        expected = {
            "version": VERSION,
            "readout": READOUT,
            **{k: v for k, v in encoding.items() if k != "encoder"},
        }
        if cfg.readout != "lm":
            raise ValueError("Swift checkpoint inputs require a native LM readout")
        hashes = ("chat_template_sha256", "tokenizer_sha256", "tokenizer_config_sha256")
        if set(recipe) != {*expected, *hashes} or any(
            recipe.get(k) != v for k, v in expected.items()
        ):
            raise ValueError("checkpoint input encoding and recipe differ")
        for key in hashes:
            value = recipe[key]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("checkpoint tokenizer/template digest is invalid")
    elif recipe != {"version": "ayaka-segmented-direct-inputs-1", **encoding}:
        raise ValueError("checkpoint segmented input recipe differs")
    return copy.deepcopy({**{key: meta[key] for key in FIELDS}, "input_encoding": encoding})


def read_contract(path, cfg, filename="meta.json"):
    file = Path(path) / filename
    meta = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}
    return metadata_contract(meta, cfg)


def checkpoint_metadata(model, meta=None):
    """Preserve bindings on a second save/export; forbid accidental replacement."""
    result = copy.deepcopy(meta or {})
    existing = getattr(model, "input_contract", None)
    if existing is not None:
        for key in FIELDS:
            if key in result and result[key] != existing[key]:
                raise ValueError("cannot replace the loaded checkpoint input contract")
            result[key] = copy.deepcopy(existing[key])
    metadata_contract(result, model.cfg)
    return result


def validate_tokenizer(contract, tok):
    if (
        contract is not None
        and input_serving_recipe(tok, contract["input_encoding"]) != contract["input_recipe"]
    ):
        raise ValueError("serving tokenizer or chat template differs from checkpoint input recipe")


def serving_question(view, index=0):
    """Keep API labels separate from descriptions, including sorted Score levels."""
    from .data.schema import Candidate, Question

    labels = view.candidate_ids
    if labels is None:
        labels = ["false", "true"] if view.type == "noul" else list(view.descriptions)
    if len(labels) != len(view.descriptions) or len(set(labels)) != len(labels):
        raise ValueError("Swift serving requires unique aligned candidate IDs")
    if view.type == "noul" and labels != ["false", "true"]:
        raise ValueError("Swift Noul serving requires false/true candidate order")
    return Question(
        str(index),
        view.type,
        view.instruction,
        [
            Candidate(label, description, view.ordinals[i] if view.type == "score" else None)
            for i, (label, description) in enumerate(zip(labels, view.descriptions, strict=True))
        ],
        {},
    )


def swift_wire_question(view):
    return swift_question(serving_question(view))


@scoped_tokenizer_preparation
def encode_serving(state, views, tok, cfg, contract, context_limit):
    from .data.schema import Sample

    validate_tokenizer(contract, tok)
    if not 1 <= len(views) <= 64 or any(not 2 <= len(v.descriptions) <= 26 for v in views):
        raise ValidationError(
            "Swift checkpoint reads require 1..64 questions with 2..26 candidates", "questions"
        )
    try:
        questions = [serving_question(v, i) for i, v in enumerate(views)]
        # Validate semantic levels before renderer/tokenizer work. Internal
        # tokenizer or contract failures remain backend errors.
        for question in questions:
            swift_question(question)
    except ValueError as exc:
        raise ValidationError(str(exc), "questions") from exc
    try:
        items = encode_direct_sample(
            Sample(state, questions),
            tok,
            cfg,
            input_encoding=contract["input_encoding"],
            context_limit=context_limit,
        )
    except ContextLimitError as exc:
        raise ValidationError(str(exc), "questions") from exc
    return items[0].enc.prefix_ids, [item.enc for item in items]
