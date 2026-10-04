"""Cache native direct probabilities once, independently of reasoned teachers."""

from __future__ import annotations

import math

import torch

from ..eval.read_artifact import fingerprint
from .batching import _noul_canonical
from .swift_direct import direct_readout_binding, validate_direct_input_items

VERSION = "ayaka-frozen-native-replay-1"


def attach_base_replay(trainer, samples, groups, native_weights_sha256, *, saved=None):
    """Attach only train natural rows; disable adapters when collecting the base.

    Restoring saved reads does not regenerate targets after an interrupted run.
    The caller binds their digest into the optimizer state. No extra base model
    or per-step teacher forward is retained in GPU memory.
    """
    from .run_v2 import trainable_digest

    if trainer.step_i or trainer.opt.state:
        raise ValueError("frozen native replay must be bound before optimizer updates")
    validate_direct_input_items([item for group in groups for item in group])
    if trainer.model.cfg.readout != "lm" or not hasattr(trainer.model.backbone, "disable_adapter"):
        raise ValueError("frozen native replay requires a LoRA native LM model")
    if (
        not isinstance(native_weights_sha256, str)
        or len(native_weights_sha256) != 64
        or any(c not in "0123456789abcdef" for c in native_weights_sha256)
    ):
        raise ValueError("frozen native replay requires the exact loaded-weight digest")
    selected, schema = [], []
    for sample, group in zip(samples, groups, strict=True):
        if sample.metadata.get("split") != "train":
            raise ValueError("base replay must never read heldout samples")
        if sample.metadata.get("data_kind") == "natural":
            for q, item in zip(sample.questions, group, strict=True):
                if not item.direct_distillation or any(
                    getattr(item, key) is not None
                    for key in (
                        "reasoning_labels",
                        "reasoning_positions",
                        "proposal_labels",
                        "native_inputs",
                    )
                ):
                    raise ValueError("base replay requires original-input direct rows only")
                selected.append((sample.metadata["source_example_id"] + "/" + q.id, item))
                schema.append(
                    {
                        "id": selected[-1][0],
                        "input_sha256": fingerprint(
                            item.enc.prefix_ids + item.enc.rendered.suffix_ids
                        ),
                        "candidate_ids": [c.id for c in _noul_canonical(q).candidates],
                        "count": len(item.target),
                        "direct_readout_binding": direct_readout_binding(item),
                    }
                )
    if not selected:
        raise ValueError("frozen-base replay needs nonempty natural train rehearsal")
    if len({row["id"] for row in schema}) != len(schema):
        raise ValueError("duplicate frozen native replay question identity")
    header = {
        "version": VERSION,
        "native_weights_sha256": native_weights_sha256,
        "model_sha256": fingerprint(trainer.model.cfg.__dict__),
        "schema": schema,
        "inference_mode": "off",
        "generated_tokens": 0,
        "adapters_disabled": True,
        "scope": "frozen native direct probabilities on natural train only",
    }
    if saved is None:
        before = trainable_digest(trainer.model)
        mode, rng = trainer.model.training, torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if trainer.device.type == "cuda" else None
        try:
            with trainer.model.backbone.disable_adapter():
                probabilities = trainer.predict(
                    [item for _, item in selected], apply_temperature=False
                )
        finally:
            trainer.model.train(mode)
            torch.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        if trainable_digest(trainer.model) != before or trainer.step_i or trainer.opt.state:
            raise ValueError("frozen native collection changed live training parameters")
        saved = {"header": header, "probabilities": probabilities}
    elif (
        not isinstance(saved, dict)
        or set(saved) != {"header", "probabilities"}
        or saved["header"] != header
    ):
        raise ValueError("saved base replay belongs to another native model/input/split/order")
    probabilities = saved["probabilities"]
    if not isinstance(probabilities, list) or len(probabilities) != len(selected):
        raise ValueError("saved base replay must cover every natural train question")
    for values, (_, item) in zip(probabilities, selected, strict=True):
        if (
            not isinstance(values, list)
            or len(values) != len(item.target)
            or any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in values)
            or not math.isclose(math.fsum(values), 1, rel_tol=0, abs_tol=1e-6)
        ):
            raise ValueError(
                "saved base replay probabilities must be finite normalized and aligned"
            )
    for values, (_, item) in zip(probabilities, selected, strict=True):
        item.base_probs = list(values)
    return saved
