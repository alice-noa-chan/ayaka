import copy

import pytest
import torch
from test_direct_corpus_plan import planned_inputs
from test_direct_state import trainer

from ayaka.eval.read_artifact import fingerprint
from ayaka.losses import LossWeights
from ayaka.training.direct_bundle import audit_bundle, prepare_bundle, training_batches
from ayaka.training.direct_corpus_plan import MARKER, plan_sha256, prepared_groups_sha256
from ayaka.training.direct_state import (
    load_training_state,
    save_training_state,
    train_fixed_schedule,
    training_binding,
)
from ayaka.training.frozen_replay import attach_base_replay
from ayaka.training.trainer import TrainConfig


def replay_bundle(tmp_path, *, weight=0.2):
    torch.set_num_threads(1)
    splits, cfg, tok, registry, plan = planned_inputs()
    plan["settings"].update(epochs=1, rows_per_step=40)
    for samples in splits.values():
        for sample in samples:
            sample.metadata[MARKER] = plan_sha256(plan)
    root = tmp_path / "bundle"
    prepare_bundle(
        root,
        splits,
        tok,
        cfg,
        {},
        steps=1,
        rows_per_step=40,
        seed=7,
        allow_tiny=True,
        natural_registry=registry,
        corpus_plan=plan,
        weights=LossWeights(
            pointer_aux=0, gold_nll_with_teacher=True, distill=0, base_replay=weight
        ),
    )
    manifest, recipe, _, inventory, groups = audit_bundle(
        root, allow_tiny=True, natural_registry=registry
    )
    tcfg = TrainConfig(
        steps=1,
        questions_per_step=40,
        seed=7,
        bf16=False,
        log_every=0,
        loss_weights=LossWeights(**recipe["loss_weights"]),
        reasoning_ce_weight=0,
        proposal_ce_weight=0,
        grad_checkpointing=False,
    )
    return splits, manifest, recipe, inventory, groups, tcfg


def test_planned_replay_keeps_static_group_digest_and_trains_and_restores_exact_state(tmp_path):
    samples, manifest, recipe, inventory, groups, tcfg = replay_bundle(tmp_path)
    live = trainer(recipe, tcfg)
    native = fingerprint("random CPU native fixture bytes; no pretrained weights")
    before = prepared_groups_sha256(groups)
    with pytest.raises(ValueError, match="require aligned"):
        list(training_batches(recipe, inventory, groups))
    base = attach_base_replay(live, samples["train"], groups, native)
    assert (
        prepared_groups_sha256(groups)
        == before
        == recipe["corpus_contract"]["prepared_groups_sha256"]
    )
    assert len(list(training_batches(recipe, inventory, groups))) == 1
    binding = training_binding(manifest, recipe, tcfg, native_weights_sha256=native)
    binding["frozen_base_reads_sha256"] = fingerprint(base)
    records = train_fixed_schedule(live, recipe, inventory, groups)
    assert records[0]["base_replay_kl"] >= 0 and live.step_i == 1
    save_training_state(live, tmp_path / "state", binding)
    restored = trainer(recipe, tcfg)
    fresh_groups = copy.deepcopy(groups)
    for group in fresh_groups:
        for item in group:
            item.base_probs = None
    restored_reads = attach_base_replay(
        restored, samples["train"], fresh_groups, native, saved=base
    )
    assert restored_reads == base
    load_training_state(restored, tmp_path / "state", binding)
    assert train_fixed_schedule(restored, recipe, inventory, fresh_groups) == []
    for original, actual in zip(live.model.parameters(), restored.model.parameters(), strict=True):
        torch.testing.assert_close(original, actual, atol=0, rtol=0)


@pytest.mark.parametrize(
    "damage", ["authored", "missing", "nan", "unnormalized", "width", "teacher", "tokens"]
)
def test_runtime_replay_cannot_relax_original_input_or_probability_contract(tmp_path, damage):
    samples, _, recipe, inventory, groups, tcfg = replay_bundle(tmp_path)
    live = trainer(recipe, tcfg)
    attach_base_replay(live, samples["train"], groups, fingerprint("random CPU base fixture"))
    authored = groups[0][0]
    natural = next(
        group[0]
        for row, group in zip(inventory, groups, strict=True)
        if row["data_kind"] == "natural"
    )
    if damage == "authored":
        authored.base_probs = [1 / len(authored.target)] * len(authored.target)
    elif damage == "missing":
        natural.base_probs = None
    elif damage == "nan":
        natural.base_probs[0] = float("nan")
    elif damage == "unnormalized":
        natural.base_probs = [2.0] * len(natural.target)
    elif damage == "width":
        natural.base_probs.pop()
    elif damage == "teacher":
        natural.teacher = list(natural.base_probs)
    else:
        natural.enc.rendered.suffix_ids[0] += 1
    with pytest.raises(
        ValueError, match="runtime base replay|require aligned|actual corpus training"
    ):
        list(training_batches(recipe, inventory, groups))


def test_gold_only_plan_rejects_even_normalized_runtime_replay_injection(tmp_path):
    _, _, recipe, inventory, groups, _ = replay_bundle(tmp_path, weight=0)
    item = groups[-1][0]
    item.base_probs = [1 / len(item.target)] * len(item.target)
    with pytest.raises(ValueError, match="unplanned/authored"):
        list(training_batches(recipe, inventory, groups))
