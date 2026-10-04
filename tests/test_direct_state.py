import copy
import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import load_file, save_file

from ayaka.checkpoint import apply_lora
from ayaka.config import ElectraConfig, tiny_config
from ayaka.data.reasoning_v2 import SPLITS, curriculum
from ayaka.eval.read_artifact import fingerprint
from ayaka.losses import LossWeights
from ayaka.model.electra import ElectraDecisionModel
from ayaka.tokenization import ToyTokenizer
from ayaka.training.direct_bundle import audit_bundle, prepare_bundle
from ayaka.training.direct_state import (
    file_digest,
    load_training_state,
    save_training_state,
    train_fixed_schedule,
    training_binding,
)
from ayaka.training.prepare_v2 import canonical
from ayaka.training.trainer import TrainConfig, Trainer


def fixture(tmp_path, *, checkpointed=False):
    torch.set_num_threads(1)
    splits = {}
    for split in SPLITS:
        splits[split] = [sample for sample, _ in curriculum(split, 10)]
        for sample in splits[split]:
            sample.metadata["source_lineage"] = sample.metadata["case_facts_sha256"]
    tok = ToyTokenizer()
    cfg = tiny_config(readout="lm", max_seq_len=2048, lora_dropout=0.05)
    prepare_bundle(
        tmp_path / "bundle",
        splits,
        tok,
        cfg,
        {},
        steps=3,
        rows_per_step=3,
        seed=19,
        allow_tiny=True,
    )
    manifest, recipe, _, inventory, groups = audit_bundle(tmp_path / "bundle", allow_tiny=True)
    tcfg = TrainConfig(
        steps=3,
        questions_per_step=3,
        seed=19,
        bf16=False,
        log_every=0,
        loss_weights=LossWeights(**recipe["loss_weights"]),
        reasoning_ce_weight=0,
        proposal_ce_weight=0,
        grad_checkpointing=checkpointed,
    )
    binding = training_binding(
        manifest, recipe, tcfg, native_weights_sha256=fingerprint("tiny seed0")
    )
    return recipe, inventory, groups, tcfg, binding


def trainer(recipe, tcfg):
    model = ElectraDecisionModel.from_config(ElectraConfig(**recipe["model"]), dtype=torch.float32)
    model.backbone.requires_grad_(False)
    apply_lora(model)
    return Trainer(model, ToyTokenizer(), tcfg, "cpu")


@pytest.mark.parametrize("checkpointed", [False, True])
def test_atomic_resume_matches_uninterrupted_lora_adam_scheduler_and_dropout_rng(
    tmp_path, checkpointed
):
    recipe, inventory, groups, cfg, binding = fixture(tmp_path, checkpointed=checkpointed)
    uninterrupted = trainer(recipe, cfg)
    full = train_fixed_schedule(uninterrupted, recipe, inventory, groups)
    complete_rng = torch.get_rng_state().clone()
    interrupted = trainer(recipe, cfg)

    def simulated_crash(step, record):
        save_training_state(interrupted, tmp_path / "state", binding)
        raise RuntimeError("simulated process loss after durable optimizer step")

    with pytest.raises(RuntimeError, match="simulated process loss"):
        train_fixed_schedule(interrupted, recipe, inventory, groups, on_step=simulated_crash)
    assert interrupted.step_i == 1
    resumed = trainer(recipe, cfg)
    header = load_training_state(resumed, tmp_path / "state", binding)
    assert header["complete_schedule"] is False
    continuation = train_fixed_schedule(resumed, recipe, inventory, groups)
    assert [r["step"] for r in continuation] == [2, 3]
    for expected, actual in zip(full[1:], continuation, strict=True):
        assert actual == expected
    for name, value in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], value, atol=0, rtol=0)
    assert resumed.sched.state_dict() == uninterrupted.sched.state_dict()
    assert torch.equal(torch.get_rng_state(), complete_rng)
    for index, values in uninterrupted.opt.state_dict()["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(
                resumed.opt.state_dict()["state"][index][key], value, atol=0, rtol=0
            )
    final = save_training_state(resumed, tmp_path / "complete", binding)
    assert final["complete_schedule"] is True and final["promotable"] is False
    assert train_fixed_schedule(resumed, recipe, inventory, groups) == []
    with pytest.raises(ValueError, match="must be new"):
        save_training_state(resumed, tmp_path / "complete", binding)


def one_step(trainer, recipe, inventory, groups):
    def stop(step, record):
        raise RuntimeError("one step fixture")

    with pytest.raises(RuntimeError):
        train_fixed_schedule(trainer, recipe, inventory, groups, on_step=stop)


@pytest.mark.parametrize(
    "field", ["source_sha256", "recipe_sha256", "native_weights_sha256", "training_sha256"]
)
def test_resume_rejects_other_run_binding_before_changing_parameters(tmp_path, field):
    recipe, inventory, groups, cfg, binding = fixture(tmp_path)
    original = trainer(recipe, cfg)
    one_step(original, recipe, inventory, groups)
    save_training_state(original, tmp_path / "state", binding)
    resumed = trainer(recipe, cfg)
    before = copy.deepcopy(resumed.model.state_dict())
    changed = copy.deepcopy(binding)
    changed[field] = fingerprint("different")
    with pytest.raises(ValueError, match="another data/model/source/training"):
        load_training_state(resumed, tmp_path / "state", changed)
    for name, value in before.items():
        torch.testing.assert_close(resumed.model.state_dict()[name], value, atol=0, rtol=0)
    assert resumed.step_i == 0 and not resumed.opt.state


def resign(root, filename):
    header = json.loads((root / "manifest.json").read_bytes())
    header["files"][filename] = file_digest(root / filename)
    (root / "manifest.json").write_bytes(canonical(header))


@pytest.mark.parametrize(
    "change",
    [
        "checksum",
        "nan",
        "scheduler",
        "parameter_order",
        "microbatch",
        "moment_nan",
        "optimizer_lr",
        "moment_shape",
        "scheduler_base_lr",
    ],
)
def test_corrupt_or_rechecksummed_invalid_states_fail_before_parameter_copy(tmp_path, change):
    recipe, inventory, groups, cfg, binding = fixture(tmp_path)
    original = trainer(recipe, cfg)
    one_step(original, recipe, inventory, groups)
    root = tmp_path / "state"
    save_training_state(original, root, binding)
    if change == "checksum":
        with (root / "training.pt").open("ab") as stream:
            stream.write(b"corrupt")
    elif change == "nan":
        tensors = load_file(str(root / "trainable.safetensors"))
        next(iter(tensors.values())).flatten()[0] = float("nan")
        save_file(tensors, str(root / "trainable.safetensors"))
        resign(root, "trainable.safetensors")
    else:
        state = torch.load(root / "training.pt", weights_only=True)
        if change == "scheduler":
            state["scheduler"]["last_epoch"] += 1
        elif change == "scheduler_base_lr":
            state["scheduler"]["base_lrs"][0] *= 10
        elif change == "parameter_order":
            state["optimizer_names"][0].reverse()
        elif change == "moment_nan":
            next(iter(state["optimizer"]["state"].values()))["exp_avg"].flatten()[0] = float("nan")
        elif change == "optimizer_lr":
            state["optimizer"]["param_groups"][0]["lr"] *= 3
        elif change == "moment_shape":
            next(iter(state["optimizer"]["state"].values()))["exp_avg"] = torch.zeros(1)
        else:
            state["micro_tokens"] = -1
        torch.save(state, root / "training.pt")
        resign(root, "training.pt")
    resumed = trainer(recipe, cfg)
    before = copy.deepcopy(resumed.model.state_dict())
    with pytest.raises(ValueError):
        load_training_state(resumed, root, binding)
    for name, value in before.items():
        torch.testing.assert_close(resumed.model.state_dict()[name], value, atol=0, rtol=0)
    assert resumed.step_i == 0


def test_complete_direct_run_rejects_unmarked_rows_or_changed_schedule_before_forward(tmp_path):
    recipe, inventory, groups, cfg, _ = fixture(tmp_path)
    original = trainer(recipe, cfg)
    groups[0][0].direct_distillation = False
    with pytest.raises(ValueError, match="every prepared row"):
        train_fixed_schedule(original, recipe, inventory, groups)
    groups[0][0].direct_distillation = True
    original.cfg = replace(cfg, max_train_seconds=0.01)
    with pytest.raises(ValueError, match="without time truncation"):
        train_fixed_schedule(original, recipe, inventory, groups)
    original.cfg = replace(cfg, steps=1)
    with pytest.raises(ValueError, match="complete workload"):
        train_fixed_schedule(original, recipe, inventory, groups)
    assert original.step_i == 0 and not original.opt.state


def test_failed_durable_write_does_not_publish_a_resume_checkpoint(tmp_path, monkeypatch):
    from ayaka.training import direct_state

    recipe, inventory, groups, cfg, binding = fixture(tmp_path)
    original = trainer(recipe, cfg)
    one_step(original, recipe, inventory, groups)

    def failed_sync(fd):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(direct_state.os, "fsync", failed_sync)
    with pytest.raises(OSError, match="simulated storage failure"):
        save_training_state(original, tmp_path / "unpublished", binding)
    assert not (tmp_path / "unpublished").exists()
    assert list(tmp_path.glob(".unpublished-*"))
