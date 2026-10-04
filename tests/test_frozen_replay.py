import copy
import json
from dataclasses import replace

import pytest
import torch

from ayaka.checkpoint import apply_lora
from ayaka.config import tiny_config
from ayaka.data.schema import Candidate, Question, Sample
from ayaka.eval.read_artifact import fingerprint
from ayaka.losses import LossWeights, decision_loss
from ayaka.model.electra import DecisionOutput, ElectraDecisionModel
from ayaka.tokenization import ToyTokenizer
from ayaka.training.batching import collate_items, sample_to_items
from ayaka.training.frozen_replay import attach_base_replay
from ayaka.training.trainer import TrainConfig, Trainer


def setup():
    torch.set_num_threads(1)
    tok = ToyTokenizer()
    cfg = tiny_config(readout="lm", lora_dropout=0.2)
    model = ElectraDecisionModel.from_config(cfg, dtype=torch.float32)
    model.backbone.requires_grad_(False)
    apply_lora(model)
    # Simulate a nonzero adapter without updating the native base. Collection
    # must actually disable it, not assume every step-zero adapter is zero.
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                parameter.normal_(std=0.1)
    weights = LossWeights(pointer_aux=0, gold_nll_with_teacher=True, base_replay=0.2)
    trainer = Trainer(model, tok, TrainConfig(steps=2, loss_weights=weights), "cpu")
    samples = [
        Sample(
            f"Original natural rehearsal input {i}.",
            [
                Question(
                    "q",
                    "choice",
                    "Select the answer.",
                    [Candidate("a", "first"), Candidate("b", "second"), Candidate("c", "third")],
                    {"a": 1},
                )
            ],
            {
                "source_example_id": f"fixture/{i}",
                "source_lineage": f"fixture/{i}",
                "language": "en",
                "data_kind": "natural",
                "split": "train",
            },
        )
        for i in range(2)
    ]
    groups = [
        [replace(item, direct_distillation=True) for item in sample_to_items(s, tok, cfg)]
        for s in samples
    ]
    return trainer, samples, groups


def test_collection_disables_adapter_and_preserves_weights_mode_rng_and_step():
    trainer, samples, groups = setup()
    items = [item for group in groups for item in group]
    with trainer.model.backbone.disable_adapter():
        native = trainer.predict(items, apply_temperature=False)
    student = trainer.predict(items, apply_temperature=False)
    assert not torch.allclose(torch.tensor(native), torch.tensor(student), atol=1e-5)
    trainer.model.train()
    weights = copy.deepcopy(trainer.model.state_dict())
    rng = torch.get_rng_state().clone()
    saved = attach_base_replay(trainer, samples, groups, fingerprint("actual fixture native bytes"))
    assert saved["probabilities"] == native
    assert trainer.model.training and torch.equal(torch.get_rng_state(), rng)
    assert trainer.step_i == 0 and not trainer.opt.state
    for name, value in weights.items():
        torch.testing.assert_close(trainer.model.state_dict()[name], value, atol=0, rtol=0)
    tensors = collate_items(items, trainer.tok.pad_id).to("cpu")
    assert tensors.base_mask.tolist() == [True, True]
    assert trainer.train_step(items)["base_replay_kl"] >= 0


@pytest.mark.parametrize("damage", ["native", "input", "probability", "heldout"])
def test_restored_replay_refuses_other_native_input_or_holdout_before_attachment(damage):
    trainer, samples, groups = setup()
    digest = fingerprint("fixture native bytes")
    saved = attach_base_replay(trainer, samples, groups, digest)
    for group in groups:
        for item in group:
            item.base_probs = None
    if damage == "native":
        saved["header"]["native_weights_sha256"] = fingerprint("other weights")
    elif damage == "input":
        saved["header"]["schema"][0]["input_sha256"] = fingerprint("different input")
    elif damage == "probability":
        saved["probabilities"][0][0] = float("nan")
    else:
        samples[0].metadata["split"] = "test"
    with pytest.raises(ValueError):
        attach_base_replay(trainer, samples, groups, digest, saved=saved)
    assert all(item.base_probs is None for group in groups for item in group)


def test_replay_kl_keeps_gold_nll_detaches_reference_and_reduces_over_all_questions():
    logits = torch.tensor([0.4, -0.2, 0.1, 0.5], dtype=torch.float64, requires_grad=True)
    cu = torch.tensor([0, 2, 4])
    out = DecisionOutput(
        logits, logits, logits * 0, cu, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0])
    )
    target = torch.tensor([1, 0, 0, 1], dtype=torch.float64)
    reference = torch.tensor([0.2, 0.8, 0, 0], dtype=torch.float64, requires_grad=True)
    weights = LossWeights(
        nll=1,
        brier=0,
        rps=0,
        missing=0,
        pointer_aux=0,
        distill=0,
        gold_nll_with_teacher=True,
        base_replay=0.3,
    )
    parts = decision_loss(
        out, target, base_probs=reference, base_mask=torch.tensor([True, False]), weights=weights
    )
    logp = torch.log_softmax(logits.reshape(2, 2), dim=1).flatten()
    gold = -(target * logp).sum() / 2
    kl = (reference[:2] * (reference[:2].log() - logp[:2])).sum()
    torch.testing.assert_close(parts["total"], gold + 0.3 * kl / 2)
    parts["total"].backward()
    assert reference.grad is None and logits.grad is not None
    legacy = decision_loss(out, target, weights=replace(weights, base_replay=0))
    torch.testing.assert_close(legacy["total"], gold)
    assert "base_replay_kl" not in legacy


def test_replay_cannot_leak_into_legacy_training():
    trainer, samples, groups = setup()
    attach_base_replay(trainer, samples, groups, fingerprint("fixture native bytes"))
    groups[0][0].direct_distillation = False
    with pytest.raises(ValueError, match="explicitly direct-distillation"):
        trainer.train_step(groups[0])
    assert trainer.step_i == 0 and not trainer.opt.state


def test_pipeline_replay_resume_uses_original_native_reads_and_matches_continuous_run(tmp_path):
    from ayaka.data.direct_natural import NaturalGoldRegistry
    from ayaka.data.natural_training_v2 import SOURCES, partition_sources
    from ayaka.training import run_direct
    from ayaka.training.direct_bundle import prepare_bundle

    repo, revision, filename, _ = SOURCES["commonsense_qa"]
    rows = [
        {
            "question": f"Original human five-way source case {i}?",
            "choices": {"label": list("ABCDE"), "text": [f"choice {j} from {i}" for j in range(5)]},
            "answerKey": "C",
        }
        for i in range(160)
    ]
    registry = NaturalGoldRegistry(
        {
            "commonsense_qa": {
                "rows": rows,
                "features": {},
                "provenance": {
                    "repo": repo,
                    "revision": revision,
                    "file": filename,
                    "sha256": fingerprint(rows),
                },
            }
        }
    )
    splits, _ = partition_sources(
        registry.sources(), [], fits=lambda _: True, train_limit=4, heldout_limit=2
    )
    cfg = tiny_config(readout="lm", lora_dropout=0.2)
    prepare_bundle(
        tmp_path / "bundle",
        splits,
        ToyTokenizer(),
        cfg,
        {},
        steps=2,
        rows_per_step=2,
        weights=LossWeights(pointer_aux=0, gold_nll_with_teacher=True, base_replay=0.2),
        allow_tiny=True,
        natural_registry=registry,
    )
    full = run_direct.run_pipeline(
        tmp_path / "bundle",
        tmp_path / "full",
        action="train",
        mechanics_only=True,
        checkpoint_every=1,
        natural_registry=registry,
    )
    assert full["reload_probability_parity"]
    saved = json.loads((tmp_path / "full/frozen_base_reads.json").read_bytes())
    resumed = run_direct.run_pipeline(
        tmp_path / "bundle",
        tmp_path / "resume",
        action="train",
        mechanics_only=True,
        checkpoint_every=1,
        resume=tmp_path / "full/state-00000001",
        saved_base_reads=saved,
        natural_registry=registry,
    )
    assert resumed["complete_schedule"]
    from safetensors.torch import load_file

    a = load_file(tmp_path / "full/state-00000002/trainable.safetensors")
    b = load_file(tmp_path / "resume/state-00000002/trainable.safetensors")
    for name in a:
        torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="original frozen-base reads"):
        run_direct.run_pipeline(
            tmp_path / "bundle",
            tmp_path / "absent",
            action="train",
            mechanics_only=True,
            resume=tmp_path / "full/state-00000001",
            natural_registry=registry,
        )
    assert not (tmp_path / "absent").exists()
