"""Gold alignment and real causal prompt permutations, without pretrained runs."""

import random
from dataclasses import replace

import pytest
import torch
from test_evidence_swift_bridge import QUESTIONS, native_model, tokenizer

from ayaka.eval.read_artifact import fingerprint
from ayaka.model.evidence import EvidenceResidualHead
from ayaka.training.evidence_objective import evidence_loss
from ayaka.training.evidence_permutation import (
    dev_permutation_plan,
    extract_permuted_evidence_features,
    permutation_diagnostics,
    prepare_permuted_evidence_inputs,
)


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


GOLD = [{"false": 0.2, "true": 0.8}, "it", {"-2": 0.25, "3": 0.75}]


def view(orders=None, *, split="train", epoch=0, gold=GOLD):
    return prepare_permuted_evidence_inputs(
        "Finance approved the policy. Details x",
        QUESTIONS,
        gold,
        tokenizer(),
        split=split,
        record_id="family/document/0001",
        epoch=epoch,
        display_orders=orders,
    )


def test_train_augmentation_is_reproducible_epoch_specific_gold_independent_and_rng_isolated():
    rng = random.getstate()
    a, b = view(epoch=3), view(epoch=3)
    assert a.binding_sha256 == b.binding_sha256
    assert a.prepared.inputs == b.prepared.inputs
    assert random.getstate() == rng
    other_gold = ["false", "sales", "10"]
    c = view(epoch=3, gold=other_gold)
    assert c.prepared.inputs == a.prepared.inputs
    assert c.display_to_canonical == a.display_to_canonical
    assert len({view(epoch=e).display_to_canonical for e in range(8)}) > 1


def test_rerendered_permutation_carries_gold_and_score_meaning_into_real_head_training():
    orders = [(1, 0), (1, 0), (2, 0, 1)]
    augmented = view(orders)
    canonical = view([(0, 1), (0, 1), (0, 1, 2)])
    assert augmented.prepared.inputs.prefix_ids == canonical.prepared.inputs.prefix_ids
    assert (
        augmented.prepared.inputs.questions[1].suffix_ids
        != canonical.prepared.inputs.questions[1].suffix_ids
    )
    assert augmented.prepared.recipe["questions"][0]["labels"] == ["true", "false"]
    assert augmented.prepared.recipe["questions"][2]["labels"] == ["10", "-2", "3"]
    _, text = native_model("granite")
    features = extract_permuted_evidence_features(text, augmented, cache_strategy="copy_on_write")
    supervision = augmented.supervision(features)
    assert supervision["targets"][0, 0].tolist() == [0.8, 0.2, 0.0]
    assert supervision["targets"][0, 1].tolist() == [1.0, 0.0, 0.0]
    assert supervision["targets"][0, 2].tolist() == [0.0, 0.25, 0.75]
    assert supervision["ordinals"][0, 2].tolist() == [10, -2, 3]
    head = EvidenceResidualHead(32, dim=16, heads=4)
    output = head(**features.head_inputs())
    parts = evidence_loss(output, primitive=features.tensors["primitive"], **supervision)
    parts["total"].backward()
    assert head.correction[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in text.parameters())
    other = extract_permuted_evidence_features(text, canonical)
    with pytest.raises(ValueError, match="exact prompt"):
        augmented.supervision(other)


def test_default_dev_plan_is_bounded_deduplicated_and_preserves_semantic_probabilities():
    orders = dev_permutation_plan(QUESTIONS)
    assert len(orders) == 3
    assert len(dev_permutation_plan([QUESTIONS[0]])) == 2
    prepared = [view(order, split="dev") for order in orders]
    canonical = [[0.4, 0.6], [0.1, 0.9], [0.2, 0.5, 0.3]]
    displayed = [
        [[canonical[qi][i] for i in order] for qi, order in enumerate(v.display_to_canonical)]
        for v in prepared
    ]
    report = permutation_diagnostics(prepared, displayed)
    assert all(row["max_pairwise_total_variation"] == 0 for row in report["rows"])
    assert all(row["argmax_flips_from_canonical"] == 0 for row in report["rows"])
    assert report["selected_view"] is None and not report["probability_ensemble"]
    assert report["promotable"] is False


def test_diagnostics_reports_causal_changes_and_not_best_view_accuracy():
    prepared = [view(order, split="dev") for order in dev_permutation_plan(QUESTIONS)]
    displayed = [
        [[0.9, 0.1], [0.8, 0.2], [0.8, 0.1, 0.1]],
        [[0.9, 0.1], [0.8, 0.2], [0.8, 0.1, 0.1]],
        [[0.1, 0.9], [0.2, 0.8], [0.1, 0.1, 0.8]],
    ]
    report = permutation_diagnostics(prepared, displayed)
    assert report["rows"][0]["max_pairwise_total_variation"] == pytest.approx(0.8)
    assert report["rows"][0]["argmax_flips_from_canonical"] == 1
    assert (
        report["rows"][2]["metrics_range"]["rps"][1] > report["rows"][2]["metrics_range"]["rps"][0]
    )
    assert report["rows"][0]["canonical_probs"] == [0.9, 0.1]


def test_actual_causal_backbone_is_not_declared_permutation_invariant():
    prepared = [view(order, split="dev") for order in dev_permutation_plan(QUESTIONS)]
    _, text = native_model("granite")
    probabilities = []
    for v in prepared:
        feature = extract_permuted_evidence_features(text, v)
        probabilities.append(
            [
                feature.tensors["native_logits"][0, qi, : len(labels)].softmax(-1).tolist()
                for qi, labels in enumerate(v.canonical_labels)
            ]
        )
    report = permutation_diagnostics(prepared, probabilities)
    assert any(row["max_pairwise_total_variation"] > 1e-5 for row in report["rows"])


@pytest.mark.parametrize("bad", ["test", "implicit_dev", "duplicate", "bool", "missing", "gold"])
def test_augmentation_rejects_invalid_role_orders_or_gold(bad):
    options = {"split": "train", "orders": None, "gold": GOLD}
    if bad == "test":
        options["split"] = "test"
    elif bad == "implicit_dev":
        options["split"] = "dev"
    elif bad == "gold":
        options["gold"] = ["absent", "it", "3"]
    else:
        options["orders"] = [(0, 1), (0, 1), (0, 1, 2)]
        options["orders"][0] = (
            (0, 0) if bad == "duplicate" else ((False, True) if bad == "bool" else (0,))
        )
    with pytest.raises(ValueError):
        view(**options)


@pytest.mark.parametrize(
    "bad",
    [
        "missing_view",
        "missing_last",
        "duplicate_view",
        "gold",
        "recipe",
        "train",
        "nan",
        "partial",
        "question",
    ],
)
def test_dev_diagnostics_rejects_partial_mismatched_or_selected_views(bad):
    views = [view(order, split="dev") for order in dev_permutation_plan(QUESTIONS)]
    probs = [[[0.5, 0.5], [0.5, 0.5], [0.25, 0.5, 0.25]] for _ in views]
    if bad == "missing_view":
        views = views[1:]
        probs = probs[1:]
    elif bad == "missing_last":
        views = views[:-1]
        probs = probs[:-1]
    elif bad == "duplicate_view":
        views[1] = views[0]
    elif bad == "gold":
        orders = views[1].display_to_canonical
        views[1] = view(orders, split="dev", gold=["true", "sales", "10"])
    elif bad == "recipe":
        views[1].prepared.recipe["prompt_variant"] = "rules"
    elif bad == "train":
        views[1] = view(views[1].display_to_canonical)
    elif bad == "nan":
        probs[1][0] = [float("nan"), 0.5]
    elif bad == "question":
        questions = [dict(q) for q in QUESTIONS]
        questions[0]["instructions"] = "Different instruction?"
        views[1] = prepare_permuted_evidence_inputs(
            "Finance approved the policy. Details x",
            questions,
            GOLD,
            tokenizer(),
            split="dev",
            record_id="family/document/0001",
            display_orders=views[1].display_to_canonical,
        )
    else:
        probs[1] = probs[1][:-1]
    with pytest.raises(ValueError):
        permutation_diagnostics(views, probs)


def test_gold_binding_mutation_fails_before_forward():
    original = view()
    changed = replace(original, targets=((1.0, 0.0), *original.targets[1:]))
    assert fingerprint(changed.binding()) != changed.binding_sha256
    with pytest.raises(ValueError, match="binding changed"):
        extract_permuted_evidence_features(None, changed)
