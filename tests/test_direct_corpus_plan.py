import copy
import json
from collections import Counter
from dataclasses import asdict

import pytest
from test_direct_bundle import dataset
from test_direct_natural import raw_registry

from ayaka.config import tiny_config
from ayaka.data.natural_training_v2 import SOURCES, partition_sources
from ayaka.data.reasoning_v2 import SPLITS
from ayaka.eval.read_artifact import fingerprint
from ayaka.tokenization import ToyTokenizer
from ayaka.training import direct_audit, direct_bundle
from ayaka.training.direct_corpus_plan import (
    MARKER,
    SELECTION,
    VERSION,
    plan_sha256,
    validate_plan,
    whole_epochs,
)
from ayaka.training.prepare_v2 import canonical, sha256
from ayaka.training.swift_direct import normalize_input_encoding
from ayaka.training.workload import scheduled_batches


def planned_inputs():
    registry, splits = raw_registry(), dataset()
    quotas = {source: dict.fromkeys(SPLITS, 1) for source in SOURCES}
    natural, _ = partition_sources(registry.sources(), [], fits=lambda _: True, quotas=quotas)
    for split in SPLITS:
        splits[split] += natural[split]
    cfg, tok = tiny_config(readout="lm", max_seq_len=2048), ToyTokenizer()
    encoding = normalize_input_encoding(None)
    plan = {
        "version": VERSION,
        "settings": {
            "natural_sample_quotas": quotas,
            "authored_per_type": dict.fromkeys(SPLITS, 10),
            "epochs": 2,
            "rows_per_step": 8,
            "seed": 7,
            "minimum_english_question_fraction": 0.9,
        },
        "assets": {
            "model_sha256": fingerprint(asdict(cfg)),
            "tokenizer_sha256": direct_bundle._tokenizer_identity(tok, cfg),
            "native_metadata_sha256": fingerprint(None),
            "input_encoding_sha256": fingerprint(encoding),
            "gold_sources_sha256": fingerprint(registry.binding),
            "public_files": {"test.jsonl": fingerprint("test-only benchmark inventory")},
            "reserved": {
                "scope": "draft_without_prior_private_inventory",
                "manifest_sha256": None,
                "files": {},
            },
        },
        "selection_policy": copy.deepcopy(SELECTION),
    }
    for rows in splits.values():
        for sample in rows:
            sample.metadata[MARKER] = plan_sha256(plan)
    return splits, cfg, tok, registry, plan


def prepare(root, inputs):
    splits, cfg, tok, registry, plan = inputs
    return direct_bundle.prepare_bundle(
        root,
        splits,
        tok,
        cfg,
        {},
        steps=10,
        rows_per_step=8,
        seed=7,
        allow_tiny=True,
        natural_registry=registry,
        corpus_plan=plan,
    )


@pytest.mark.parametrize(
    "damage",
    [
        "version",
        "unknown",
        "bool_quota",
        "missing_source",
        "bool_epoch",
        "nan_fraction",
        "policy",
        "reserved",
    ],
)
def test_input_plan_is_strict_and_never_coerces_settings(damage):
    plan = planned_inputs()[-1]
    if damage == "version":
        plan["version"] = "future"
    elif damage == "unknown":
        plan["verified_rows"] = 123
    elif damage == "bool_quota":
        plan["settings"]["natural_sample_quotas"]["helpsteer2"]["train"] = True
    elif damage == "missing_source":
        del plan["settings"]["natural_sample_quotas"]["massive_ko"]
    elif damage == "bool_epoch":
        plan["settings"]["epochs"] = True
    elif damage == "nan_fraction":
        plan["settings"]["minimum_english_question_fraction"] = float("nan")
    elif damage == "policy":
        plan["selection_policy"]["context"] = "truncate"
    else:
        plan["assets"]["reserved"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_planned_full_fast_and_resume_share_exact_two_epoch_contract(tmp_path, monkeypatch):
    inputs = planned_inputs()
    root = tmp_path / "bundle"
    prepare(root, inputs)
    anchor = sha256((root / "manifest.json").read_bytes())
    full = direct_audit.audit_snapshot(
        root, allow_tiny=True, natural_registry=inputs[3], expected_manifest_sha256=anchor
    )
    contract = full.binding["corpus_contract"]
    assert contract["whole_epochs"]["rows"] == 40
    assert contract["whole_epochs"]["min_visits"] == 2
    assert contract["splits"]["train"]["language_questions"] == {"en": 36, "ko": 2, "ja": 2}
    receipt = tmp_path / "audit.json"
    receipt.write_bytes(canonical(full.binding) + b"\n")
    monkeypatch.setattr(
        "ayaka.data.direct_natural.local_raw_sources",
        lambda: pytest.fail("fast load opened raw sources"),
    )
    monkeypatch.setattr(
        direct_bundle, "_gold_verifier", lambda *a, **k: pytest.fail("fast load regenerated gold")
    )
    monkeypatch.setattr(
        direct_bundle,
        "encode_direct_sample",
        lambda *a, **k: pytest.fail("fast load rerendered input"),
    )
    fast = direct_audit.load_audited_bundle(
        root,
        receipt,
        allow_tiny=True,
        expected_manifest_sha256=anchor,
        expected_receipt_sha256=sha256(receipt.read_bytes()),
    )
    assert fast.binding == full.binding
    all_batches = list(direct_bundle.training_batches(fast.recipe, fast.inventory, fast.groups))
    assert len(all_batches) == 10
    visits = Counter(
        pair
        for batch in scheduled_batches(fast.inventory, **fast.recipe["schedule"])
        for pair in batch
    )
    assert len(visits) == 40 and set(visits.values()) == {2}
    resumed = list(
        direct_bundle.training_batches(fast.recipe, fast.inventory, fast.groups, start_step=3)
    )
    assert resumed == all_batches[3:]
    assert not (root / "test.jsonl").exists()
    opaque = json.loads((root / "test_commitment.json").read_bytes())
    assert opaque["corpus_contract"]["summary"] == contract["splits"]["test"]
    assert "questions" not in opaque["members"][0]
    replaced = list(fast.groups)
    replaced[1] = replaced[0]
    with pytest.raises(ValueError, match="actual corpus training groups"):
        list(direct_bundle.training_batches(fast.recipe, fast.inventory, replaced))


@pytest.mark.parametrize(
    "damage",
    [
        "train_marker",
        "test_marker",
        "no_plan",
        "input",
        "quota",
        "language",
        "question_loss",
        "seed",
        "steps",
    ],
)
def test_prepare_rejects_partial_or_different_contract_before_creating_output(tmp_path, damage):
    inputs = list(planned_inputs())
    splits, cfg, tok, registry, plan = inputs
    steps, seed = 10, 7
    if damage == "train_marker":
        del splits["train"][0].metadata[MARKER]
    elif damage == "test_marker":
        del splits["test"][0].metadata[MARKER]
    elif damage == "no_plan":
        plan = None
    elif damage == "input":
        plan["assets"]["input_encoding_sha256"] = "0" * 64
    elif damage == "quota":
        splits["train"].pop()
    elif damage == "language":
        splits["train"][0].metadata["language"] = "ko"
    elif damage == "question_loss":
        next(s for s in splits["train"] if s.metadata["source"] == "helpsteer2").questions.pop()
    elif damage == "seed":
        seed = 9
    else:
        steps = 9
    root = tmp_path / "absent"
    with pytest.raises(ValueError):
        direct_bundle.prepare_bundle(
            root,
            splits,
            tok,
            cfg,
            {},
            steps=steps,
            rows_per_step=8,
            seed=seed,
            allow_tiny=True,
            natural_registry=registry,
            corpus_plan=plan,
        )
    assert not root.exists() and not root.with_name("absent-holdout").exists()


def test_whole_epochs_can_cross_batch_boundaries_but_cannot_weight_or_repeat_tail():
    plan = planned_inputs()[-1]
    plan["settings"].update(epochs=2, rows_per_step=8)
    inventory = [{"language": "en", "rows": [{}] * 5}, {"language": "ko", "rows": [{}] * 7}]
    schedule = {"steps": 3, "rows_per_step": 8, "seed": 7}
    assert whole_epochs(plan, inventory, schedule)["rows"] == 12
    with pytest.raises(ValueError, match="weighted"):
        whole_epochs(plan, inventory, schedule, weights={"en": 2, "ko": 1})
    plan["settings"]["epochs"] = 1
    with pytest.raises(ValueError, match="integer batches"):
        whole_epochs(plan, inventory, schedule)


@pytest.mark.parametrize(
    "damage",
    ["recipe_only", "plan_sha", "test_sources", "test_languages", "receipt_only", "report_only"],
)
def test_fast_consumer_rejects_resigned_internal_contract_inconsistency(tmp_path, damage):
    inputs, root = planned_inputs(), tmp_path / "bundle"
    prepare(root, inputs)
    full = direct_audit.audit_snapshot(root, allow_tiny=True, natural_registry=inputs[3])
    binding = full.binding
    recipe = json.loads((root / "recipe.json").read_bytes())
    if damage == "recipe_only":
        del recipe["corpus_plan"]
        del recipe["corpus_plan_sha256"]
    elif damage == "plan_sha":
        recipe["corpus_plan_sha256"] = "0" * 64
    elif damage == "test_sources":
        commitment = json.loads((root / "test_commitment.json").read_bytes())
        summary = commitment["corpus_contract"]["summary"]
        summary["sources"]["massive_ko"]["samples"] = 2
        (root / "test_commitment.json").write_bytes(canonical(commitment) + b"\n")
    elif damage == "test_languages":
        commitment = json.loads((root / "test_commitment.json").read_bytes())
        commitment["counts"]["languages"] = {"ko": commitment["counts"]["samples"]}
        (root / "test_commitment.json").write_bytes(canonical(commitment) + b"\n")
    elif damage == "receipt_only":
        del binding["corpus_contract"]
    else:
        preparation = json.loads((root / "preparation.json").read_bytes())
        del preparation["corpus_contract"]
        (root / "preparation.json").write_bytes(canonical(preparation) + b"\n")
    (root / "recipe.json").write_bytes(canonical(recipe) + b"\n")
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest["files"] = {name: sha256((root / name).read_bytes()) for name in manifest["files"]}
    (root / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    anchor = sha256((root / "manifest.json").read_bytes())
    binding["bundle_manifest_sha256"] = anchor
    if damage == "test_languages":
        with pytest.raises(ValueError, match="opaque test corpus"):
            direct_audit.audit_snapshot(root, allow_tiny=True, natural_registry=inputs[3])
    receipt = tmp_path / "audit.json"
    receipt.write_bytes(canonical(binding) + b"\n")
    with pytest.raises(ValueError, match="corpus|plan"):
        direct_audit.load_audited_bundle(
            root,
            receipt,
            allow_tiny=True,
            expected_manifest_sha256=anchor,
            expected_receipt_sha256=sha256(receipt.read_bytes()),
        )
