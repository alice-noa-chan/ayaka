"""Audit receipt integration without rereading corpus/tokenizer or changing steps."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from test_run_direct import bundle

from ayaka.config import tiny_config
from ayaka.data.reasoning_v2 import SPLITS, curriculum
from ayaka.tokenization import ToyTokenizer
from ayaka.training import direct_audit, direct_bundle, run_direct
from ayaka.training.direct_audit import create_audit_receipt
from ayaka.training.direct_bundle import prepare_bundle
from ayaka.training.prepare_v2 import sha256
from ayaka.training.run_direct import run_pipeline

TRAINING = {"bf16": False, "log_every": 0, "micro_batch_tokens": 8192}


def receipt_kwargs(root, receipt):
    anchor = sha256((root / "manifest.json").read_bytes())
    result = create_audit_receipt(root, receipt, expected_manifest_sha256=anchor, allow_tiny=True)
    return {
        "expected_bundle_sha256": anchor,
        "audit_receipt": receipt,
        "expected_audit_receipt_sha256": result["audit_receipt_sha256"],
    }


def test_receipt_train_and_full_regeneration_have_identical_updates_and_can_resume(
    tmp_path, monkeypatch
):
    root = bundle(tmp_path)
    kwargs = receipt_kwargs(root, tmp_path / "cpu-audit.json")
    full = run_pipeline(
        root,
        tmp_path / "full",
        action="train",
        mechanics_only=True,
        training=TRAINING,
        checkpoint_every=1,
        expected_bundle_sha256=kwargs["expected_bundle_sha256"],
    )

    def no_reread(*args, **kw):
        pytest.fail("frozen execution must not regenerate rows or reload unchecked corpus")

    monkeypatch.setattr(direct_bundle, "_prepare", no_reread)
    monkeypatch.setattr(run_direct, "read_splits", no_reread)
    fast = run_pipeline(
        root,
        tmp_path / "fast",
        action="train",
        mechanics_only=True,
        training=TRAINING,
        checkpoint_every=1,
        **kwargs,
    )
    resumed = run_pipeline(
        root,
        tmp_path / "resumed",
        action="train",
        mechanics_only=True,
        training=TRAINING,
        checkpoint_every=1,
        resume=tmp_path / "full/state-00000001",
        **kwargs,
    )
    assert full["optimizer_steps"] == fast["optimizer_steps"] == resumed["optimizer_steps"] == 2
    tensors = [
        load_file(tmp_path / name / "state-00000002/trainable.safetensors")
        for name in ("full", "fast", "resumed")
    ]
    for name in tensors[0]:
        for actual in tensors[1:]:
            torch.testing.assert_close(actual[name], tensors[0][name], atol=0, rtol=0)
    assert json.loads((tmp_path / "full/dev.json").read_bytes())["rows"] != []
    a, b = [
        json.loads((tmp_path / name / "dev.json").read_bytes())["rows"] for name in ("full", "fast")
    ]
    assert [row["probs"] for row in a] == [row["probs"] for row in b]
    evidence = json.loads((tmp_path / "fast/cpu_audit.json").read_bytes())
    assert evidence["external_receipt_sha256"] == kwargs["expected_audit_receipt_sha256"]
    assert not evidence["model_weights_loaded"] and not evidence["execution_attested"]
    assert json.loads((tmp_path / "fast/kernel_parity.json").read_bytes())["parity_verified"]


@pytest.mark.parametrize("fast", [False, True])
def test_filesystem_replacements_after_validated_return_do_not_change_training_evaluation_or_tokenizer(
    tmp_path, monkeypatch, fast
):
    root = bundle(tmp_path)
    kwargs = receipt_kwargs(root, tmp_path / "cpu-audit.json") if fast else {}
    attribute = "load_audited_bundle" if fast else "audit_snapshot"
    original = getattr(run_direct, attribute)
    captured = {}

    def replace_after_audit(*args, **kw):
        snapshot = original(*args, **kw)
        captured["snapshot"] = snapshot
        for split in ("train", "calibration", "dev"):
            (root / f"{split}.jsonl").write_bytes(b"unverified replacement")
        return snapshot

    monkeypatch.setattr(run_direct, attribute, replace_after_audit)
    original_trainer = run_direct.Trainer

    def checked_trainer(model, tok, cfg, dev):
        assert tok is captured["snapshot"].tokenizer
        return original_trainer(model, tok, cfg, dev)

    monkeypatch.setattr(run_direct, "Trainer", checked_trainer)
    original_tokenizer = direct_bundle.local_tokenizer
    loads = []

    def once(*args, **kw):
        loads.append(True)
        assert len(loads) == 1, "tokenizer must be loaded only for the audit"
        return original_tokenizer(*args, **kw)

    monkeypatch.setattr(direct_bundle, "local_tokenizer", once)
    result = run_pipeline(
        root, tmp_path / "result", action="train", mechanics_only=True, training=TRAINING, **kwargs
    )
    assert result["complete_schedule"] and result["reload_probability_parity"]
    rows = json.loads((tmp_path / "result/dev.json").read_bytes())["rows"]
    expected = [
        sample.metadata["source_example_id"] + "/" + q.id
        for sample in captured["snapshot"].splits["dev"]
        for q in sample.questions
    ]
    assert [row["id"] for row in rows] == expected


def test_duplicate_sample_source_ids_profile_the_correct_original_rows(tmp_path, monkeypatch):
    splits = {}
    for split in SPLITS:
        splits[split] = [sample for sample, _ in curriculum(split, 10)]
        for sample in splits[split]:
            sample.metadata["source_lineage"] = sample.metadata["case_facts_sha256"]
    first, second = splits["train"][:2]
    second.metadata["source_example_id"] = first.metadata["source_example_id"]
    second.questions[0].id += "_distinct"
    root = tmp_path / "bundle"
    prepare_bundle(
        root,
        splits,
        ToyTokenizer(),
        tiny_config(readout="lm", max_seq_len=2048),
        {},
        steps=2,
        rows_per_step=3,
        seed=19,
        allow_tiny=True,
    )
    original = run_direct.profile_production
    checked = []

    def inspect_groups(trainer, stream, samples, inventory, prepare, **kw):
        rows = [prepare(sample) for sample in samples]
        assert samples[0].metadata["source_example_id"] == samples[1].metadata["source_example_id"]
        assert rows[0][0].enc.prefix_ids != rows[1][0].enc.prefix_ids
        for sample, group in zip(samples, rows, strict=True):
            expected = direct_audit.bundles.encode_direct_sample(
                sample, trainer.tok, trainer.model.cfg
            )
            assert direct_bundle._item_bytes(group) == direct_bundle._item_bytes(
                [
                    # The direct preparation marker is added by distillation.
                    replace(item, direct_distillation=True)
                    for item in expected
                ]
            )
        checked.append(True)
        return original(trainer, stream, samples, inventory, prepare, **kw)

    monkeypatch.setattr(run_direct, "profile_production", inspect_groups)
    result = run_pipeline(root, tmp_path / "profile", mechanics_only=True, training=TRAINING)
    assert checked == [True] and result["optimizer_steps"] == 0


@pytest.mark.parametrize("which", ["path", "digest", "missing_bundle_anchor", "wrong_digest"])
def test_unbound_receipts_fail_before_any_native_weight_load_or_output(
    tmp_path, monkeypatch, which
):
    root = bundle(tmp_path)
    kwargs = receipt_kwargs(root, tmp_path / "cpu-audit.json")
    if which == "path":
        kwargs.pop("expected_audit_receipt_sha256")
    elif which == "digest":
        kwargs.pop("audit_receipt")
    elif which == "missing_bundle_anchor":
        kwargs.pop("expected_bundle_sha256")
    else:
        kwargs["expected_audit_receipt_sha256"] = "0" * 64
    monkeypatch.setattr(
        run_direct.AyakaDecisionModel, "from_config", lambda *a, **k: pytest.fail("no weights")
    )
    with pytest.raises(ValueError, match="together|pinned|digest"):
        run_pipeline(root, tmp_path / "absent", action="train", mechanics_only=True, **kwargs)
    assert not (tmp_path / "absent").exists()


def test_real_offline_pipeline_cli_reports_receipt_mode_and_original_manifest_digest(tmp_path):
    root = bundle(tmp_path)
    kwargs = receipt_kwargs(root, tmp_path / "cpu-audit.json")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ayaka.training.run_direct",
            "audit",
            "--bundle",
            str(root),
            "--audit-receipt",
            str(kwargs["audit_receipt"]),
            "--mechanics-only",
            "--expected-bundle-sha256",
            kwargs["expected_bundle_sha256"],
            "--expected-audit-receipt-sha256",
            kwargs["expected_audit_receipt_sha256"],
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["audit_mode"] == "frozen_cpu_receipt" and report["optimizer_steps"] == 0
    assert report["bundle_manifest_sha256"] == kwargs["expected_bundle_sha256"]
