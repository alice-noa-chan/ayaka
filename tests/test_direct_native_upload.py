"""Complete uploaded native packages work without repo/cache resolution."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from test_direct_bundle import dataset, resign
from test_direct_swift_bundle import ENCODING
from test_evidence_swift_bridge import tokenizer
from test_native_metadata import REVISION, forbid_cache
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma4ForCausalLM,
    GraniteConfig,
    GraniteForCausalLM,
)
from transformers.utils import CHAT_TEMPLATE_DIR

from ayaka.backbone import tiny_text_config
from ayaka.config import tiny_config
from ayaka.training import direct_bundle, native_snapshot, run_direct
from ayaka.training.direct_audit import audit_snapshot, create_audit_receipt, load_audited_bundle
from ayaka.training.direct_bundle import audit_bundle, local_tokenizer, prepare_bundle
from ayaka.training.native_snapshot import (
    DEPLOYABLE_VERSION,
    inspect_snapshot,
    main,
    verify_snapshot,
)
from ayaka.training.prepare_v2 import canonical, sha256
from ayaka.training.run_direct import run_pipeline


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def package(tmp_path, family="gemma"):
    native = tmp_path / "native"
    torch.manual_seed(73)
    # Save the pristine publisher-shaped LM, before Ayaka detaches/registers
    # its output head. No artificial bias absent from native config is added.
    lm = (
        Gemma4ForCausalLM(tiny_text_config())
        if family == "gemma"
        else GraniteForCausalLM(
            GraniteConfig(
                vocab_size=512,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                logits_scaling=2.5,
                tie_word_embeddings=False,
                max_position_embeddings=2048,
            )
        )
    )
    lm.save_pretrained(native, max_shard_size="40KB")
    tokenizer().save_pretrained(native)
    cfg = replace(
        tiny_config(readout="lm", max_seq_len=2048, lora_dropout=0.05), backbone_revision=REVISION
    )
    return native, cfg


def prepared(tmp_path, family="gemma"):
    native, cfg = package(tmp_path, family)
    root = tmp_path / "bundle"
    prepare_bundle(
        root,
        dataset(),
        local_tokenizer(cfg, allow_tiny=True, native_path=native),
        cfg,
        {},
        steps=2,
        rows_per_step=3,
        seed=19,
        allow_tiny=True,
        input_encoding=ENCODING,
        native_path=native,
    )
    anchor = sha256((root / "manifest.json").read_bytes())
    receipt = tmp_path / "audit.json"
    result = create_audit_receipt(
        root, receipt, expected_manifest_sha256=anchor, allow_tiny=True, native_path=native
    )
    kwargs = {
        "allow_tiny": True,
        "expected_manifest_sha256": anchor,
        "expected_receipt_sha256": result["audit_receipt_sha256"],
    }
    snapshot, _ = inspect_snapshot(
        cfg.backbone, cfg.backbone_revision, path=native, require_tokenizer=True
    )
    return root, native, cfg, receipt, kwargs, snapshot


@pytest.mark.parametrize("family", ["gemma", "granite"])
def test_relocated_upload_full_fast_audit_retains_exact_tokens_and_binding(
    tmp_path, monkeypatch, family
):
    root, native, cfg, receipt, kwargs, record = prepared(tmp_path, family)
    target = tmp_path / "relocated"
    shutil.copytree(native, target)
    forbid_cache(monkeypatch)
    original = audit_snapshot(
        root,
        allow_tiny=True,
        expected_manifest_sha256=kwargs["expected_manifest_sha256"],
        native_path=native,
    )
    moved = audit_snapshot(
        root,
        allow_tiny=True,
        expected_manifest_sha256=kwargs["expected_manifest_sha256"],
        native_path=target,
    )
    fast = load_audited_bundle(root, receipt, native_path=target, **kwargs)
    assert (
        direct_bundle._item_bytes(original.items)
        == direct_bundle._item_bytes(moved.items)
        == direct_bundle._item_bytes(fast.items)
    )
    assert original.binding == moved.binding == fast.binding
    assert original.recipe == moved.recipe == fast.recipe
    assert original.splits == moved.splits == fast.splits
    assert fast.tokenizer.name == cfg.backbone
    assert record["version"] == DEPLOYABLE_VERSION
    assert record["native_metadata"] == fast.recipe["native_metadata"]
    assert verify_snapshot(record, path=target) == target


@pytest.mark.parametrize("family", ["gemma", "granite"])
def test_uploaded_native_full_fast_training_and_resume_have_identical_updates(
    tmp_path, monkeypatch, family
):
    root, native, _, receipt, kwargs, record = prepared(tmp_path, family)
    target = tmp_path / "relocated"
    shutil.copytree(native, target)
    forbid_cache(monkeypatch)
    common = {
        "action": "train",
        "mechanics_only": True,
        "training": {"bf16": False, "log_every": 0, "micro_batch_tokens": 8192},
        "checkpoint_every": 1,
        "expected_bundle_sha256": kwargs["expected_manifest_sha256"],
        "snapshot_record": record,
    }
    full = run_pipeline(root, tmp_path / "full", snapshot_path=native, **common)
    monkeypatch.setattr(direct_bundle, "_prepare", lambda *a, **k: pytest.fail("no regeneration"))
    fast_args = {
        "audit_receipt": receipt,
        "expected_audit_receipt_sha256": kwargs["expected_receipt_sha256"],
    }
    fast = run_pipeline(root, tmp_path / "fast", snapshot_path=target, **common, **fast_args)
    resumed = run_pipeline(
        root,
        tmp_path / "resumed",
        snapshot_path=target,
        resume=tmp_path / "full/state-00000001",
        **common,
        **fast_args,
    )
    assert full["optimizer_steps"] == fast["optimizer_steps"] == resumed["optimizer_steps"] == 2
    states = [
        load_file(tmp_path / name / "state-00000002/trainable.safetensors")
        for name in ("full", "fast", "resumed")
    ]
    for key in states[0]:
        for current in states[1:]:
            torch.testing.assert_close(current[key], states[0][key], atol=0, rtol=0)
    rows = [
        json.loads((tmp_path / name / "dev.json").read_bytes())["rows"]
        for name in ("full", "fast", "resumed")
    ]
    assert (
        [r["probs"] for r in rows[0]]
        == [r["probs"] for r in rows[1]]
        == [r["probs"] for r in rows[2]]
    )
    assert all(x["reload_probability_parity"] for x in (full, fast, resumed))


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize(
    "damage",
    [
        "config_scaling",
        "config_dropout",
        "missing_config",
        "missing_tokenizer",
        "template",
        "preferred_template",
        "decode_flags",
        "embedded_adapter",
    ],
)
def test_upload_drift_fails_before_tokenizer_meta_or_cuda(tmp_path, monkeypatch, fast, damage):
    root, native, _, receipt, kwargs, _ = prepared(tmp_path, "granite")
    if damage in {"config_scaling", "config_dropout", "decode_flags"}:
        path = native / ("tokenizer_config.json" if damage == "decode_flags" else "config.json")
        value = json.loads(path.read_bytes())
        value[
            {
                "config_scaling": "logits_scaling",
                "config_dropout": "attention_dropout",
                "decode_flags": "clean_up_tokenization_spaces",
            }[damage]
        ] = 0.25
        path.write_text(json.dumps(value))
    elif damage.startswith("missing"):
        (native / ("config.json" if damage == "missing_config" else "tokenizer.json")).unlink()
    elif damage == "embedded_adapter":
        (native / "adapter_config.json").write_text("{}")
    elif damage == "preferred_template":
        (native / CHAT_TEMPLATE_DIR).mkdir()
        (native / CHAT_TEMPLATE_DIR / "default.jinja").write_text("changed template")
    else:
        (native / "chat_template.jinja").write_text("changed template")
    forbid_cache(monkeypatch)
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *a, **k: pytest.fail("reject before tokenizer")
    )
    monkeypatch.setattr(
        AutoModelForCausalLM, "from_config", lambda *a, **k: pytest.fail("reject before meta")
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("reject before CUDA"))
    with pytest.raises(ValueError, match="assets changed|native upload requires|embedded adapter"):
        if fast:
            load_audited_bundle(root, receipt, native_path=native, **kwargs)
        else:
            audit_snapshot(root, allow_tiny=True, native_path=native)


@pytest.mark.parametrize(
    "damage",
    ["missing_shard", "legacy_snapshot", "wrong_metadata", "wrong_revision", "no_snapshot"],
)
def test_runner_requires_same_complete_deployable_snapshot_before_weight_load(
    tmp_path, monkeypatch, damage
):
    root, native, _, receipt, kwargs, record = prepared(tmp_path, "granite")
    if damage == "missing_shard":
        next(native.glob("*.safetensors")).unlink()
    elif damage == "legacy_snapshot":
        record, _ = inspect_snapshot("tiny", REVISION, path=native)
    elif damage == "wrong_metadata":
        record["native_metadata"] = {"changed": True}
    elif damage == "wrong_revision":
        record["revision"] = "b" * 40
    else:
        record = None
    monkeypatch.setattr(
        run_direct.ElectraDecisionModel,
        "from_config",
        lambda *a, **k: pytest.fail("weight allocation prohibited"),
    )
    with pytest.raises(ValueError, match="shard inventory|deployable snapshot|immutable model"):
        run_pipeline(
            root,
            tmp_path / "out",
            action="train",
            mechanics_only=True,
            snapshot_path=native,
            snapshot_record=record,
            audit_receipt=receipt,
            expected_bundle_sha256=kwargs["expected_manifest_sha256"],
            expected_audit_receipt_sha256=kwargs["expected_receipt_sha256"],
        )
    assert not (tmp_path / "out").exists()


def test_cpu_runner_audit_checks_supplied_native_weights_without_loading_them(
    tmp_path, monkeypatch
):
    root, native, _, receipt, kwargs, record = prepared(tmp_path, "granite")
    monkeypatch.setattr(
        run_direct.ElectraDecisionModel,
        "from_config",
        lambda *a, **k: pytest.fail("must not load weights"),
    )
    arguments = {
        "action": "audit",
        "mechanics_only": True,
        "snapshot_path": native,
        "audit_receipt": receipt,
        "expected_bundle_sha256": kwargs["expected_manifest_sha256"],
        "expected_audit_receipt_sha256": kwargs["expected_receipt_sha256"],
    }
    metadata_only = run_pipeline(root, None, **arguments)
    complete = run_pipeline(root, None, snapshot_record=record, **arguments)
    assert metadata_only["native_weight_bytes_verified"] is False
    assert (
        complete["native_weight_bytes_verified"] is True
        and complete["model_weights_loaded"] is False
    )
    next(native.glob("*.safetensors")).unlink()
    with pytest.raises(ValueError, match="shard inventory"):
        run_pipeline(root, None, snapshot_record=record, **arguments)


def test_missing_metadata_and_old_bundle_require_preparation_again(tmp_path):
    root, native, _, _, _, _ = prepared(tmp_path)
    recipe = json.loads((root / "recipe.json").read_bytes())
    del recipe["native_metadata"]
    resign(root, "recipe.json", canonical(recipe) + b"\n")
    with pytest.raises(ValueError, match="v6 direct bundle requires"):
        audit_bundle(root, allow_tiny=True, native_path=native)
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest["version"] = "ayaka-direct-bundle-5"
    (root / "manifest.json").write_bytes(canonical(manifest))
    with pytest.raises(ValueError, match="unsupported"):
        audit_bundle(root, allow_tiny=True, native_path=native)


def test_native_mutation_at_audit_exit_cannot_publish_receipt(tmp_path, monkeypatch):
    root, native, _, _, kwargs, _ = prepared(tmp_path, "granite")
    original = direct_bundle.audit_bundle

    def changed(*args, **kw):
        result = original(*args, **kw)
        with (native / "tokenizer_config.json").open("ab") as stream:
            stream.write(b"\n")
        return result

    monkeypatch.setattr(direct_bundle, "audit_bundle", changed)
    destination = tmp_path / "new-audit.json"
    with pytest.raises(ValueError, match="assets changed"):
        create_audit_receipt(
            root,
            destination,
            allow_tiny=True,
            native_path=native,
            expected_manifest_sha256=kwargs["expected_manifest_sha256"],
        )
    assert not destination.exists()


def test_audit_receipt_cannot_invalidate_native_assets_by_writing_inside_package(tmp_path):
    root, native, _, _, kwargs, _ = prepared(tmp_path, "granite")
    destination = native / "cpu-audit.json"
    with pytest.raises(ValueError, match="outside the native loader"):
        create_audit_receipt(
            root,
            destination,
            allow_tiny=True,
            native_path=native,
            expected_manifest_sha256=kwargs["expected_manifest_sha256"],
        )
    assert not destination.exists()


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("asset", ["preferred_template", "embedded_adapter", "preferred_unsharded"])
def test_late_loader_file_additions_during_weight_hashing_are_rejected(
    tmp_path, monkeypatch, verify, asset
):
    native, cfg = package(tmp_path, "granite")
    record, _ = inspect_snapshot(cfg.backbone, REVISION, path=native, require_tokenizer=True)
    original = native_snapshot.file_digest
    added = False

    def changed(path):
        nonlocal added
        result = original(path)
        if Path(path).suffix == ".safetensors" and not added:
            added = True
            if asset == "preferred_template":
                (native / CHAT_TEMPLATE_DIR).mkdir()
                (native / CHAT_TEMPLATE_DIR / "default.jinja").write_text("preferred template")
            elif asset == "embedded_adapter":
                (native / "adapter_config.json").write_text("{}")
            else:
                shutil.copyfile(path, native / "model.safetensors")
        return result

    monkeypatch.setattr(native_snapshot, "file_digest", changed)
    with pytest.raises(ValueError, match="changed during snapshot|embedded adapter"):
        if verify:
            verify_snapshot(record, path=native)
        else:
            inspect_snapshot(cfg.backbone, REVISION, path=native, require_tokenizer=True)


def test_deployable_snapshot_record_cli_stays_outside_native_directory(tmp_path, capsys):
    native, cfg = package(tmp_path, "granite")
    with pytest.raises(ValueError, match="outside the native"):
        main(
            [
                "--repo",
                cfg.backbone,
                "--revision",
                REVISION,
                "--path",
                str(native),
                "--out",
                str(native / "receipt.json"),
            ]
        )
    out = tmp_path / "snapshot.json"
    main(["--repo", cfg.backbone, "--revision", REVISION, "--path", str(native), "--out", str(out)])
    result = json.loads(capsys.readouterr().out)
    assert result["record_sha256"] == sha256(out.read_bytes())
    record = json.loads(out.read_bytes())
    assert (
        record["version"] == DEPLOYABLE_VERSION and verify_snapshot(record, path=native) == native
    )


def test_actual_offline_bundle_audit_snapshot_and_runner_clis(tmp_path):
    native, cfg = package(tmp_path, "granite")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for split, samples in dataset().items():
        (corpus / f"{split}.jsonl").write_bytes(
            b"\n".join(canonical(s.to_json()) for s in samples) + b"\n"
        )
    config = tmp_path / "config.json"
    config.write_bytes(canonical(asdict(cfg)))
    root, receipt, snapshot, cache = [
        tmp_path / name for name in ("bundle", "audit.json", "snapshot.json", "empty-cache")
    ]
    environment = dict(
        os.environ,
        HF_HOME=str(cache),
        HF_HUB_CACHE=str(cache / "hub"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )

    def cli(module, *arguments):
        result = subprocess.run(
            [sys.executable, "-m", f"ayaka.training.{module}", *map(str, arguments)],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    cli(
        "direct_bundle",
        "prepare",
        "--corpus",
        corpus,
        "--config",
        config,
        "--out",
        root,
        "--steps",
        2,
        "--rows-per-step",
        3,
        "--mechanics-only",
        "--native-path",
        native,
        "--input-encoder",
        "swift_canonical",
        "--prompt-variant",
        "labeled",
        "--state-format",
        "compact",
    )
    anchor = sha256((root / "manifest.json").read_bytes())
    cli(
        "direct_bundle",
        "audit",
        "--bundle",
        root,
        "--native-path",
        native,
        "--mechanics-only",
        "--expected-manifest-sha256",
        anchor,
    )
    audited = cli(
        "direct_audit",
        "--bundle",
        root,
        "--out",
        receipt,
        "--native-path",
        native,
        "--mechanics-only",
        "--expected-manifest-sha256",
        anchor,
    )
    cli(
        "native_snapshot",
        "--repo",
        cfg.backbone,
        "--revision",
        REVISION,
        "--path",
        native,
        "--out",
        snapshot,
    )
    result = cli(
        "run_direct",
        "audit",
        "--bundle",
        root,
        "--snapshot-path",
        native,
        "--snapshot-record",
        snapshot,
        "--audit-receipt",
        receipt,
        "--mechanics-only",
        "--expected-bundle-sha256",
        anchor,
        "--expected-audit-receipt-sha256",
        audited["audit_receipt_sha256"],
    )
    assert result["status"] == "audited_cpu_only" and result["optimizer_steps"] == 0
    assert result["native_weight_bytes_verified"] is True and not result["model_weights_loaded"]
    assert not list(cache.rglob("config.json")) and not list(cache.rglob("*.safetensors"))
