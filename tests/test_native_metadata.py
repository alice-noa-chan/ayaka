"""Offline native loader-byte identity, including preferred tokenizer assets."""

import json
import shutil
from dataclasses import replace

import pytest
import torch
from test_evidence_swift_bridge import tokenizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import CHAT_TEMPLATE_DIR

from ayaka.backbone import tiny_text_config
from ayaka.config import tiny_config
from ayaka.training.direct_preflight import inspect_direct_model
from ayaka.training.native_metadata import (
    configuration_binding,
    inspect_metadata,
    local_root,
    verify_metadata,
)

REVISION = "a" * 40


def package(root):
    tiny_text_config().save_pretrained(root)
    tokenizer().save_pretrained(root)
    return replace(tiny_config(readout="lm"), backbone_revision=REVISION)


def forbid_cache(monkeypatch):
    import huggingface_hub

    def fail(*args, **kwargs):
        pytest.fail("explicit native package must not resolve a repository or HF cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)


def test_relocated_complete_metadata_retains_identity_without_cache(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    cfg = package(source)
    record, root = inspect_metadata(cfg.backbone, cfg.backbone_revision, path=source)
    shutil.copytree(source, target)
    forbid_cache(monkeypatch)
    assert verify_metadata(record, cfg.backbone, cfg.backbone_revision, path=target) == target
    assert "chat_template.jinja" in record["files"]
    assert configuration_binding(cfg, path=source)[0] == configuration_binding(cfg, path=target)[0]
    state = torch.random.get_rng_state().clone()
    a = inspect_direct_model(cfg, native_path=root)
    b = inspect_direct_model(cfg, native_path=target, expected_config=a["native_config"])
    assert a == b and a["native_config"]["kind"] == "config_file"
    assert torch.equal(state, torch.random.get_rng_state())
    assert a["weights_loaded"] is False and a["materialized_parameter_bytes"] == 0


@pytest.mark.parametrize("key", ["attention_dropout", "final_logit_softcapping", "rope_parameters"])
def test_same_shape_config_drift_fails_before_autoconfig_or_meta(tmp_path, monkeypatch, key):
    cfg = package(tmp_path)
    binding, _ = configuration_binding(cfg, path=tmp_path)
    path = tmp_path / "config.json"
    value = json.loads(path.read_bytes())
    if key == "rope_parameters":
        value[key]["full_attention"]["rope_theta"] *= 2
    else:
        value[key] = 0.125
    path.write_text(json.dumps(value))
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **k: pytest.fail("too late"))
    monkeypatch.setattr(
        AutoModelForCausalLM, "from_config", lambda *a, **k: pytest.fail("too late")
    )
    with pytest.raises(ValueError, match="bytes changed before architecture"):
        inspect_direct_model(cfg, native_path=tmp_path, expected_config=binding)


@pytest.mark.parametrize(
    "asset",
    [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "tokenizer.5.0.0.json",
        "chat_templates/alternate.jinja",
        f"{CHAT_TEMPLATE_DIR}/default.jinja",
    ],
)
def test_changed_or_newly_preferred_assets_reject_old_binding(tmp_path, asset):
    cfg = package(tmp_path)
    record, _ = inspect_metadata(cfg.backbone, cfg.backbone_revision, path=tmp_path)
    path = tmp_path / asset
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(path.read_bytes() + b"\n" if path.exists() else b"{}")
    with pytest.raises(ValueError, match="assets changed"):
        verify_metadata(record, cfg.backbone, cfg.backbone_revision, path=tmp_path)


@pytest.mark.parametrize("missing", ["config.json", "tokenizer.json", "tokenizer_config.json"])
def test_incomplete_explicit_upload_cannot_fallback(tmp_path, monkeypatch, missing):
    cfg = package(tmp_path)
    (tmp_path / missing).unlink()
    forbid_cache(monkeypatch)
    with pytest.raises(ValueError, match="native upload requires"):
        inspect_metadata(cfg.backbone, cfg.backbone_revision, path=tmp_path)
    with pytest.raises(ValueError, match="refuse cache fallback"):
        local_root(cfg.backbone, cfg.backbone_revision, tmp_path / "absent")


@pytest.mark.parametrize("redirect", ["config", "tokenizer", "missing_versioned"])
def test_redirects_cannot_escape_bound_metadata(tmp_path, monkeypatch, redirect):
    cfg = package(tmp_path)
    forbid_cache(monkeypatch)
    name = "config.json" if redirect == "config" else "tokenizer_config.json"
    path = tmp_path / name
    value = json.loads(path.read_bytes())
    value["configuration_files" if redirect == "config" else "fast_tokenizer_files"] = [
        "tokenizer.5.0.0.json" if redirect == "missing_versioned" else "../outside.5.0.0.json"
    ]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="redirects|versioned tokenizer"):
        inspect_metadata(cfg.backbone, cfg.backbone_revision, path=tmp_path)


def test_bound_versioned_tokenizer_is_inventoried(tmp_path):
    cfg = package(tmp_path)
    shutil.copyfile(tmp_path / "tokenizer.json", tmp_path / "tokenizer.5.0.0.json")
    path = tmp_path / "tokenizer_config.json"
    value = json.loads(path.read_bytes())
    value["fast_tokenizer_files"] = ["tokenizer.5.0.0.json"]
    path.write_text(json.dumps(value))
    record, _ = inspect_metadata(cfg.backbone, cfg.backbone_revision, path=tmp_path)
    assert "tokenizer.5.0.0.json" in record["files"]


def test_actual_preferred_template_override_rejects_existing_binding(tmp_path):
    cfg = package(tmp_path)
    old = AutoTokenizer.from_pretrained(
        str(tmp_path), local_files_only=True, trust_remote_code=False
    )
    record, _ = inspect_metadata(cfg.backbone, cfg.backbone_revision, path=tmp_path)
    directory = tmp_path / CHAT_TEMPLATE_DIR
    directory.mkdir()
    (directory / "default.jinja").write_text("changed loader-preferred template")
    changed = AutoTokenizer.from_pretrained(
        str(tmp_path), local_files_only=True, trust_remote_code=False
    )
    assert old.chat_template != changed.chat_template
    with pytest.raises(ValueError, match="assets changed"):
        verify_metadata(record, cfg.backbone, cfg.backbone_revision, path=tmp_path)


def test_metadata_rejects_unpinned_production_identity_even_for_local_path(tmp_path):
    package(tmp_path)
    with pytest.raises(ValueError, match="immutable revision"):
        inspect_metadata("publisher/native", "main", path=tmp_path)


@pytest.mark.parametrize(
    "asset", ["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"]
)
def test_embedded_adapters_reject_before_model_allocation(tmp_path, monkeypatch, asset):
    cfg = package(tmp_path)
    (tmp_path / asset).write_bytes(b"{}")
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **k: pytest.fail("too late"))
    with pytest.raises(ValueError, match="embedded adapter"):
        inspect_direct_model(cfg, native_path=tmp_path)
    with pytest.raises(ValueError, match="embedded adapter"):
        inspect_metadata(cfg.backbone, cfg.backbone_revision, path=tmp_path)


@pytest.mark.parametrize("name", ["config.json", "tokenizer.json", "tokenizer_config.json"])
def test_case_renamed_required_files_are_not_linux_ready(tmp_path, name):
    cfg = package(tmp_path)
    (tmp_path / name).rename(tmp_path / name.capitalize())
    with pytest.raises(ValueError, match="native upload requires"):
        inspect_metadata(cfg.backbone, cfg.backbone_revision, path=tmp_path)
