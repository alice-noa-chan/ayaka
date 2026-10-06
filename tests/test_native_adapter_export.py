"""Real HFReader loading, not a mocked loader or detached self-reload."""

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch
from test_checkpoint_serving_inputs import ENCODING, QUESTIONS, STATE, setup_checkpoint

from ayaka import adapter_export
from ayaka.adapter_export import RECEIPT, export_hf_adapter
from ayaka.input_contract import serving_question
from ayaka.serve import parse_question
from ayaka.swift.prompt import render_question
from ayaka.swift.readers import HFReader
from ayaka.training.swift_direct import swift_question, swift_sample_to_items
from ayaka.training.trainer import TrainConfig, Trainer


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("family", ["granite", "gemma"])
def test_real_full_hf_reader_loads_every_lora_and_matches_trainer_raw(tmp_path, family):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    from ayaka.data.schema import Sample

    source, _, tok, checkpoint, meta = setup_checkpoint(tmp_path, family)
    output = tmp_path / "portable"
    receipt = export_hf_adapter(checkpoint, output)
    weights = load_file(str(output / "adapter_model.safetensors"))
    assert any(tensor.count_nonzero() for key, tensor in weights.items() if ".lora_B." in key)
    assert receipt["tensor_count"] == len(weights) > 1
    assert receipt["calibration_applied"] is False and receipt["native_weights_loaded"] is False
    assert receipt["source_temperature"] == source.temperature.tolist()
    reader = HFReader(source.cfg.backbone, device="cpu", dtype="float32", lora_path=str(output))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reader._load()
    assert not any(
        "missing adapter keys" in str(w.message).lower() or "unexpected" in str(w.message).lower()
        for w in caught
    )
    loaded = get_peft_model_state_dict(reader.model, save_embedding_layers=False)
    from ayaka.swift.readers import _read_forward_kwargs

    assert _read_forward_kwargs(reader.model) == {"logits_to_keep": 1, "use_cache": False}
    assert set(loaded) == set(weights)
    for key in weights:
        assert torch.equal(loaded[key].cpu(), weights[key])
    exported_config = json.loads((output / "adapter_config.json").read_text())
    assert exported_config["auto_mapping"]["base_model_class"].endswith("ForCausalLM")
    questions = [
        serving_question(parse_question(q)[0].view(), i) for i, q in enumerate(QUESTIONS.values())
    ]
    items = swift_sample_to_items(
        Sample(STATE, questions),
        tok,
        source.cfg,
        **{k: v for k, v in ENCODING.items() if k != "encoder"},
    )
    trainer = Trainer(source, tok, TrainConfig(steps=1, bf16=False), "cpu")
    probabilities, logits = trainer.predict(items, apply_temperature=False, return_logits=True)
    calibrated = trainer.predict(items, apply_temperature=True)
    assert any(
        max(abs(x - y) for x, y in zip(a, b, strict=True)) > 1e-5
        for a, b in zip(probabilities, calibrated, strict=True)
    )
    for question, item, p, raw in zip(questions, items, probabilities, logits, strict=True):
        _, wire, semantic = swift_question(question)
        messages, mapping = render_question(
            STATE, wire, prompt_variant="labeled", state_format="compact"
        )
        actual = reader.read(messages, list(mapping))
        order = [
            list(mapping.values()).index(label)
            for label in [
                str(c.ordinal) if question.type == "score" else c.id for c in question.candidates
            ]
        ]
        assert item.enc.prefix_ids + item.enc.rendered.suffix_ids == actual.input_token_ids
        assert [list(actual.letter_probs.values())[i] for i in order] == pytest.approx(p, abs=3e-6)
        assert [list(actual.letter_log_masses.values())[i] for i in order] == pytest.approx(
            raw, abs=3e-5
        )
    second = tmp_path / "portable-again"
    assert export_hf_adapter(checkpoint, second) == receipt
    for name in ("adapter_model.safetensors", "adapter_config.json", RECEIPT):
        assert (second / name).read_bytes() == (output / name).read_bytes()


@pytest.mark.parametrize("family", ["granite", "gemma"])
def test_actual_peft_auto_loader_uses_full_native_class(tmp_path, family, monkeypatch):
    from peft import AutoPeftModelForCausalLM, get_peft_model_state_dict
    from safetensors.torch import load_file

    _, model, _, checkpoint, _ = setup_checkpoint(tmp_path, family)
    output = tmp_path / "portable"
    export_hf_adapter(checkpoint, output)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    loaded = AutoPeftModelForCausalLM.from_pretrained(
        str(output), local_files_only=True, dtype=torch.float32
    ).eval()
    expected = load_file(str(output / "adapter_model.safetensors"))
    actual = get_peft_model_state_dict(loaded, save_embedding_layers=False)
    assert set(actual) == set(expected)
    for name, tensor in expected.items():
        assert torch.equal(actual[name].cpu(), tensor)


def test_full_gemma_audio_meta_layout_selects_language_only(tmp_path):
    from dataclasses import replace

    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import Gemma4AudioConfig, Gemma4Config

    from ayaka.backbone import tiny_text_config

    _, model, _, checkpoint, _ = setup_checkpoint(tmp_path, "gemma")
    native_path = tmp_path / "full-native-config"
    config = Gemma4Config(
        text_config=tiny_text_config(),
        audio_config=Gemma4AudioConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            output_proj_dims=64,
            subsampling_conv_channels=[8, 8],
        ),
    )
    config.save_pretrained(native_path)
    cfg = replace(model.cfg, backbone=str(native_path))
    prefix, shapes, exported, _ = adapter_export._native_layout(
        cfg, checkpoint / "adapter", str(native_path)
    )
    assert prefix == "model.language_model"
    assert shapes and all("model.language_model." in name for name in shapes)
    assert not any("audio" in name or "vision" in name for name in shapes)
    assert exported["auto_mapping"]["base_model_class"] == "Gemma4ForConditionalGeneration"
    # PEFT may compress fully qualified target patterns. Reloading its saved
    # config must still select exactly these text tensors, never audio targets.
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        config._name_or_path = str(native_path)
        reloaded = get_peft_model(AutoModelForCausalLM.from_config(config), LoraConfig(**exported))
        actual = get_peft_model_state_dict(reloaded, save_embedding_layers=False)
    assert {key: tuple(tensor.shape) for key, tensor in actual.items()} == shapes


@pytest.mark.parametrize("change", ["metadata", "preferred_head"])
def test_source_change_during_conversion_is_rejected(tmp_path, monkeypatch, change):
    _, _, _, checkpoint, meta = setup_checkpoint(tmp_path)
    if change == "metadata":
        original = adapter_export.read_contract

        def read_then_change(*args, **kwargs):
            result = original(*args, **kwargs)
            meta["changed_after_capture"] = True
            (checkpoint / "meta.json").write_text(json.dumps(meta))
            return result

        monkeypatch.setattr(adapter_export, "read_contract", read_then_change)
    else:
        from ayaka.checkpoint import save_head

        original = adapter_export.load_head
        (checkpoint / "head.safetensors").unlink()

        def load_then_add_preferred(*args, **kwargs):
            result = original(*args, **kwargs)
            save_head(result, str(checkpoint), pickle=False)
            return result

        monkeypatch.setattr(adapter_export, "load_head", load_then_add_preferred)
    output = tmp_path / "changed-export"
    with pytest.raises(ValueError, match="source checkpoint changed"):
        export_hf_adapter(checkpoint, output)
    assert not output.exists()


@pytest.mark.parametrize("defect", ["missing_A", "missing_pair", "shape", "nan", "foreign"])
def test_incomplete_or_mismatched_adapters_fail_before_output(tmp_path, defect):
    from safetensors.torch import load_file, save_file

    _, _, _, checkpoint, _ = setup_checkpoint(tmp_path)
    path = checkpoint / "adapter" / "adapter_model.safetensors"
    weights = load_file(str(path))
    first = next(key for key in weights if key.endswith("lora_A.weight"))
    if defect == "missing_A":
        weights.pop(first)
    elif defect == "missing_pair":
        weights.pop(first)
        weights.pop(first.replace("lora_A", "lora_B"))
    elif defect == "shape":
        weights[first] = weights[first][0]
    elif defect == "nan":
        weights[first][0, 0] = float("nan")
    else:
        weights["unrelated.weight"] = torch.ones(1)
    save_file(weights, str(path))
    output = tmp_path / "bad-export"
    with pytest.raises(ValueError):
        export_hf_adapter(checkpoint, output)
    assert not output.exists()


def test_output_is_new_and_raw_only_and_bad_contract_rejected(tmp_path):
    _, _, _, checkpoint, _ = setup_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="outside|new"):
        export_hf_adapter(checkpoint, checkpoint / "portable")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="new"):
        export_hf_adapter(checkpoint, existing)
    config_path = checkpoint / "ayaka_config.json"
    config = json.loads(config_path.read_text())
    config["readout"] = "hybrid"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        export_hf_adapter(checkpoint, tmp_path / "bad-contract")


def test_explicit_native_override_requires_local_directory_before_layout(tmp_path, monkeypatch):
    _, _, _, checkpoint, _ = setup_checkpoint(tmp_path)

    def forbid(*args, **kwargs):
        pytest.fail("invalid local native override reached architecture loading")

    monkeypatch.setattr(adapter_export, "_native_layout", forbid)
    for override in ("", str(tmp_path / "missing"), "google/gemma-4-12B-it"):
        with pytest.raises(ValueError, match="local native LM directory"):
            export_hf_adapter(checkpoint, tmp_path / "invalid-override", backbone_path=override)


def test_offline_cli_exports_actual_adapter_and_receipt(tmp_path):
    _, _, _, checkpoint, _ = setup_checkpoint(tmp_path)
    output = tmp_path / "cli-export"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ayaka.adapter_export",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["gpu_seconds"] == 0 and report["native_weights_loaded"] is False
    assert report["calibration_applied"] is False
    assert json.loads((output / RECEIPT).read_text())["tensor_count"] == report["tensor_count"]
