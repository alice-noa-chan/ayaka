"""CPU header coverage rejects loader failures before pretrained allocation."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import Gemma4Config, Gemma4ForCausalLM, GraniteConfig, GraniteForCausalLM

from ayaka.backbone import tiny_text_config
from ayaka.training.native_snapshot import inspect_snapshot
from ayaka.training.prepare_v2 import canonical, sha256
from scripts.direct_v2 import native_layout

REVISION = "a" * 40


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def native(tmp_path, *, family="gemma", multimodal=False):
    torch.manual_seed(78)
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
                tie_word_embeddings=False,
            )
        )
    )
    root = tmp_path / "native"
    lm.save_pretrained(root)
    if multimodal:
        weights = load_file(root / "model.safetensors")
        weights = {
            key.replace("model.", "model.language_model.", 1): value
            for key, value in weights.items()
        }
        weights["model.vision_embedder.pos_embedding"] = torch.zeros(2, 2, 64)
        weights["model.embed_audio.embedding_projection.weight"] = torch.zeros(64, 8)
        save_file(weights, root / "model.safetensors")
        Gemma4Config(text_config=lm.config.to_dict()).save_pretrained(root)
    return root


def record(root):
    return inspect_snapshot("publisher/native", REVISION, path=root)[0]


@pytest.mark.parametrize(
    "family,multimodal", [("gemma", False), ("gemma", True), ("granite", False)]
)
def test_real_native_headers_cover_meta_parameters_without_tensor_or_gpu_loads(
    tmp_path, monkeypatch, family, multimodal
):
    from transformers import AutoModelForCausalLM

    root = native(tmp_path, family=family, multimodal=multimodal)
    snapshot = record(root)

    def forbidden(*args, **kwargs):
        pytest.fail("no pretrained allocation, CUDA or full safetensors read is allowed")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(Gemma4ForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.UntypedStorage, "from_file", forbidden)
    original_open = native_layout.safe_open

    class HeadersOnly:
        def __init__(self, *args, **kwargs):
            self.handle = original_open(*args, **kwargs)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def keys(self):
            return self.handle.keys()

        def get_slice(self, key):
            view = self.handle.get_slice(key)

            class HeaderSlice:
                get_shape = view.get_shape
                get_dtype = view.get_dtype
                __getitem__ = forbidden

            return HeaderSlice()

        get_tensor = forbidden
        get_tensors = forbidden

    monkeypatch.setattr(native_layout, "safe_open", HeadersOnly)
    before = torch.random.get_rng_state().clone()
    result = native_layout.audit_layout(snapshot, root)
    assert torch.equal(before, torch.random.get_rng_state())
    assert result["materialized_parameter_bytes"] == 0
    assert not result["pretrained_tensors_loaded"] and not result["gpu_allocated"]
    assert (
        result["matched_text_tensors"] + len(result["tied_aliases"]) == result["text_state_entries"]
    )
    assert result["tied_aliases"] == (
        {"lm_head.weight": "model.embed_tokens.weight"} if family == "gemma" else {}
    )
    assert len(result["ignored_auxiliary_tensors"]) == (2 if multimodal else 0)


@pytest.mark.parametrize(
    "damage", ["parameter", "persistent_buffer", "shape", "unknown", "collision"]
)
def test_resigned_invalid_headers_fail_meta_layout_without_loading(tmp_path, damage):
    root = native(tmp_path, multimodal=damage == "collision")
    path = root / "model.safetensors"
    weights = load_file(path)
    if damage == "parameter":
        del weights[next(key for key in weights if key.endswith("q_proj.weight"))]
    elif damage == "persistent_buffer":
        del weights[next(key for key in weights if key.endswith("layer_scalar"))]
    elif damage == "shape":
        key = next(key for key in weights if key.endswith("q_proj.weight"))
        weights[key] = weights[key][:1]
    elif damage == "unknown":
        weights["model.nonexistent_text.weight"] = torch.zeros(1)
    else:
        source = next(key for key in weights if key.endswith("q_proj.weight"))
        weights[source.replace("model.language_model.", "model.", 1)] = weights[source].clone()
    save_file(weights, path)
    with pytest.raises(ValueError, match="missing|shape mismatch|unexpected|collide"):
        native_layout.audit_layout(record(root), root)


def test_input_embedding_cannot_replace_an_untied_output_head(tmp_path):
    root = native(tmp_path, family="granite")
    path = root / "model.safetensors"
    weights = load_file(path)
    del weights["lm_head.weight"]
    save_file(weights, path)
    with pytest.raises(ValueError, match="missing"):
        native_layout.audit_layout(record(root), root)


def test_text_only_config_cannot_hide_unknown_weights_in_multimodal_namespace(tmp_path):
    root = native(tmp_path)
    path = root / "model.safetensors"
    weights = load_file(path)
    weights["model.vision_embedder.extra.weight"] = torch.zeros(1)
    save_file(weights, path)
    with pytest.raises(ValueError, match="unexpected"):
        native_layout.audit_layout(record(root), root)


def test_matching_tensor_converter_requires_a_runtime_check(tmp_path, monkeypatch):
    from transformers import core_model_loading

    root = native(tmp_path)
    original = core_model_loading.rename_source_key

    def conversion(*args, **kwargs):
        target, pattern = original(*args, **kwargs)
        return target, "transform-needed" if target.endswith("q_proj.weight") else pattern

    monkeypatch.setattr(core_model_loading, "rename_source_key", conversion)
    with pytest.raises(ValueError, match="runtime tensor conversion"):
        native_layout.audit_layout(record(root), root)


def test_serialized_tied_duplicates_need_tensor_equality_not_meta_identity(tmp_path):
    root = native(tmp_path)
    path = root / "model.safetensors"
    weights = load_file(path)
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"].clone()
    weights["lm_head.weight"][0, 0] += 1
    save_file(weights, path)
    with pytest.raises(ValueError, match="serialized tied aliases"):
        native_layout.audit_layout(record(root), root)


def test_same_header_data_mutation_during_meta_construction_is_detected(tmp_path, monkeypatch):
    from transformers import AutoConfig

    root = native(tmp_path)
    snapshot = record(root)
    original = AutoConfig.from_pretrained

    def mutate(*args, **kwargs):
        config = original(*args, **kwargs)
        path = root / "model.safetensors"
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            value = stream.read(1)[0]
            stream.seek(-1, 2)
            stream.write(bytes([value ^ 1]))
        return config

    monkeypatch.setattr(AutoConfig, "from_pretrained", mutate)
    with pytest.raises(ValueError, match="changed since preparation"):
        native_layout.audit_layout(snapshot, root)


def test_unknown_loader_transforms_are_not_silently_dropped(tmp_path, monkeypatch):
    from transformers import conversion_mapping

    root = native(tmp_path)
    monkeypatch.setattr(conversion_mapping, "get_model_conversion_mapping", lambda *a: [object()])
    with pytest.raises(ValueError, match="unsupported native loader transformation"):
        native_layout.audit_layout(record(root), root)


def test_cli_pins_snapshot_record_and_refuses_overwriting_or_native_output(tmp_path, capsys):
    root = native(tmp_path)
    source = tmp_path / "snapshot.json"
    source.write_bytes(canonical(record(root)) + b"\n")
    out = tmp_path / "layout.json"
    args = [
        "--snapshot-record",
        str(source),
        "--expected-snapshot-record-sha256",
        sha256(source.read_bytes()),
        "--snapshot-path",
        str(root),
        "--out",
        str(out),
    ]
    native_layout.main(args)
    report = json.loads(out.read_bytes())
    assert report["version"] == native_layout.VERSION
    assert json.loads(capsys.readouterr().out)["report_sha256"] == sha256(out.read_bytes())
    with pytest.raises(ValueError, match="new file"):
        native_layout.main(args)
    with pytest.raises(ValueError, match="new file"):
        native_layout.main([*args[:-1], str(root / "new-report.json")])
    with pytest.raises(ValueError, match="external SHA256"):
        native_layout.read_record(source, "0" * 64)
