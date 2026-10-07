import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import Gemma4ForCausalLM

from ayaka.backbone import load_text_backbone, native_logits, tiny_text_config
from ayaka.config import tiny_config
from ayaka.model.decision import AyakaDecisionModel
from ayaka.training.native_snapshot import inspect_snapshot, verify_snapshot

REVISION = "a" * 40


def native(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    lm = Gemma4ForCausalLM(tiny_text_config()).eval()
    lm.save_pretrained(tmp_path, max_shard_size="40KB")
    return lm


def test_local_sharded_snapshot_binds_every_actual_header_and_byte(tmp_path, monkeypatch):
    lm = native(tmp_path)
    import huggingface_hub

    def no_network(*args, **kwargs):
        raise AssertionError("no network or HF cache resolution is allowed for an explicit path")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_network)
    record, root = inspect_snapshot("publisher/native", REVISION, path=tmp_path)
    assert root == tmp_path.resolve()
    assert record["network_downloads"] == 0 and record["weights_loaded"] is False
    unique = {tensor.data_ptr(): tensor for tensor in lm.state_dict().values()}
    assert record["stored_weight_elements"] == sum(t.numel() for t in unique.values())
    assert record["tensor_count"] > 0
    assert verify_snapshot(record, path=root) == root
    index = json.loads((root / "model.safetensors.index.json").read_bytes())
    shard = root / next(iter(index["weight_map"].values()))
    weights = load_file(shard)
    next(iter(weights.values())).flatten()[0] += 0.1
    save_file(weights, shard)
    with pytest.raises(ValueError, match="changed since preparation"):
        verify_snapshot(record, path=root)


def test_header_only_snapshot_preserves_identity_without_writable_torch_storage(
    tmp_path, monkeypatch
):
    (tmp_path / "config.json").write_text('{"model_type":"gemma4_unified_text"}')
    save_file(
        {
            "bf16": torch.ones(2, 2, dtype=torch.bfloat16),
            "empty": torch.empty(0),
            "scalar": torch.tensor(1.0),
        },
        tmp_path / "model.safetensors",
    )
    original, _ = inspect_snapshot("publisher/native", REVISION, path=tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("header inspection must not map a writable full-shard TorchStorage")

    monkeypatch.setattr(torch.UntypedStorage, "from_file", forbidden)
    actual, _ = inspect_snapshot("publisher/native", REVISION, path=tmp_path)
    assert actual == original
    assert actual["tensor_count"] == 3
    assert actual["stored_weight_elements"] == 5
    assert actual["dtype_elements"] == {"BF16": 4, "F32": 1}
    assert verify_snapshot(original, path=tmp_path) == tmp_path.resolve()


@pytest.mark.parametrize("damage", ["missing", "unsafe", "wrong_shard", "missing_entry"])
def test_incomplete_or_misbound_weight_maps_fail_without_model_allocation(tmp_path, damage):
    native(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_bytes())
    key = next(iter(index["weight_map"]))
    if damage == "missing":
        (tmp_path / index["weight_map"][key]).unlink()
    elif damage == "unsafe":
        index["weight_map"][key] = "../outside.safetensors"
    elif damage == "wrong_shard":
        index["weight_map"][key] = index["weight_map"][next(reversed(index["weight_map"]))]
    else:
        del index["weight_map"][key]
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        inspect_snapshot("publisher/native", REVISION, path=tmp_path)


def test_strict_offline_native_loader_retains_logits_and_declared_repository(tmp_path):
    lm = native(tmp_path)
    backbone, _ = load_text_backbone(
        str(tmp_path), dtype=torch.float32, local_files_only=True, strict_loading=True
    )
    backbone.eval()
    ids = torch.tensor([[1, 7, 9, 13]])
    with torch.no_grad():
        expected = lm(ids).logits
        actual = native_logits(backbone, backbone(ids).last_hidden_state)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    cfg = replace(
        tiny_config(readout="lm"), backbone="publisher/native", backbone_revision=REVISION
    )
    model = AyakaDecisionModel.from_config(
        cfg,
        dtype=torch.float32,
        backbone_path=str(tmp_path),
        local_files_only=True,
        strict_loading=True,
    )
    assert model.cfg.backbone == "publisher/native" and model.cfg.backbone_revision == REVISION


def test_strict_native_loader_rejects_reinitialized_missing_text_weights(tmp_path):
    native(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_bytes())
    key = next(key for key in index["weight_map"] if key.endswith("q_proj.weight"))
    shard = tmp_path / index["weight_map"].pop(key)
    tensors = load_file(shard)
    del tensors[key]
    save_file(tensors, shard)
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="did not load completely"):
        load_text_backbone(
            str(tmp_path), dtype=torch.float32, local_files_only=True, strict_loading=True
        )
