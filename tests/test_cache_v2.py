import json

import pytest

from ayaka.training import cache_v2


@pytest.mark.parametrize("layout", ["sharded", "single"])
def test_cache_only_uses_pinned_audited_shards_and_records_hashes(tmp_path, monkeypatch, layout):
    from types import SimpleNamespace

    from huggingface_hub.errors import RemoteEntryNotFoundError

    bundle, snapshot = tmp_path / "bundle", tmp_path / "snapshot"
    bundle.mkdir()
    snapshot.mkdir()
    cfg = {"backbone": "approved/model", "backbone_revision": "a" * 40}
    (bundle / "training_config.json").write_text(json.dumps({"model": cfg}))
    (bundle / "model_preflight.json").write_text(
        json.dumps(
            {"license": "apache-2.0", "repo": cfg["backbone"], "revision": cfg["backbone_revision"]}
        )
    )
    (bundle / "manifest.json").write_bytes(b"{}")
    weight_name = "weights.safetensors" if layout == "sharded" else "model.safetensors"
    (snapshot / weight_name).write_bytes(b"fixture")
    index = snapshot / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {"weight_map": {"parameter": "weights.safetensors"}, "metadata": {"total_size": 7}}
        )
    )
    calls = []
    monkeypatch.setattr(cache_v2, "validate_bundle", lambda _: None)
    monkeypatch.setattr(cache_v2, "HF_HUB_CACHE", str(snapshot))

    def get_file(repo, filename, revision):
        if layout == "single" and filename.endswith("index.json"):
            import httpx

            response = httpx.Response(404, request=httpx.Request("GET", "https://example.test"))
            raise RemoteEntryNotFoundError("no sharded index", response=response)
        return str(index)

    monkeypatch.setattr(cache_v2, "hf_hub_download", get_file)
    monkeypatch.setattr(
        cache_v2,
        "HfApi",
        lambda: SimpleNamespace(
            model_info=lambda *args, **kwargs: SimpleNamespace(
                siblings=[SimpleNamespace(rfilename="model.safetensors", size=7)]
            )
        ),
    )

    def download(repo, revision, **kwargs):
        calls.append((repo, revision, kwargs))
        return str(snapshot)

    monkeypatch.setattr(cache_v2, "snapshot_download", download)
    report = cache_v2.prepare_weights(bundle, tmp_path / "cache.json")
    assert calls[0][:2] == (cfg["backbone"], cfg["backbone_revision"])
    assert report["optimizer_steps"] == 0 and report["shards"][weight_name]["bytes"] == 7
    assert len(report["shards"][weight_name]["sha256"]) == 64
    with pytest.raises(ValueError, match="new file"):
        cache_v2.prepare_weights(bundle, tmp_path / "cache.json")
