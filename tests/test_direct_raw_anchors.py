import gzip
import json
from contextlib import contextmanager

import pytest

from ayaka.data import direct_natural as module
from ayaka.training.direct_state import file_digest


def cached_raw_fixture(tmp_path, monkeypatch):
    path = tmp_path / "train.jsonl.gz"
    row = {
        "prompt": "original prompt",
        "response": "original response",
        "helpfulness": 3.5,
        "correctness": 4,
        "coherence": 3,
        "complexity": 1.5,
        "verbosity": 2,
    }
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    monkeypatch.setattr(module, "SOURCES", {"helpsteer2": module.SOURCES["helpsteer2"]})
    monkeypatch.setattr(module, "PINNED_RAW_SHA256", {"helpsteer2": file_digest(path)})

    def download(repo, filename, *, repo_type, revision, local_files_only):
        assert local_files_only is True and repo_type == "dataset"
        assert (repo, revision, filename) == module.SOURCES["helpsteer2"][:3]
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return path


def test_raw_anchor_is_checked_before_parse_and_again_after_validation(tmp_path, monkeypatch):
    path = cached_raw_fixture(tmp_path, monkeypatch)
    registry = module.NaturalGoldRegistry()
    registry.verify_files()
    sample = registry.sample("helpsteer2", 0)
    assert all(registry(sample, q) == q.target_distribution for q in sample.questions)
    assert registry.binding["sources"]["helpsteer2"]["sha256"] == file_digest(path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed after validation"):
        registry.verify_files()
    monkeypatch.setattr(gzip, "open", lambda *a, **k: pytest.fail("must not parse changed bytes"))
    with pytest.raises(ValueError, match="pinned byte anchor"):
        module.local_raw_sources()


def test_parse_time_raw_change_is_rejected_before_returning_any_registry(tmp_path, monkeypatch):
    path = cached_raw_fixture(tmp_path, monkeypatch)
    original_open = gzip.open

    @contextmanager
    def replacing_open(*args, **kwargs):
        with original_open(*args, **kwargs) as stream:
            yield stream
        path.write_bytes(path.read_bytes() + b"replaced after parse")

    monkeypatch.setattr(gzip, "open", replacing_open)
    with pytest.raises(ValueError, match="changed during parsing"):
        module.NaturalGoldRegistry()


def test_bound_raw_recheck_never_reparses_and_rejects_changed_policy(tmp_path, monkeypatch):
    path = cached_raw_fixture(tmp_path, monkeypatch)
    registry = module.NaturalGoldRegistry()
    monkeypatch.setattr(gzip, "open", lambda *a, **k: pytest.fail("must not reparse raw corpus"))
    module.verify_raw_binding(registry.binding)
    changed = json.loads(json.dumps(registry.binding))
    changed["version"] = "unknown"
    with pytest.raises(ValueError, match="unsupported raw human"):
        module.verify_raw_binding(changed)
    changed = json.loads(json.dumps(registry.binding))
    changed["sources"]["helpsteer2"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pinned policy"):
        module.verify_raw_binding(changed)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed after validation"):
        module.verify_raw_binding(registry.binding)


def test_bundle_refuses_output_if_raw_changes_after_item_preparation(tmp_path, monkeypatch):
    from test_direct_bundle import dataset

    from ayaka.config import tiny_config
    from ayaka.tokenization import ToyTokenizer
    from ayaka.training import direct_bundle

    path = cached_raw_fixture(tmp_path, monkeypatch)
    registry = module.NaturalGoldRegistry()
    splits = dataset()
    natural = registry.sample("helpsteer2", 0)
    natural.metadata["split"] = "train"
    splits["train"].append(natural)
    original_prepare = direct_bundle._prepare

    def replace_after_preparation(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b"changed after preparation")
        return result

    monkeypatch.setattr(direct_bundle, "_prepare", replace_after_preparation)
    with pytest.raises(ValueError, match="changed after validation"):
        direct_bundle.prepare_bundle(
            tmp_path / "absent",
            splits,
            ToyTokenizer(),
            tiny_config(readout="lm", max_seq_len=2048),
            {},
            steps=1,
            rows_per_step=8,
            allow_tiny=True,
            natural_registry=registry,
        )
    assert not (tmp_path / "absent").exists()
    assert not (tmp_path / "absent-holdout").exists()


@pytest.mark.parametrize("damage", ["rating", "features", "binding"])
def test_mutable_parsed_gold_cannot_claim_original_raw_file_identity(tmp_path, monkeypatch, damage):
    cached_raw_fixture(tmp_path, monkeypatch)
    registry = module.NaturalGoldRegistry()
    if damage == "rating":
        registry.raw["helpsteer2"]["rows"][0]["helpfulness"] = 0
    elif damage == "features":
        registry.raw["helpsteer2"]["features"]["changed_ontology"] = True
    else:
        registry.binding["scope"] = "changed"
    with pytest.raises(ValueError, match="parsed human sources or bindings changed"):
        registry.verify_files()


def test_natural_bundle_reuses_one_raw_registry_without_reparsing(tmp_path, monkeypatch):
    from test_direct_bundle import dataset

    from ayaka.config import tiny_config
    from ayaka.tokenization import ToyTokenizer
    from ayaka.training import direct_bundle

    cached_raw_fixture(tmp_path, monkeypatch)
    registry = module.NaturalGoldRegistry()
    splits = dataset()
    natural = registry.sample("helpsteer2", 0)
    natural.metadata["split"] = "train"
    splits["train"].append(natural)
    original_init, calls = module.NaturalGoldRegistry.__init__, []

    def counted_init(self, *args, **kwargs):
        calls.append(1)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(module.NaturalGoldRegistry, "__init__", counted_init)
    root = tmp_path / "bundle"
    direct_bundle.prepare_bundle(
        root,
        splits,
        ToyTokenizer(),
        tiny_config(readout="lm", max_seq_len=2048),
        {},
        steps=1,
        rows_per_step=8,
        allow_tiny=True,
    )
    assert len(calls) == 1
    direct_bundle.audit_bundle(root, allow_tiny=True)
    assert len(calls) == 2


def test_raw_change_after_full_audit_prevents_receipt_publication(tmp_path, monkeypatch):
    from test_direct_bundle import dataset

    from ayaka.config import tiny_config
    from ayaka.tokenization import ToyTokenizer
    from ayaka.training import direct_audit, direct_bundle
    from ayaka.training.prepare_v2 import sha256

    path = cached_raw_fixture(tmp_path, monkeypatch)
    registry = module.NaturalGoldRegistry()
    splits = dataset()
    natural = registry.sample("helpsteer2", 0)
    natural.metadata["split"] = "train"
    splits["train"].append(natural)
    root = tmp_path / "bundle"
    direct_bundle.prepare_bundle(
        root,
        splits,
        ToyTokenizer(),
        tiny_config(readout="lm", max_seq_len=2048),
        {},
        steps=1,
        rows_per_step=8,
        allow_tiny=True,
        natural_registry=registry,
    )
    original_audit = direct_audit.audit_snapshot

    def change_after_audit(*args, **kwargs):
        result = original_audit(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b"changed after audit")
        return result

    monkeypatch.setattr(direct_audit, "audit_snapshot", change_after_audit)
    receipt = tmp_path / "must-not-publish.json"
    with pytest.raises(ValueError, match="changed after validation"):
        direct_audit.create_audit_receipt(
            root,
            receipt,
            allow_tiny=True,
            expected_manifest_sha256=sha256((root / "manifest.json").read_bytes()),
        )
    assert not receipt.exists()
