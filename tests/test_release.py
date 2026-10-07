"""Release plumbing: release specs, model card, publish dry-run guards."""

import json

import pytest
import torch

from ayaka.checkpoint import apply_lora, save_checkpoint
from ayaka.config import tiny_config
from ayaka.export import export_model
from ayaka.model.decision import AyakaDecisionModel
from ayaka.modelcard import build_card, write_card
from ayaka.publish import bundle_code, check_export
from ayaka.publish import main as publish_main
from ayaka.training.run import (
    DEFAULT_SPECS,
    RELEASE_EXCLUDED,
    RESTRICTED_SPECS,
    RunConfig,
    apply_release_policy,
)


def test_release_specs_drop_restricted_licenses():
    from ayaka.data.loaders import DATASET_SPECS

    assert {"jev_distill", "anli_r1", "super_glue_multirc", "amazon_reviews"} == RELEASE_EXCLUDED
    assert not set(DEFAULT_SPECS) & RELEASE_EXCLUDED  # default data is license-clean
    assert "jev_open" in DEFAULT_SPECS and "jev_distill" not in DEFAULT_SPECS
    for s in DEFAULT_SPECS:
        lic = DATASET_SPECS[s]["license"].lower()
        assert "non-commercial" not in lic and "nc" not in lic.split()
        assert "jev api" not in lic


def test_release_policy_drops_restricted_and_jev_output_splits():
    cfg = RunConfig(
        specs=["jev_distill", "snli", "anli_r1"],
        calibration_spec="jev_distill_calibration",
        fidelity_spec="jev_distill_test30k",
    )
    apply_release_policy(cfg, verbose=False)
    assert cfg.specs == ["snli"]
    assert cfg.calibration_spec == "jev_open_calibration"
    assert cfg.fidelity_spec == "jev_open_test"
    opt_in = RunConfig(specs=["snli"], include_restricted=True)
    apply_release_policy(opt_in, verbose=False)
    assert opt_in.specs == ["snli", *RESTRICTED_SPECS]


@pytest.fixture()
def exported(tmp_path):
    """A fake finished run: checkpoint meta, reports, manifest, then an export."""
    m = AyakaDecisionModel.from_config(tiny_config(), dtype=torch.float32)
    run = tmp_path / "runs" / "small-v1"
    ck = run / "checkpoint"
    meta = {
        "steps": 2000,
        "run": {
            "model_size": "electra-small",
            "questions_per_step": 64,
            "lr": 1e-4,
            "include_restricted": False,
        },
        "temperatures": {"noul": 1.1, "choice": 0.9},
        "reference_eval": {
            "spec": "jev_open_test",
            "n": 300,
            "accuracy": 0.81,
            "kl": 0.12,
            "brier": 0.2,
            "ece": 0.03,
        },
        "heldout": {"n": 900, "accuracy": 0.77, "kl": 0.2, "brier": 0.3, "ece": 0.04},
    }
    save_checkpoint(apply_lora(m), str(ck), meta)
    (run / "run_config.json").write_text(json.dumps(meta["run"]))
    tiers = {
        t: {"n": 2, "accuracy": a, "brier": 0.1, "ece": 0.05, "latency_p50_s": 0.7, "results": []}
        for t, a in (("easy", 1.0), ("original", 0.9), ("hard", 0.5))
    }
    summary = {
        "accuracy": {t: v["accuracy"] for t, v in tiers.items()},
        "intelligence_proxy": 61.2,
        "references": {
            "jev-1.13.0": {
                "accuracy": {"easy": 1.0, "original": 0.99, "hard": 0.74},
                "intelligence_proxy": 88.0,
            }
        },
    }
    (run / "jevbench_report.json").write_text(json.dumps({"tiers": tiers, "summary": summary}))
    manifest = [
        {
            "dataset_id": "jev_open",
            "source_url": "hf://datasets/SargeDev/jev-distill-corpus-v3/train.jsonl",
            "split": "train",
            "language": "en",
            "license": "Apache-2.0",
            "notes": "decontam_dropped=0",
        },
        {
            "dataset_id": "snli",
            "source_url": "hf://stanfordnlp/snli/None/train",
            "split": "train",
            "language": "en",
            "license": "CC BY-SA 3.0",
            "notes": "",
        },
    ]
    (run / "dataset_manifest.jsonl").write_text("\n".join(json.dumps(r) for r in manifest))
    out = tmp_path / "exports" / "electra-small"
    export_model(m, str(out), "tiny", meta={"source": str(ck)})
    return out


def test_model_card_reads_run_artifacts(exported):
    card = build_card(str(exported), code_url="https://github.com/example/ayaka")
    assert card.startswith("---\nlicense: mit\nbase_model: google/gemma-4-E2B-it")
    assert "  - SargeDev/jev-distill-corpus-v3" in card and "  - stanfordnlp/snli" in card
    assert "61.2" in card and "Jev 1.13.0" in card  # our proxy + reference row
    assert "81.0%" in card and "`jev_open_test`" in card  # reference held-out set
    assert "git+https://github.com/example/ayaka" in card
    assert "no commercial-API outputs were used as training labels" in card


def test_publish_dry_run_guards(exported, capsys):
    # placeholder code URL blocks
    write_card(str(exported))
    assert any("<code-url>" in p for p in check_export(str(exported)))
    write_card(str(exported), "https://github.com/example/ayaka")
    assert check_export(str(exported)) == []
    # no upload without --yes
    report = publish_main(["--export", str(exported), "--repo", "me/electra-small"])
    assert report["visibility"] == "private" and "url" not in report
    assert "dry-run OK" in capsys.readouterr().out


def test_bundle_code_makes_installable_copy(exported):
    dest = bundle_code(str(exported))
    import os

    assert os.path.exists(os.path.join(dest, "pyproject.toml"))
    assert os.path.exists(os.path.join(dest, "ayaka", "primitives.py"))
    assert os.path.exists(
        os.path.join(dest, "ayaka", "eval", "data", "jevbench_public", "hard.jsonl")
    )
    assert not any("__pycache__" in r for r, _, _ in os.walk(dest))


def test_checkpoint_publish_requires_pinned_base_revision_and_card(tmp_path, monkeypatch):
    import json as _json

    import torch

    from ayaka.checkpoint import apply_lora, compact_checkpoint, resolve_checkpoint, save_checkpoint
    from ayaka.config import MODEL_FAMILY, tiny_config
    from ayaka.model.decision import AyakaDecisionModel
    from ayaka.publish import is_checkpoint

    assert all(c.backbone_revision for c in MODEL_FAMILY.values())
    m = apply_lora(AyakaDecisionModel.from_config(tiny_config(), dtype=torch.float32))
    save_checkpoint(m, str(tmp_path / "ck"), {"step": 1})
    assert is_checkpoint(str(tmp_path / "ck"))
    problems = check_export(str(tmp_path / "ck"))
    assert any("backbone_revision" in p for p in problems)
    assert any("README.md" in p for p in problems)

    compact_checkpoint(str(tmp_path / "ck"), str(tmp_path / "rel"), backbone_revision="abc123")
    cfg = _json.loads((tmp_path / "rel" / "electra_config.json").read_text())
    assert cfg["backbone_revision"] == "abc123"
    (tmp_path / "rel" / "README.md").write_text("---\nlicense: mit\n---\n# card\n")
    assert check_export(str(tmp_path / "rel")) == []

    # local folders are used as-is; anything else is a Hub repo at a revision
    assert resolve_checkpoint(str(tmp_path / "rel")) == str(tmp_path / "rel")
    calls = []
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw) or "/cache/snap"
    )
    assert resolve_checkpoint("alice-noa-chan/ayaka-large", "v1") == "/cache/snap"
    assert calls == [{"repo_id": "alice-noa-chan/ayaka-large", "revision": "v1"}]


def test_tokenizer_for_config_uses_the_pinned_revision(monkeypatch):
    import transformers

    from ayaka.config import MODEL_FAMILY
    from ayaka.tokenization import HFTokenizer

    seen = {}

    class Fake:
        bos_token_id = 2
        pad_token_id = 0

    def fake(repo, revision=None):
        seen.update(repo=repo, revision=revision)
        return Fake()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake)
    cfg = MODEL_FAMILY["electra-large"]
    HFTokenizer.for_config(cfg)
    assert seen == {"repo": cfg.backbone, "revision": cfg.backbone_revision}
