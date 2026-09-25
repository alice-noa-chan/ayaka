"""Release plumbing: release specs, model card, publish dry-run guards."""

import json

import pytest
import torch

from ayaka.checkpoint import apply_lora, save_checkpoint
from ayaka.config import tiny_config
from ayaka.export import export_model
from ayaka.model.electra import ElectraDecisionModel
from ayaka.modelcard import build_card, write_card
from ayaka.publish import bundle_code, check_export
from ayaka.publish import main as publish_main
from ayaka.training.run import DEFAULT_SPECS, RELEASE_EXCLUDED, RELEASE_SPECS


def test_release_specs_drop_restricted_licenses():
    from ayaka.data.loaders import DATASET_SPECS

    assert {"anli_r1", "super_glue_multirc", "amazon_reviews"} <= RELEASE_EXCLUDED
    assert set(RELEASE_SPECS) == set(DEFAULT_SPECS) - RELEASE_EXCLUDED
    for s in RELEASE_SPECS:
        lic = DATASET_SPECS[s]["license"].lower()
        assert "non-commercial" not in lic and "nc" not in lic.split()


@pytest.fixture()
def exported(tmp_path):
    """A fake finished run: checkpoint meta, reports, manifest, then an export."""
    m = ElectraDecisionModel.from_config(tiny_config(), dtype=torch.float32)
    run = tmp_path / "runs" / "small-v1"
    ck = run / "checkpoint"
    meta = {
        "steps": 2000,
        "run": {
            "model_size": "electra-small",
            "questions_per_step": 64,
            "lr": 1e-4,
            "release": True,
        },
        "temperatures": {"noul": 1.1, "choice": 0.9},
        "jev_fidelity": {"n": 300, "accuracy": 0.81, "kl": 0.12, "brier": 0.2, "ece": 0.03},
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
            "dataset_id": "jev_distill",
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
    assert card.startswith("---\nlicense: apache-2.0\nbase_model: google/gemma-4-E2B-it")
    assert "  - SargeDev/jev-distill-corpus-v3" in card and "  - stanfordnlp/snli" in card
    assert "61.2" in card and "Jev 1.13.0" in card  # our proxy + reference row
    assert "81.0%" in card  # Jev fidelity accuracy
    assert "git+https://github.com/example/ayaka" in card
    assert "TypeSafe" in card  # data-terms notice


def test_publish_dry_run_guards(exported, capsys):
    # placeholder code URL blocks
    write_card(str(exported))
    assert any("<code-url>" in p for p in check_export(str(exported)))
    write_card(str(exported), "https://github.com/example/ayaka")
    assert check_export(str(exported)) == []
    # jev-distill provenance requires the explicit terms flag, and no upload without --yes
    with pytest.raises(SystemExit):
        publish_main(["--export", str(exported), "--repo", "me/electra-small"])
    report = publish_main(
        ["--export", str(exported), "--repo", "me/electra-small", "--confirm-data-terms"]
    )
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
