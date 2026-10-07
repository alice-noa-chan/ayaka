import io
import json
import runpy
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import zstandard

from scripts.beam_v2 import mechanism_worker
from scripts.beam_v2.budget import admit_serverless
from scripts.beam_v2.verify import verify_delivery
from scripts.beam_v2.worker import digest, package_results, write_json


def admission():
    return admit_serverless(
        {"gpu": "RTX5090", "serverless": "ready"},
        credits="3.278",
        spend_cap="3.73",
        seconds=3400,
    )


def test_fixed_diagnostic_budget_fits_existing_credit_and_refuses_retry(tmp_path):
    plan = admission()
    assert plan["compute_ceiling_usd"] == "2.35" and plan["planned_ceiling_usd"] == "2.45"
    mechanism_worker.validate_admission(plan)
    for change in (
        {"full_training": True},
        {"task_timeout_seconds": 9700},
        {"credit_usd": "2.44"},
        {"credit_usd": "NaN"},
    ):
        with pytest.raises(ValueError):
            mechanism_worker.validate_admission({**plan, **change})
    write_json(tmp_path / (mechanism_worker.RUN_NAME + ".receipt.json"), {"status": "failed"})
    with pytest.raises(ValueError, match="refuse paid retry"):
        mechanism_worker.execute(tmp_path, "sha", "sha", plan)


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "ayaka/model/decision.py",
        "diagnostic-pilot/adapter/a.pt",
        "diagnostic-pilot/adapter/a\\b.json",
    ],
)
def test_overlay_paths_cannot_change_unrelated_code_or_load_pickle(name):
    assert not mechanism_worker.overlay_allowed(name)


def test_overlay_validates_inventory_and_every_byte(tmp_path):
    archive = tmp_path / "overlay.tar.zst"
    data = b"{}"
    import hashlib

    manifest = {
        "files": {"mechanism-plan.json": {"bytes": 2, "sha256": hashlib.sha256(data).hexdigest()}}
    }
    with (
        archive.open("wb") as raw,
        zstandard.ZstdCompressor(level=3).stream_writer(raw) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as tar,
    ):
        for name, payload in [
            ("mechanism-plan.json", data),
            ("mechanism-overlay-manifest.json", json.dumps(manifest).encode()),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
    kit = tmp_path / "kit"
    kit.mkdir()
    assert mechanism_worker.apply_overlay(archive, kit, digest(archive)) == manifest
    with pytest.raises(ValueError, match="checksum"):
        mechanism_worker.apply_overlay(archive, kit, "0" * 64)
    env = mechanism_worker.runtime_environment(kit)
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["PYTHONPATH"] == str(kit)


def test_remote_diagnostic_imports_with_older_sdk(monkeypatch):
    calls = []

    def decorate(**kwargs):
        assert "allow_marketplace" not in kwargs
        calls.append(kwargs)
        return lambda fn: fn

    monkeypatch.setitem(
        sys.modules,
        "beam",
        SimpleNamespace(
            function=decorate,
            Image=lambda **_: SimpleNamespace(add_python_packages=lambda _: None),
            Volume=lambda **_: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "mechanism_worker", mechanism_worker)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/beam_v2/mechanism_job.py"))
    assert len(calls) == 1
    assert calls[0]["gpu"] == "RTX5090" and calls[0]["timeout"] == 3400
    assert calls[0]["retries"] == 0 and calls[0]["headless"]
    assert calls[0]["cpu"] == 2 and calls[0]["memory"] == "32Gi"


def test_delivery_requires_all_frozen_conditions_without_training(tmp_path):
    root = tmp_path / "outputs"
    (root / "mechanism").mkdir(parents=True)
    write_json(root / "operational-outcome.json", {"status": "complete"})
    write_json(
        root / "mechanism/complete.json",
        {
            "complete": True,
            "optimizer_steps": 0,
            "full_training_started": False,
            "test_evaluated": False,
        },
    )
    plan = {
        "checkpoints": {"parent": "parent", "pilot": "pilot"},
        "cohort_sha256": "cohort",
        "questions_per_checkpoint": 36,
    }
    write_json(root / "plan.json", plan)
    rows = {
        f"{c}/{h}": [{}] * 36
        for c in ("direct", "empty", "generated", "oracle", "distractor")
        for h in ("lm", "pointer", "hybrid")
    }
    for checkpoint in ("parent", "pilot"):
        write_json(
            root / ("mechanism/" + checkpoint + ".json"),
            {
                "complete": True,
                "model_id": checkpoint,
                "cohort_sha256": "cohort",
                "rows": rows,
                "optimizer_steps": 0,
            },
        )
    archive = tmp_path / "complete.tar.zst"
    receipt = {**package_results(root, archive), "status": "complete"}
    result = verify_delivery(archive, receipt, tmp_path / "received", completion="mechanism")
    assert result["completion"]["optimizer_steps"] == 0
    # A byte-perfect archive cannot turn a missing arm into a completed experiment.
    path = root / "mechanism/pilot.json"
    report = json.loads(path.read_text())
    report["rows"]["generated/lm"].pop()
    write_json(path, report)
    (root / "receipt-manifest.json").unlink()
    archive = tmp_path / "partial.tar.zst"
    receipt = {**package_results(root, archive), "status": "complete"}
    with pytest.raises(ValueError, match="every context"):
        verify_delivery(archive, receipt, tmp_path / "partial", completion="mechanism")


def test_v1_only_delivery_cannot_claim_an_unevaluated_pilot(tmp_path):
    root = tmp_path / "outputs"
    (root / "mechanism").mkdir(parents=True)
    write_json(root / "operational-outcome.json", {"status": "complete"})
    complete = {
        "complete": True,
        "optimizer_steps": 0,
        "full_training_started": False,
        "test_evaluated": False,
        "checkpoints_evaluated": ["parent"],
    }
    write_json(root / "mechanism/complete.json", complete)
    plan = {
        "checkpoints": {"parent": "parent"},
        "cohort_sha256": "cohort",
        "questions_per_checkpoint": 36,
        "active_checkpoints": ["parent"],
    }
    write_json(root / "plan.json", plan)
    rows = {
        f"{c}/{h}": [{}] * 36
        for c in ("direct", "empty", "generated", "oracle", "distractor")
        for h in ("lm", "pointer", "hybrid")
    }
    write_json(
        root / "mechanism/parent.json",
        {
            "complete": True,
            "model_id": "parent",
            "cohort_sha256": "cohort",
            "rows": rows,
            "optimizer_steps": 0,
        },
    )
    archive = tmp_path / "v1.tar.zst"
    receipt = {**package_results(root, archive), "status": "complete"}
    assert verify_delivery(archive, receipt, tmp_path / "received", completion="mechanism")[
        "completion"
    ]["checkpoints_evaluated"] == ["parent"]
    complete["checkpoints_evaluated"] = ["parent", "pilot"]
    write_json(root / "mechanism/complete.json", complete)
    (root / "receipt-manifest.json").unlink()
    archive = tmp_path / "invalid.tar.zst"
    receipt = {**package_results(root, archive), "status": "complete"}
    with pytest.raises(ValueError, match="receipt"):
        verify_delivery(archive, receipt, tmp_path / "invalid", completion="mechanism")
