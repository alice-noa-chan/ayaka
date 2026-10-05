import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_swift_matched_2x2 import cohort

from scripts.swift import gpu_runner, matched_2x2, matched_native


def test_p5_admission_implies_p4_exact_fit_and_prefix():
    models = gpu_runner.model_specs(None, dry_run=True)
    full = gpu_runner.admission_plan(models, 164, with_matched_2x2=True)
    assert full["planned_minutes"] == full["admitted_minutes"] == 164
    assert all(p["admitted"] for p in full["priorities"])
    p5 = full["priorities"][-1]
    assert (p5["id"], p5["minutes"], p5["load_minutes"], p5["total_minutes"]) == ("P5", 15, 4, 19)
    refused = gpu_runner.admission_plan(models, 143.99, with_matched_2x2=True)
    assert refused["refused"] and not any(p["admitted"] for p in refused["priorities"])
    for cap, last in [(144, "P3"), (145, "P4"), (163.99, "P4"), (164, "P5")]:
        partial = gpu_runner.admission_plan(models, cap, with_matched_2x2=True, allow_partial=True)
        assert partial["admitted_minutes"] <= cap
        assert [p["id"] for p in partial["priorities"] if p["admitted"]][-1] == last
    assert "DIAGNOSTIC ONLY" in p5["work"]


def test_p5_dry_run_no_download_and_commands(capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run launched a process or downloaded a checkpoint")

    monkeypatch.setattr(gpu_runner.subprocess, "Popen", forbidden)
    monkeypatch.setattr(matched_native, "prepare_checkpoint", forbidden)
    assert gpu_runner.main(["--dry-run", "--with-matched-2x2", "--max-minutes", "164"]) == 0
    text = capsys.readouterr().out
    assert "Enabled plan total: 164 minutes; admitted total: 164 minutes" in text
    assert "P4           15  True" in text and "P5           15  True" in text
    assert "--max-seq-len 8192" in text and "--zero-shot electra-large" in text
    assert matched_native.CHECKPOINT_REVISION in text and "selects nothing" in text
    with pytest.raises(SystemExit):
        gpu_runner.main(["--dry-run", "--with-matched-2x2", "--lora-revision", "a" * 40])


def test_native_command_and_pinned_config():
    from dataclasses import asdict

    from ayaka.config import model_config

    config = asdict(model_config("electra-large"))
    matched_native.validate_config(config)
    for key, value in [("backbone_revision", "main"), ("readout", "lm"), ("version", 2)]:
        with pytest.raises(ValueError, match="pinned v1 large"):
            matched_native.validate_config({**config, key: value})
    command = matched_native.native_command("lora_native", Path("snapshot"), Path("report.json"))
    assert command[1:3] == ["-m", "ayaka.eval.jevbench"]
    assert command[command.index("--ckpt") + 1] == "snapshot"
    assert command[command.index("--max-seq-len") + 1] == "8192"


def test_adapter_conversion_preserves_payload_and_tensor_metadata(tmp_path):
    key = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
    tensor = {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, 8]}
    header = json.dumps({key: tensor, "__metadata__": {"format": "pt"}}).encode()
    header += b" " * (-len(header) % 8)
    payload = struct.pack("<ff", 1.25, -2.5)
    source, target = tmp_path / "native.safetensors", tmp_path / "swift.safetensors"
    source.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    conversion = matched_native.convert_adapter(source, target, "gemma4")
    data = target.read_bytes()
    length = struct.unpack("<Q", data[:8])[0]
    converted_header = json.loads(data[8 : 8 + length])
    expected_key = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
    assert converted_header[expected_key] == tensor
    assert data[8 + length :] == payload
    assert conversion["tensor_count"] == 1
    assert conversion["key_mapping"] == {key: expected_key}
    assert matched_native.swift_adapter_key(key, "gemma4_text").startswith(
        "base_model.model.model.layers."
    )
    with pytest.raises(ValueError, match="unexpected published adapter key"):
        matched_native.swift_adapter_key("unrelated.weight", "gemma4")


@pytest.mark.parametrize("fail_after", [None, 3])
def test_native_cli_recording_retains_partials_without_extra_reads(
    tmp_path, monkeypatch, fail_after
):
    from ayaka.eval import jevbench

    items = cohort(231)
    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    (checkpoint / "electra_config.json").write_text(
        json.dumps(
            {
                "backbone": matched_native.BASE,
                "backbone_revision": matched_native.BASE_REVISION,
            }
        )
    )
    monkeypatch.setattr(matched_native, "public_items", lambda directory: items)
    rows = [
        {
            "id": i,
            "expected": "a",
            "labels": ["a", "b"],
            "question": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
            "state": "synthetic",
            "family": metadata["family"],
        }
        for i, metadata in items.items()
    ]
    calls = []

    class Decision:
        def decide(self, state, questions, device=None):
            if len(calls) == fail_after:
                raise RuntimeError("synthetic interruption")
            calls.append(state)
            return [SimpleNamespace(probs=[0.75, 0.25])]

    def fake_run(*args, **kwargs):
        return {"tiers": {"synthetic": jevbench.evaluate(Decision(), rows)}}

    def fake_main(argv):
        assert argv[argv.index("--zero-shot") + 1] == "electra-large"
        assert argv[argv.index("--dtype") + 1] == "bfloat16"
        return jevbench.run_jevbench(None, None)

    monkeypatch.setattr(jevbench, "run_jevbench", fake_run)
    monkeypatch.setattr(jevbench, "main", fake_main)
    original_evaluate = jevbench.evaluate
    output = tmp_path / "native"
    if fail_after is None:
        receipt = matched_native.run_native("frozen_native", checkpoint, output)
        assert receipt["complete"] and receipt["completed_items"] == 231
        assert (output / "report.json").is_file()
    else:
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            matched_native.run_native("frozen_native", checkpoint, output)
        receipt = json.loads((output / "load_times.json").read_text())
        assert receipt["complete"] is False and receipt["completed_items"] == 3
    assert len(matched_2x2.records(output / "results.jsonl")) == len(calls)
    assert receipt["hf_load_s"] >= 0
    assert receipt["base_revision"] == matched_native.BASE_REVISION
    assert jevbench.run_jevbench is fake_run and jevbench.evaluate is original_evaluate
