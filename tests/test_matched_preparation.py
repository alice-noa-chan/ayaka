import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_matched_contract import matched as matched
from test_matched_execution import assets as assets

from ayaka.eval.matched_contract import digest
from scripts.direct_v2 import matched_preflight, prepare_matched


def test_preparation_requires_full_consumed_inventory_and_external_fit_declarations(
    assets, matched
):
    _, protocol, receipt, _, _ = assets
    policy_bytes = matched[4].read_bytes()
    fit = {
        "inputs_sha256": protocol["policy_fit_inputs_sha256"],
        "source_sha256": protocol["policy_fit_source_sha256"],
    }
    result = prepare_matched.prepare(
        json.dumps(receipt).encode(), policy_bytes, json.dumps(fit).encode()
    )
    assert result == protocol
    receipt["source_sha256"].pop("meta.json")
    with pytest.raises(ValueError, match="inventory"):
        prepare_matched.prepare(
            json.dumps(receipt).encode(), policy_bytes, json.dumps(fit).encode()
        )


def test_cpu_preflight_validates_existing_reads_without_model_work(
    matched, tmp_path, monkeypatch, capsys
):
    items, v2, _, protocol, policy = matched
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(protocol), encoding="utf-8")
    reads = tmp_path / "reads.jsonl"
    reads.write_text("".join(json.dumps(r) + "\n" for r in v2[1:]), encoding="utf-8")
    monkeypatch.setattr(
        matched_preflight,
        "cohort_parts",
        lambda *a: {"procedural": items[:1], "hard_calibration": items[1:2], "hard_dev": items[2:]},
    )
    monkeypatch.setattr(
        matched_preflight, "preflight_procedural", lambda items: {item.id: {} for item in items}
    )
    monkeypatch.setattr(matched_preflight, "verify_swift_source", lambda: {})
    argv = [
        "--protocol",
        str(p),
        "--protocol-sha256",
        digest(p.read_bytes()),
        "--policy",
        str(policy),
        "--procedural",
        "p",
        "--hard-calibration",
        "c",
        "--hard-dev",
        "d",
        "--v2-reads",
        str(reads),
    ]
    assert matched_preflight.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["model_forward_calls"] == 0
    v2[1]["candidate_log_masses"]["false"] = 99.0
    reads.write_text("".join(json.dumps(r) + "\n" for r in v2[1:]), encoding="utf-8")
    with pytest.raises(ValueError):
        matched_preflight.main(argv)


def test_job_requires_both_preflights_and_successful_scoring_before_done():
    job = (Path(__file__).resolve().parents[1] / "scripts/direct_v2/matched_job.sh").read_text(
        encoding="utf-8"
    )
    assert job.index("-m scripts.direct_v2.matched_preflight") < job.index("vllm serve")
    assert job.index("--preflight-only") < job.index("vllm serve")
    assert job.index("-m scripts.direct_v2.matched_compare") < job.index('"$OUT/DONE"')
    assert "scripts/swift/v1_on_runner.py" not in job
    assert "snapshot_download" not in job and "--prepare-checkpoint" not in job
    assert "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in job
    assert (
        job.index('if [ -e "$OUT" ]')
        < job.index('mkdir "$OUT"')
        < job.index("-m scripts.direct_v2.matched_preflight")
    )


@pytest.mark.parametrize("which", ["one_missing", "all_missing", "procedural_instead"])
def test_missing_or_wrong_saved_hard_rows_fail_before_gpu_start(
    matched, tmp_path, monkeypatch, which
):
    items, rows, _, protocol, policy = matched
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    selected = rows[1:2] if which == "one_missing" else [] if which == "all_missing" else rows[:1]
    reads = tmp_path / "hard.jsonl"
    reads.write_text("".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
    monkeypatch.setattr(
        matched_preflight,
        "cohort_parts",
        lambda *a: {"procedural": items[:1], "hard_calibration": items[1:2], "hard_dev": items[2:]},
    )
    with pytest.raises(ValueError, match="frozen cohort"):
        matched_preflight.main(
            [
                "--protocol",
                str(path),
                "--protocol-sha256",
                digest(path.read_bytes()),
                "--policy",
                str(policy),
                "--procedural",
                "p",
                "--hard-calibration",
                "c",
                "--hard-dev",
                "d",
                "--v2-reads",
                str(reads),
            ]
        )


def test_protocol_publication_cannot_add_a_consumed_checkpoint_file(assets, matched, tmp_path):
    _, protocol, _, root, args = assets
    fit = tmp_path / "fit.json"
    fit.write_text(
        json.dumps(
            {
                "inputs_sha256": protocol["policy_fit_inputs_sha256"],
                "source_sha256": protocol["policy_fit_source_sha256"],
            }
        ),
        encoding="utf-8",
    )
    policy = matched[4]
    output = root / "adapter/protocol.json"
    with pytest.raises(ValueError, match="must not modify checkpoint"):
        prepare_matched.main(
            [
                "--checkpoint-receipt",
                str(args.checkpoint_receipt),
                "--checkpoint-receipt-sha256",
                digest(args.checkpoint_receipt.read_bytes()),
                "--policy",
                str(policy),
                "--policy-sha256",
                digest(policy.read_bytes()),
                "--fit-provenance",
                str(fit),
                "--fit-provenance-sha256",
                digest(fit.read_bytes()),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_pending_native_inputs_are_checked_before_collection(matched, monkeypatch):
    from test_evidence_swift_bridge import tokenizer

    from ayaka.input_errors import ContextLimitError

    items = matched[0]
    monkeypatch.setattr(matched_preflight, "cached_tokenizer", lambda *a: tokenizer())
    assert set(matched_preflight.preflight_procedural(items)) == {item.id for item in items}
    items[0] = replace(items[0], state="x" * 16384)
    with pytest.raises(ContextLimitError, match="refuse GPU collection"):
        matched_preflight.preflight_procedural(items)


def test_collection_source_bytes_are_fixed_before_any_service(monkeypatch, tmp_path):
    pins = {}
    for name in matched_preflight.SWIFT_IMPLEMENTATION:
        data = (name + "\n").encode()
        (tmp_path / name).write_bytes(data)
        pins[name] = digest(data)
    monkeypatch.setattr(matched_preflight, "SWIFT_IMPLEMENTATION", pins)
    assert matched_preflight.verify_swift_source(tmp_path) == pins
    (tmp_path / "readers.py").write_bytes(b"changed source\n")
    with pytest.raises(ValueError, match="pinned collection bytes"):
        matched_preflight.verify_swift_source(tmp_path)


@pytest.mark.parametrize("failure", ["old_output", "occupied_endpoint"])
def test_job_refuses_stale_results_or_server_before_gpu_command(tmp_path, failure):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("existing bash unavailable")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = sys.executable.replace("\\", "/")
    (bin_dir / "python3").write_text(
        f'#!/bin/bash\nif [ "$1" = "-m" ]; then exit 0; fi\nexec "{python}" "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    (bin_dir / "vllm").write_text(
        '#!/bin/bash\necho launched > "$TRACE"\nexit 1\n', encoding="utf-8", newline="\n"
    )
    for path in bin_dir.iterdir():
        path.chmod(0o755)
    out, trace = tmp_path / "out", tmp_path / "gpu-started"
    if failure == "old_output":
        out.mkdir()
        (out / "DONE").write_text("stale", encoding="utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if failure == "occupied_endpoint":
            try:
                probe.bind(("127.0.0.1", 8000))
                probe.listen()
            except OSError:
                pass  # An already occupied port must produce the same refusal.
        env = {
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
            "IN": str(tmp_path / "input"),
            "OUT": out.as_posix(),
            "TRACE": trace.as_posix(),
            "PROTOCOL_SHA256": "a" * 64,
            "PYTHONIOENCODING": "utf-8",
        }
        script = Path(__file__).resolve().parents[1] / "scripts/direct_v2/matched_job.sh"
        result = subprocess.run(
            [bash, str(script)],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    assert result.returncode != 0
    assert not trace.exists(), result.stdout + result.stderr
    if failure == "old_output":
        assert "OUT must be fresh" in result.stderr
        assert (out / "DONE").read_text(encoding="utf-8") == "stale"
    else:
        assert "OSError" in result.stderr or "PermissionError" in result.stderr
        assert not (out / "DONE").exists()
