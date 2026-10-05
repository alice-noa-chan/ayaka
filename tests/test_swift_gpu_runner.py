import ctypes.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ayaka.swift.collect import collect, iter_dataset, load_reads
from ayaka.swift.readers import FakeReader
from ayaka.swift.server import DecisionService, serve
from scripts.swift import gpu_runner
from scripts.swift.inputs import DEFAULT_MANIFEST, PACKED_MANIFEST, REPO, inventory, sha256
from scripts.swift.latency_probe import probe, quantile
from scripts.swift.pack_inputs import pack_inputs


def record(index):
    return {
        "id": f"public-{index}",
        "state": "fixture",
        "split": "public",
        "labels": ["a", "b"],
        "expected": "a",
        "question": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
    }


def fixture_manifest(root):
    dataset = root / "calibration.jsonl"
    dataset.write_text(json.dumps(record(0)) + "\n", encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": [{"name": "fixture", "paths": [dataset.name]}],
            }
        ),
        encoding="utf-8",
    )
    return dataset, manifest


def test_pack_inputs_determinism_metadata_hashes_and_no_test(tmp_path):
    dataset, manifest = fixture_manifest(tmp_path)
    (tmp_path / "test.jsonl").write_bytes(b"must stay unopened")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    resolved = pack_inputs(tmp_path, manifest, first)
    os.utime(dataset, (123456789, 123456789))
    pack_inputs(tmp_path, manifest, second)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()[4:8] == b"\x00" * 4
    assert first.with_name(first.name + ".sha256").read_text().split()[0] == sha256(first)
    with tarfile.open(first) as archive:
        assert archive.getnames() == sorted([dataset.name, PACKED_MANIFEST])
        assert all(member.mtime == member.uid == member.gid == 0 for member in archive.getmembers())
        assert all(member.mode == 0o644 for member in archive.getmembers())
        assert json.load(archive.extractfile(PACKED_MANIFEST)) == resolved
        assert archive.extractfile(dataset.name).read() == dataset.read_bytes()
    assert resolved["datasets"][0]["sha256"] == {dataset.name: sha256(dataset)}
    with pytest.raises(ValueError, match="overwrite"):
        pack_inputs(tmp_path, manifest, dataset)


@pytest.mark.parametrize("bad_path", ["test.jsonl", "../secret.jsonl", "C:/secret.jsonl"])
def test_packer_rejects_heldout_and_escaping_paths_before_open(tmp_path, monkeypatch, bad_path):
    _, manifest = fixture_manifest(tmp_path)
    spec = json.loads(manifest.read_text())
    spec["datasets"][0]["paths"] = [bad_path]
    manifest.write_text(json.dumps(spec))
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.name != "test.jsonl", "held-out file was opened"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError):
        pack_inputs(tmp_path, manifest, tmp_path / "packed.tar.gz")


def test_default_inventory_and_real_pack_never_open_test(tmp_path, monkeypatch):
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.name != "test.jsonl", "held-out file was opened"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    resolved = pack_inputs(REPO, DEFAULT_MANIFEST, tmp_path / "inputs.tar.gz")
    assert [(d["name"], d["items"], d["reads"]) for d in resolved["datasets"]] == [
        ("v2_calibration", 1824, 2208),
        ("v2_dev", 1824, 2208),
        ("cygnet_calibration", 241, 241),
        ("jevbench_public", 231, 231),
    ]
    assert sum(d["reads"] for d in resolved["datasets"]) == 4888
    cygnet = REPO / "ayaka/swift/data/cygnet_calibration_items.jsonl"
    assert sha256(cygnet) == "73292f2416cc54991b39e7046b954396bfb4199bae7a082da1bca4306e3e1ea4"
    with tarfile.open(tmp_path / "inputs.tar.gz") as archive:
        assert all(Path(name).name != "test.jsonl" for name in archive.getnames())
        assert "ayaka/swift/data/cygnet_LICENSE" in archive.getnames()
        assert "ayaka/eval/data/jevbench_public/LICENSE" in archive.getnames()


def test_inventory_rejects_incomplete_public_data_and_changed_hash(tmp_path):
    dataset, manifest = fixture_manifest(tmp_path)
    spec = json.loads(manifest.read_text())
    spec["datasets"][0]["expected_items"] = 231
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="expected 231"):
        inventory(tmp_path, manifest)
    del spec["datasets"][0]["expected_items"]
    spec["datasets"][0]["sha256"] = {dataset.name: "incorrect"}
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="hash mismatch"):
        inventory(tmp_path, manifest)


def test_jevbench_audit_fallback_packs_under_canonical_paths(tmp_path):
    root = tmp_path / "repo"
    audit = tmp_path / "audit"
    root.mkdir()
    audit.mkdir()
    names = ["easy.jsonl", "hard.jsonl", "original.jsonl"]
    prefix = "ayaka/eval/data/jevbench_public/"
    for index, name in enumerate(names):
        (audit / name).write_text(json.dumps(record(index)) + "\n")
    (audit / "LICENSE").write_text("license fixture")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": [
                    {
                        "name": "jevbench_public",
                        "paths": [prefix + name for name in names],
                        "expected_items": 3,
                    }
                ],
                "notices": [prefix + "LICENSE"],
            }
        )
    )
    output = tmp_path / "inputs.tar.gz"
    with pytest.raises(ValueError, match="missing input"):
        pack_inputs(root, manifest, output)
    pack_inputs(root, manifest, output, jevbench_dir=audit)
    with tarfile.open(output) as archive:
        assert all(prefix + name in archive.getnames() for name in [*names, "LICENSE"])


def test_dry_run_no_processes_or_network_and_explicit_unresolved_pins(monkeypatch, capsys):
    monkeypatch.delenv("SWIFT_GEMMA_E4B_REVISION", raising=False)
    monkeypatch.delenv("SWIFT_QWEN_REVISION", raising=False)
    monkeypatch.delenv("SWIFT_MAX_MINUTES", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not start subprocesses or make network requests")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    assert gpu_runner.main(["--dry-run", "--manifest", str(DEFAULT_MANIFEST)]) == 0
    output = capsys.readouterr().out
    assert "vllm==0.30.0" in output
    assert "32960 bulk decisions, 39104 bulk model reads + 800 serial HTTP requests" in output
    assert "P1R upper bound: 3648 candidate decisions, 7296 additional model calls" in output
    assert "Prompt variants: min,cygnet,rules,labeled" in output
    assert "Qwen/Qwen3.5-4B" not in output
    assert "<set revision via env/--model>" in output
    assert "75 minutes" in output
    assert "Time budget table" in output
    assert "pre-launch admission" in output
    assert "Enabled plan total: 105 minutes" in output
    assert "Admission: REFUSED" in output
    assert "Cleanup/pack reserve: 120s inside the cap" in output
    assert all(priority in output for priority in ("P0", "P1", "P1R", "P2", "P3", "P4"))
    assert "best TWO" in output
    assert "P4           15  False" in output
    assert "centered log-mass <= 0.05 nats" in output


def git_bash():
    if os.name != "nt":
        return shutil.which("bash")
    candidates = [
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
    ]
    git = shutil.which("git")
    if git:
        result = subprocess.run([git, "--exec-path"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            root = Path(result.stdout.strip()).parent.parent.parent
            candidates.extend([root / "bin/bash.exe", root / "usr/bin/bash.exe"])
    return next((str(path) for path in candidates if path.is_file()), None)


def test_shell_entrypoint_dry_run_and_syntax():
    bash = git_bash()
    if bash is None:
        pytest.skip("Git Bash is unavailable; WSL system32 bash is not a supported shell")
    script = REPO / "scripts/swift/collect_gpu.sh"
    environment = dict(os.environ, SWIFT_PYTHON=sys.executable)
    for arguments in (["-n", script.as_posix()], [script.as_posix(), "--dry-run"]):
        result = subprocess.run(
            [bash, *arguments], env=environment, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr
    assert "4888 model reads" in result.stdout


def test_dry_run_fixed_variants_and_optional_lora(capsys):
    assert gpu_runner.main(["--dry-run", "--with-lora-arm"]) == 0
    output = capsys.readouterr().out
    assert "P4           15  True" in output
    assert "Enabled plan total: 125 minutes" in output
    for invalid in ("", "min,min", "min,unknown", "min,", "min,rules"):
        with pytest.raises(SystemExit):
            gpu_runner.main(["--dry-run", "--prompt-variants", invalid])


def test_models_require_live_revisions_and_respect_environment(monkeypatch):
    monkeypatch.delenv("SWIFT_GEMMA_E4B_REVISION", raising=False)
    monkeypatch.delenv("SWIFT_QWEN_REVISION", raising=False)
    with pytest.raises(ValueError, match="needs a revision"):
        gpu_runner.model_specs(None, dry_run=False)
    monkeypatch.setenv("SWIFT_GEMMA_E4B_REVISION", "e" * 40)
    monkeypatch.setenv("SWIFT_QWEN_REVISION", "qwen-release")
    models = gpu_runner.model_specs(None, dry_run=False)
    assert [model["requested_revision"] for model in models] == [
        gpu_runner.GEMMA_12B_REVISION,
        "e" * 40,
    ]
    assert gpu_runner.model_specs(["fixture@abc"], dry_run=False)[0]["model"] == "fixture"


def test_per_model_options_and_immutable_serving_settings(tmp_path):
    models = gpu_runner.model_specs(["fixture@abc"], dry_run=False)
    options_path = tmp_path / "options.json"
    options_path.write_text(
        json.dumps(
            {
                "fixture": {
                    "chat_template_kwargs": {"thinking": False},
                    "vllm_args": ["--enforce-eager"],
                }
            }
        )
    )
    options = gpu_runner.model_options(options_path, models)
    command = gpu_runner.serve_command(dict(models[0], revision="a" * 40), 8000, options["fixture"])
    assert command[command.index("--revision") + 1] == "a" * 40
    assert command[command.index("--tokenizer-revision") + 1] == "a" * 40
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--max-model-len") + 1] == "16384"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.90"
    assert "--enable-prefix-caching" in command
    assert command[command.index("--logprobs-mode") + 1] == "raw_logits"
    assert command[command.index("--max-logprobs") + 1] == "26"
    assert command[command.index("--chat-template-content-format") + 1] == "string"
    assert command[-1] == "--enforce-eager"
    options_path.write_text(json.dumps({"fixture": {"vllm_args": ["--revision=main"]}}))
    with pytest.raises(ValueError, match="must not override"):
        gpu_runner.model_options(options_path, models)


@pytest.mark.parametrize("valid_switch", [True, False])
def test_qwen_thinking_verified_at_resolved_revision_without_downloads(
    tmp_path, monkeypatch, valid_switch
):
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if kwargs["enable_thinking"] is False and valid_switch:
                return "assistant\n<think>\n\n</think>\n\n"
            return "assistant\n<think>\n"

    def from_pretrained(model, revision, local_files_only):
        assert local_files_only
        calls.append((model, revision))
        return Tokenizer()

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: SimpleNamespace(
                model_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40)
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained),
        ),
    )
    output = tmp_path / "revision.json"
    if valid_switch:
        gpu_runner.resolve_model("Qwen/Qwen3.5-4B", "tag", output, {"enable_thinking": False})
        assert json.loads(output.read_text())["revision"] == "a" * 40
    else:
        with pytest.raises(ValueError, match="did not disable thinking"):
            gpu_runner.resolve_model("Qwen/Qwen3.5-4B", "tag", output, {"enable_thinking": False})
    assert calls == [("Qwen/Qwen3.5-4B", "a" * 40)]


def test_serial_200_probe_uses_swift_http_and_raw_quantiles(tmp_path):
    reader = FakeReader()
    service = DecisionService(
        reader, "fake", max_parallel=1, prompt_variant="cygnet", diagnostic=True
    )
    server = serve(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    dataset = tmp_path / "public.jsonl"
    dataset.write_text("\n".join(json.dumps(record(index)) for index in range(231)))
    output = tmp_path / "latency.json"
    try:
        result = probe(
            [dataset],
            f"http://127.0.0.1:{server.server_port}",
            output,
            model="fake",
            revision="a" * 40,
            prompt_variant="cygnet",
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        service.close()
    assert result == json.loads(output.read_text())
    assert result["complete"] and result["completed_reads"] == 200
    assert result["concurrency"] == 1 and len(reader.calls) == 200
    assert result["prompt_variant"] == "cygnet"
    assert all("calibration engine" in messages[0]["content"] for messages, _ in reader.calls)
    assert len({sample["id"] for sample in result["samples"]}) == 200
    values = [sample["latency_s"] for sample in result["samples"]]
    assert result["p50_s"] == sorted(values)[100]
    assert result["p95_s"] == sorted(values)[190]
    assert not result["self_hosted_adjustment"]["applied"]


def test_probe_preserves_partial_samples_after_failure(tmp_path, monkeypatch):
    dataset, _ = fixture_manifest(tmp_path)
    dataset.write_text("\n".join(json.dumps(record(index)) for index in range(3)))
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        if len(requests) == 2:
            raise TimeoutError("backend timed out")
        return io.BytesIO(b'{"answers":{"probe": {}}}')

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    output = tmp_path / "latency.json"
    with pytest.raises(TimeoutError):
        probe([dataset], "http://swift", output, model="fake", revision="pin", reads=3)
    partial = json.loads(output.read_text())
    assert partial["completed_reads"] == 1 and not partial["complete"]
    assert partial["p50_s"] == partial["samples"][0]["latency_s"]
    assert partial["error"].startswith("TimeoutError")
    assert quantile([], 0.5) is None


@pytest.mark.parametrize("backend", ["python", "libzstd"])
def test_output_archive_and_sha256_include_partial_marker(tmp_path, monkeypatch, backend):
    zstandard = pytest.importorskip("zstandard")
    if backend == "libzstd":
        library = ctypes.util.find_library("zstd")
        if library is None and os.name == "nt" and git_bash():
            candidate = Path(git_bash()).parents[1] / "mingw64/bin/libzstd.dll"
            if candidate.is_file():
                library = str(candidate)
        if library is None:
            pytest.skip("native libzstd is unavailable")
        monkeypatch.setitem(sys.modules, "zstandard", None)
        monkeypatch.setattr(ctypes.util, "find_library", lambda name: library)
    output = tmp_path / "out"
    output.mkdir()
    (output / "progress.json").write_text('{"status":"time_cap","finished_steps":["install"]}')
    model_dir = output / "fixture"
    model_dir.mkdir()
    (model_dir / "reads.jsonl").write_bytes(b'{"id":"one"}\n')
    archive = tmp_path / "out.tar.zst"
    gpu_runner.pack_results(output, archive)
    compressed = zstandard.ZstdDecompressor().stream_reader(archive.open("rb"))
    with compressed, tarfile.open(fileobj=compressed, mode="r|") as tar:
        members = {member.name: tar.extractfile(member).read() for member in tar}
    assert json.loads(members["out/progress.json"])["status"] == "time_cap"
    assert members["out/fixture/reads.jsonl"] == b'{"id":"one"}\n'
    assert archive.with_name(archive.name + ".sha256").read_text().split()[0] == sha256(archive)


def test_supervisor_caps_subprocess_and_kills_entire_group(tmp_path, monkeypatch):
    killed = []
    waits = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.append((pid, sig)), raising=False)
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)

    class Process:
        pid = 42

        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Process())
    supervisor = gpu_runner.Supervisor(tmp_path, time.monotonic() + 10, time.monotonic() + 5)
    with pytest.raises(gpu_runner.DeadlineReached):
        supervisor.run(["fixture"], tmp_path / "command.log")
    assert 0 < waits[0] <= 5
    assert killed == [(42, signal.SIGTERM), (42, signal.SIGKILL)]
    assert not supervisor.processes


def runner_fixture(
    tmp_path,
    monkeypatch,
    *,
    failure=None,
    cap_after_p1=False,
    with_lora_arm=False,
    with_matched_2x2=False,
    native_seconds=0,
    max_minutes=125,
    allow_partial=False,
    started=100.0,
    cleanup_seconds=0,
    prep_seconds=0,
    priority_seconds=0,
    pack_seconds=0,
    selection_seconds=0,
    clock=None,
):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(gpu_runner, "require_free_ports", lambda ports: None)
    real_version = gpu_runner.importlib.metadata.version
    monkeypatch.setattr(
        gpu_runner.importlib.metadata,
        "version",
        lambda name: "0.30.0" if name == "vllm" else real_version(name),
    )
    now = [started] if clock is None else clock
    monkeypatch.setattr(gpu_runner.time, "monotonic", lambda: now[0])
    datasets = []
    for name, split in (
        ("v2_calibration", "calibration"),
        ("v2_dev", "dev"),
        ("cygnet_calibration", "calibration"),
        ("jevbench_public", "public"),
    ):
        path = tmp_path / (name + ".jsonl")
        rows = []
        for kind, labels in (
            ("choice", ["a", "b"]),
            ("noul", ["no", "yes"]),
            ("score", ["0", "1"]),
        ):
            rows.append(
                {
                    "id": f"{name}/{kind}",
                    "source": name,
                    "case_id": name,
                    "state": name,
                    "split": split,
                    "public": split == "public",
                    "labels": labels,
                    "expected": labels[0],
                    "question": {"type": kind, "criteria": dict.fromkeys(labels, "option")},
                }
            )
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        datasets.append({"name": name, "paths": [str(path)]})
    manifest = {"group_size": 20, "datasets": list(reversed(datasets))}
    if with_matched_2x2:
        from scripts.swift import matched_2x2

        public_rows = [
            json.loads(line)
            for line in (tmp_path / "jevbench_public.jsonl").read_text().splitlines()
        ]
        monkeypatch.setattr(
            matched_2x2,
            "public_items",
            lambda directory: {
                row["id"]: {
                    "tier": "standard",
                    "type": row["question"]["type"],
                    "family": "?",
                    "expected": row["expected"],
                    "labels": row["labels"],
                }
                for row in public_rows
            },
        )
    events, active, commands = [], [], []
    current = [None]

    class Supervisor:
        def __setattr__(self, key, value):
            if key in ("deadline", "work_deadline"):
                events.append((key, value - started))
            object.__setattr__(self, key, value)

        def __init__(self, root, deadline, work_deadline):
            self.deadline = deadline
            self.work_deadline = work_deadline

        def remaining(self):
            remaining = self.work_deadline - now[0]
            if remaining <= 0:
                raise gpu_runner.DeadlineReached("fake work deadline exhausted")
            return remaining

        def run(self, command, log):
            self.remaining()
            commands.append(command)
            assert "pip" not in command
            if command[0] == "nvidia-smi":
                log.write_text("CPU fake GPU receipt")
                now[0] += prep_seconds
            elif "_resolve" in command:
                offset = command.index("_resolve")
                current[0] = command[offset + 1]
                Path(command[offset + 3]).write_text(
                    json.dumps({"model": current[0], "revision": "a" * 40})
                )
            elif "--prepare-checkpoint" in command:
                assert not active
                output = Path(command[command.index("--output") + 1])
                adapter = output / "swift_adapter"
                adapter.mkdir(parents=True)
                (adapter / "adapter_config.json").write_text(
                    json.dumps(
                        {
                            "base_model_name_or_path": "first",
                            "revision": "a" * 40,
                        }
                    )
                )
                (adapter / "adapter_model.safetensors").write_bytes(b"CPU fake adapter")
                (output / "checkpoint.json").write_text(
                    json.dumps(
                        {
                            "checkpoint_path": str(output / "native_snapshot"),
                            "swift_adapter_path": str(adapter),
                        }
                    )
                )
                events.append(("prepare_matched_checkpoint",))
            elif "--cell" in command:
                assert not active, "vLLM must exit before either native HF load"
                output = Path(command[command.index("--output") + 1])
                output.mkdir(parents=True)
                cell = command[command.index("--cell") + 1]
                events.append(("native", cell))
                now[0] += native_seconds
                self.remaining()
                (output / "results.jsonl").write_text(
                    "".join(
                        json.dumps(
                            {
                                "id": row["id"],
                                "pred": row["expected"],
                                "expected": row["expected"],
                                "ok": True,
                            }
                        )
                        + "\n"
                        for row in public_rows
                    )
                )
            elif any(str(part).endswith("parity.py") for part in command):
                if priority_seconds:
                    now[0] += priority_seconds
                    self.remaining()
                phase = command[command.index("--phase") + 1]
                model = command[command.index("--model") + 1]
                if phase == "reference":
                    assert not active
                    assert command[command.index("--hf-device") + 1] == "cuda"
                    events.append(("hf_reference_exit", model))
                    result = {"hf_load_s": 0.1, "complete": True}
                    if failure == "reference":
                        result.update(complete=False, passed=False, error="CPU load failure")
                else:
                    assert active == ["vllm"]
                    events.append(("parity", model))
                    passed = failure != "parity"
                    result = {
                        "complete": True,
                        "passed": passed,
                        "comparison_valid": passed,
                        "samples": [{"diagnostic": True}],
                    }
                Path(command[command.index("--output") + 1]).write_text(json.dumps(result))
                if failure == "reference" and phase == "reference":
                    raise RuntimeError("reference command exited 1")
                if failure == "parity_exit" and phase == "compare":
                    raise RuntimeError("parity command exited 1")
            elif "ayaka.swift.collect" in command:
                assert active == ["vllm"]
                model = command[command.index("--model") + 1]
                variant = command[command.index("--prompt-variant") + 1]
                source = next(
                    part for part in command if part.endswith(".jsonl") and "reads" not in part
                )
                events.append(("collect", model, Path(source).stem, variant))
                reader = FakeReader()
                if "--reasoned" in command:
                    from test_swift_reasoning import TraceReader

                    reader = TraceReader()
                reader.backend = "vllm"
                reader.logprobs_mode = "raw_logits"
                collect(
                    iter_dataset([source]),
                    reader,
                    command[command.index("--output") + 1],
                    model=model,
                    revision=command[command.index("--revision") + 1],
                    prompt_variant=variant,
                    diagnostic="--diagnostic" in command,
                    reasoned="--reasoned" in command,
                    direct_reads=load_reads([command[command.index("--direct-reads") + 1]])
                    if "--reasoned" in command
                    else None,
                )
                if cap_after_p1 and Path(source).stem == "jevbench_public" and variant == "labeled":
                    now[0] = self.work_deadline
            elif "_select" in command:
                events.append(("select",))
                now[0] += selection_seconds
                self.remaining()
                gpu_runner.select_model_results(Path(command[-1]))
            elif "_reasoning_fit" in command or "_reasoning_gate" in command:
                events.append(
                    ("reasoning_gate" if "_reasoning_gate" in command else "reasoning_fit",)
                )
                gpu_runner.reasoning_results(Path(command[-1]), gate="_reasoning_gate" in command)
            elif any(str(part).endswith("latency_probe.py") for part in command):
                assert active == ["vllm", "swift"]
                events.append(("probe", command[command.index("--prompt-variant") + 1]))
                if "--non-public" in command:
                    from ayaka.eval.read_artifact import fingerprint
                    from ayaka.swift.policy import Policy

                    policy = Policy.load(command[command.index("--policy") + 1])
                    result = {
                        "complete": True,
                        "public": False,
                        "split": "dev",
                        "model": command[command.index("--model") + 1],
                        "revision": "a" * 40,
                        "prompt_variant": policy.prompt_variant,
                        "system": "reasoning_route" if policy.reasoning_route else "direct",
                        "router_sha256": fingerprint(policy.reasoning_route)
                        if policy.reasoning_route
                        else None,
                        "concurrency": 1,
                        "units": "seconds",
                        "requested_reads": 3,
                        "completed_reads": 3,
                        "seed": 15,
                        "samples": [{"id": str(i)} for i in range(3)],
                        "p50_s": 0.03,
                        "p95_s": 0.05,
                    }
                else:
                    result = {"complete": True}
                Path(command[command.index("--output") + 1]).write_text(json.dumps(result))
            elif "_pack" in command:
                assert not active
                assert float(command[command.index("--deadline") + 1]) == started + max_minutes * 60
                now[0] += pack_seconds
                self.remaining()
                gpu_runner.pack_results(Path(command[-2]), Path(command[-1]))

        def start(self, command, log):
            commands.append(command)
            if command[0] == "vllm":
                assert not active
                assert command[command.index("--logprobs-mode") + 1] == "raw_logits"
                active.append("vllm")
                events.append(("serve", command[2]))
            else:
                assert active == ["vllm"]
                assert "--revision" in command and "--policy" in command
                active.append("swift")
            return object()

        def health(self, url, process, timeout):
            if failure == "health":
                raise RuntimeError("fake health failure")

        def stop(self, processes=None):
            if processes:
                active.remove("swift")
            else:
                active.clear()
                now[0] += cleanup_seconds

    monkeypatch.setattr(gpu_runner, "Supervisor", Supervisor)
    args = SimpleNamespace(
        output=tmp_path / "out",
        archive=tmp_path / "out.tar.zst",
        max_minutes=max_minutes,
        allow_partial=allow_partial,
        vllm_port=8000,
        swift_port=8009,
        health_timeout=10,
        concurrency=3,
        prompt_variants=list(gpu_runner.FIXED_VARIANTS),
        with_lora_arm=with_lora_arm,
        with_matched_2x2=with_matched_2x2,
        lora_path=tmp_path / "adapter",
        lora_revision="b" * 40,
    )
    if with_lora_arm:
        args.lora_path.mkdir()
        (args.lora_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "first", "revision": "a" * 40})
        )
        (args.lora_path / "adapter.safetensors").write_bytes(b"CPU fake adapter")
    models = gpu_runner.model_specs(["first@pin", "second@pin"], dry_run=False)
    code = gpu_runner.execute(
        args, models, manifest, now[0], gpu_runner.model_options(None, models)
    )
    return code, args, events, commands


@pytest.mark.parametrize("failure", [None, "health", "parity", "parity_exit"])
def test_runner_reference_exit_before_vllm_and_parity_aborts_bulk(tmp_path, monkeypatch, failure):
    code, args, events, commands = runner_fixture(tmp_path, monkeypatch, failure=failure)
    progress = json.loads((args.output / "progress.json").read_text())
    assert code == (1 if failure else 0)
    assert args.archive.is_file()
    assert events.index(("hf_reference_exit", "first")) < events.index(("serve", "first"))
    if failure:
        assert not any(event[0] in ("collect", "probe") for event in events)
        assert progress["comparison_valid"] is False
        assert progress["priorities"][1]["status"] == "skipped"
        diagnostic = json.loads((args.output / "first/parity.json").read_text())
        assert diagnostic["comparison_valid"] is False
        assert diagnostic["hf_load_s"] == 0.1 and diagnostic["vllm_load_s"] >= 0
        if failure != "health":
            assert diagnostic["samples"]
        return
    assert progress["status"] == "complete"
    assert [p["status"] for p in progress["priorities"]] == ["complete"] * 5 + ["skipped"] * 2
    assert len([e for e in events if e[0] == "probe"]) == 4
    assert {e[1] for e in events if e[0] == "probe"} >= set(progress["best_two_variants"])
    assert ("reasoning_fit",) in events and ("reasoning_gate",) in events
    for model in ("first", "second"):
        collected = [e for e in events if e[0] == "collect" and e[1] == model]
        assert len(collected) == (18 if model == "first" else 16)
        assert len([e for e in collected if e[2] == "jevbench_public"]) == 4
        assert events.index(("parity", model)) < events.index(collected[0])
        timings = json.loads((args.output / model / "load_times.json").read_text())
        assert timings["hf_load_s"] == 0.1 and timings["vllm_load_s"] >= 0
    assert events.index(("hf_reference_exit", "second")) > max(
        i for i, e in enumerate(events) if e[0] == "probe"
    )


def test_runner_last_command_at_deadline_is_incomplete_and_later_priorities_skipped(
    tmp_path, monkeypatch
):
    code, args, events, _ = runner_fixture(tmp_path, monkeypatch, cap_after_p1=True)
    assert code == 124
    state = json.loads((args.output / "progress.json").read_text())
    assert state["status"] == "partial"
    assert state["stop_reason"] == "priority_timeout"
    assert [p["status"] for p in state["priorities"]] == [
        "complete",
        "timeout",
        "skipped",
        "skipped",
        "skipped",
        "skipped",
        "skipped",
    ]
    assert len([e for e in events if e[0] == "collect"]) == 16
    assert not any(e[0] == "probe" for e in events)
    assert args.archive.is_file()


def test_optional_lora_arm_only_runs_with_flag_and_records_adapter_hash(tmp_path, monkeypatch):
    code, args, events, commands = runner_fixture(tmp_path, monkeypatch, with_lora_arm=True)
    assert code == 0
    state = json.loads((args.output / "progress.json").read_text())
    assert state["priorities"][5]["status"] == "complete"
    assert len([e for e in events if e[0] == "collect" and e[1] == "ayaka-large-swift"]) == 16
    arm = state["models"]["ayaka_large_lora"]
    assert len(arm["adapter_sha256"]) == 64
    assert arm["revision"] == "a" * 40 and arm["adapter_revision"] == "b" * 40
    assert sum("--lora-modules" in command for command in commands) == 1
    command = next(c for c in commands if "--lora-modules" in c)
    assert command[command.index("--max-lora-rank") + 1] == "64"


def test_matched_p4_selected_public_variant_then_sequential_native_cells(tmp_path, monkeypatch):
    code, args, events, commands = runner_fixture(
        tmp_path, monkeypatch, with_matched_2x2=True, max_minutes=144
    )
    assert code == 0
    state = json.loads((args.output / "progress.json").read_text())
    assert all(p["status"] == "complete" for p in state["priorities"])
    arm_reads = [e for e in events if e[0] == "collect" and e[1] == "ayaka-large-swift"]
    assert arm_reads == [
        ("collect", "ayaka-large-swift", "jevbench_public", state["matched_variant"])
    ]
    native = [e for e in events if e[0] == "native"]
    assert native == [("native", "frozen_native"), ("native", "lora_native")]
    assert events.index(arm_reads[0]) < events.index(native[0]) < events.index(native[1])
    receipt = json.loads((args.output / "matched_2x2/input_files.json").read_text())
    assert receipt["complete"] and receipt["used_for_gates"] is False
    assert set(receipt["sha256"]) == {"frozen_native", "frozen_swift", "lora_native", "lora_swift"}
    assert sum("--prepare-checkpoint" in command for command in commands) == 1


def test_p5_timeout_retains_diagnostic_inputs_and_respects_absolute_cap(tmp_path, monkeypatch):
    code, args, events, _ = runner_fixture(
        tmp_path, monkeypatch, with_matched_2x2=True, max_minutes=144, native_seconds=19 * 60
    )
    assert code == 124
    state = json.loads((args.output / "progress.json").read_text())
    assert state["priorities"][-1]["status"] == "timeout"
    assert state["priorities"][-1]["work_deadline_elapsed_s"] <= 144 * 60 - 120
    assert [e for e in events if e[0] == "native"] == [("native", "frozen_native")]
    receipt = json.loads((args.output / "matched_2x2/input_files.json").read_text())
    assert receipt["complete"] is False and receipt["used_for_fit_or_selection"] is False


def test_reference_failure_keeps_invalid_artifact_and_never_starts_vllm(tmp_path, monkeypatch):
    code, args, events, _ = runner_fixture(tmp_path, monkeypatch, failure="reference")
    assert code == 1 and args.archive.is_file()
    assert not any(event[0] in ("serve", "collect", "probe") for event in events)
    diagnostic = json.loads((args.output / "first/parity.json").read_text())
    assert diagnostic["comparison_valid"] is False
    assert diagnostic["hf_load_s"] == 0.1 and diagnostic["vllm_load_s"] is None
    assert "reference command exited 1" in diagnostic["runner_error"]


def test_admission_includes_environment_all_enabled_work_loads_and_cleanup():
    models = gpu_runner.model_specs(["first@pin", "second@pin"], dry_run=False)
    admission = gpu_runner.admission_plan(models, 75)
    assert admission["planned_minutes"] == 105
    assert admission["environment_prep_minutes"] == 3
    assert admission["cleanup_reserve_seconds"] == 120
    assert [p["load_minutes"] for p in admission["priorities"]] == [5, 0, 0, 0, 5, 5, 4]
    assert admission["refused"] and not any(p["admitted"] for p in admission["priorities"])
    assert gpu_runner.admission_plan(models, 125, with_lora_arm=True)["planned_minutes"] == 125


@pytest.mark.parametrize(
    "minutes, admitted_ids",
    [
        (19, []),
        (20, ["P0"]),
        (45, ["P0"]),
        (55, ["P0", "P1"]),
        (60, ["P0", "P1"]),
        (75, ["P0", "P1", "P1R"]),
        (80, ["P0", "P1", "P1R", "P2"]),
        (105, ["P0", "P1", "P1R", "P2", "P3"]),
    ],
)
def test_allow_partial_admits_a_prefix_including_exact_fit(minutes, admitted_ids):
    models = gpu_runner.model_specs(["first@pin", "second@pin"], dry_run=False)
    admission = gpu_runner.admission_plan(models, minutes, allow_partial=True)
    assert not admission["refused"]
    assert admission["admitted_minutes"] <= minutes
    assert [p["id"] for p in admission["priorities"] if p["admitted"]] == admitted_ids
    for priority in admission["priorities"]:
        if priority["enabled"] and not priority["admitted"]:
            assert priority["status"] == "skipped" and priority["skip_reason"] == "not_admitted"


def test_overbudget_execute_refuses_before_any_environment_or_model_work(tmp_path, monkeypatch):
    models = gpu_runner.model_specs(["first@pin", "second@pin"], dry_run=False)
    args = SimpleNamespace(max_minutes=1, output=tmp_path / "out")
    monkeypatch.setattr(gpu_runner.time, "monotonic", lambda: 0.0)

    def forbidden(*args, **kwargs):
        pytest.fail("over-budget admission must refuse before preparing the environment")

    monkeypatch.setattr(gpu_runner, "Supervisor", forbidden)
    monkeypatch.setattr(gpu_runner, "require_free_ports", forbidden)
    monkeypatch.setattr(gpu_runner.importlib.metadata, "version", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="planned total 105 minutes exceeds --max-minutes 1"):
        gpu_runner.execute(
            args, models, {"datasets": []}, 0.0, gpu_runner.model_options(None, models)
        )
    assert not args.output.exists()


def test_r15_counterexample_60_second_cap_never_extends_any_deadline(tmp_path, monkeypatch):
    # The original .dev audit writes a fixed receipt. Reproduce its zero-origin
    # clock and 1-minute plan here, preserving all historical .dev receipts.
    code, args, events, commands = runner_fixture(
        tmp_path, monkeypatch, max_minutes=1, allow_partial=True, started=0.0, cleanup_seconds=30
    )
    assert code == 0
    observed = [e[1] for e in events if e[0] in ("deadline", "work_deadline")]
    assert max(observed) == 60.0
    assert all(deadline <= 60 for deadline in observed)
    assert all("_pack" in command for command in commands)
    state = json.loads((args.output / "progress.json").read_text())
    assert state["status"] == "partial"
    assert state["hard_deadline_elapsed_s"] == 60
    assert state["finished_steps"] == []
    assert [p["skip_reason"] for p in state["priorities"]] == ["not_admitted"] * 5 + [
        "disabled"
    ] * 2


def test_partial_execution_marks_excluded_priorities_up_front(tmp_path, monkeypatch):
    real_write = gpu_runner.write_json
    initial_states = []

    def capture(path, data):
        if path.name == "progress.json":
            initial_states.append(json.loads(json.dumps(data)))
        real_write(path, data)

    monkeypatch.setattr(gpu_runner, "write_json", capture)
    code, args, events, _ = runner_fixture(
        tmp_path, monkeypatch, max_minutes=45, allow_partial=True
    )
    assert code == 0
    initial = initial_states[0]
    assert [p["status"] for p in initial["priorities"]] == ["pending"] + ["skipped"] * 6
    assert [p["skip_reason"] for p in initial["priorities"]] == [None] + ["not_admitted"] * 4 + [
        "disabled"
    ] * 2
    state = json.loads((args.output / "progress.json").read_text())
    assert state["status"] == "partial" and state["priorities"][0]["status"] == "complete"
    assert not any(e[0] in ("collect", "probe") for e in events)


def test_priority_timeout_stops_work_and_preserves_partial_archive(tmp_path, monkeypatch):
    code, args, events, commands = runner_fixture(
        tmp_path, monkeypatch, priority_seconds=901, max_minutes=105
    )
    assert code == 124
    state = json.loads((args.output / "progress.json").read_text())
    assert state["status"] == "partial"
    assert state["priorities"][0]["status"] == "timeout"
    assert all(p["status"] == "skipped" for p in state["priorities"][1:])
    assert not any(e[0] in ("serve", "collect", "probe") for e in events)
    assert not any(command[0] == "vllm" for command in commands)
    assert args.archive.is_file()
    assert all(e[1] <= 105 * 60 for e in events if e[0] in ("deadline", "work_deadline"))


def test_environment_deadline_is_capped_and_timeout_skips_all_priorities(tmp_path, monkeypatch):
    code, args, events, commands = runner_fixture(tmp_path, monkeypatch, prep_seconds=181)
    assert code == 124
    state = json.loads((args.output / "progress.json").read_text())
    assert state["status"] == "partial" and not state["finished_steps"]
    assert all(p["status"] == "skipped" for p in state["priorities"])
    assert all(command[0] == "nvidia-smi" or "_pack" in command for command in commands)
    assert ("work_deadline", 180.0) in events


def test_cleanup_after_hard_deadline_cannot_launch_pack_or_extend_cap(tmp_path, monkeypatch):
    code, args, events, commands = runner_fixture(
        tmp_path, monkeypatch, max_minutes=1, allow_partial=True, started=0, cleanup_seconds=60
    )
    assert code == 124
    assert not commands
    state = json.loads((args.output / "progress.json").read_text())
    assert state["status"] == "partial" and state["archive_status"] == "timeout"
    assert not args.archive.exists()
    assert max(e[1] for e in events if e[0] in ("deadline", "work_deadline")) == 60


def test_pack_timeout_cannot_report_complete(tmp_path, monkeypatch):
    code, args, _, _ = runner_fixture(tmp_path, monkeypatch, max_minutes=105, pack_seconds=105 * 60)
    assert code == 124
    state = json.loads((args.output / "progress.json").read_text())
    assert state["status"] == "partial" and state["archive_status"] == "timeout"
    assert all(p["status"] == "complete" for p in state["priorities"] if p["enabled"])


def test_selection_timeout_stops_live_backend_and_skips_public_and_latency(tmp_path, monkeypatch):
    code, args, events, _ = runner_fixture(tmp_path, monkeypatch, selection_seconds=35 * 60)
    assert code == 124
    state = json.loads((args.output / "progress.json").read_text())
    assert state["priorities"][1]["status"] == "timeout"
    assert state["status"] == "partial" and state["archive_status"] == "complete"
    assert ("select",) in events
    assert not any(e[0] == "collect" and e[2] == "jevbench_public" for e in events)
    assert not any(e[0] == "probe" for e in events)


def test_next_priority_deadline_is_clamped_to_global_work_limit(tmp_path, monkeypatch):
    now = [0.0]
    real_write = gpu_runner.write_json
    advanced = False

    def slow_checkpoint(path, data):
        nonlocal advanced
        if path.name == "progress.json" and "P1" in data["finished_steps"] and not advanced:
            # Simulate time between priorities: just one second of work remains.
            now[0] = 75 * 60 - gpu_runner.CLEANUP_RESERVE_SECONDS - 1
            advanced = True
        real_write(path, data)

    monkeypatch.setattr(gpu_runner, "write_json", slow_checkpoint)
    code, args, events, _ = runner_fixture(
        tmp_path, monkeypatch, max_minutes=75, allow_partial=True, started=0, clock=now
    )
    assert code == 0
    state = json.loads((args.output / "progress.json").read_text())
    priority = state["priorities"][2]
    assert priority["started_elapsed_s"] == 4379
    assert priority["work_deadline_elapsed_s"] == 4380
    assert priority["work_deadline_elapsed_s"] < priority["started_elapsed_s"] + 20 * 60
    assert all(p.get("work_deadline_elapsed_s", 0) <= 4380 for p in state["priorities"])
    assert all(e[1] <= 4500 for e in events if e[0] in ("deadline", "work_deadline"))
    root = args.output / "first"
    assert json.loads((root / "policy.json").read_text()).get("reasoning_route") is None
    assert (root / "reasoning_candidate.policy.json").exists()
    assert ("reasoning_gate",) not in events
    assert not any(e[0] == "probe" for e in events)


def test_supervisor_cleanup_grace_is_bounded_by_absolute_deadline(tmp_path, monkeypatch):
    now = [59.0]
    waits, killed = [], []
    monkeypatch.setattr(gpu_runner.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.append((pid, sig)), raising=False)
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)

    class Process:
        pid = 42

        def wait(self, timeout):
            waits.append(timeout)
            now[0] += timeout
            raise subprocess.TimeoutExpired("fixture", timeout)

    supervisor = gpu_runner.Supervisor(tmp_path, 60.0, 0.0)
    stream = io.BytesIO()
    supervisor.processes[Process()] = stream
    supervisor.stop()
    assert waits == [1.0, 0.0] and now[0] == 60.0
    assert killed == [(42, signal.SIGTERM), (42, signal.SIGKILL)]
    assert stream.closed and not supervisor.processes


def test_dry_run_allow_partial_prints_admission_prefix(capsys):
    assert gpu_runner.main(["--dry-run", "--max-minutes", "45", "--allow-partial"]) == 0
    output = capsys.readouterr().out
    assert "Admission: partial prefix" in output
    assert "admitted total: 20 minutes" in output
    rows = [line for line in output.splitlines() if line.startswith(("P0 ", "P1 ", "P2 ", "P3 "))]
    assert "admitted" in rows[0]
    assert all("skipped: not_admitted" in line for line in rows[1:])


def test_system_latency_probe_non_public_standard_judge_only(tmp_path, monkeypatch):
    from test_swift_reasoning import always_router

    from ayaka.eval.read_artifact import fingerprint
    from ayaka.swift.policy import Policy

    path = tmp_path / "dev.jsonl"
    rows = [
        {**record(i), "id": str(i), "public": False, "split": "dev", "tier": tier}
        for i, tier in enumerate(("hard", "standard", "judge"))
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    calls = []

    def urlopen(request, timeout):
        calls.append(json.loads(request.data))
        return io.BytesIO(b'{"answers":{"probe":{}}}')

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    policy = Policy(reasoning_route=always_router())
    result = probe(
        [path],
        "http://fake",
        tmp_path / "latency.json",
        model="frozen",
        revision="a" * 40,
        reads=2,
        non_public=True,
        policy=policy,
    )
    assert {s["id"] for s in result["samples"]} == {"1", "2"}
    assert result["public"] is False and result["split"] == "dev"
    assert result["system"] == "reasoning_route"
    assert result["router_sha256"] == fingerprint(policy.reasoning_route)
    assert len(calls) == 2
    path.write_text(json.dumps({**rows[1], "public": True}) + "\n")
    with pytest.raises(ValueError, match="non-public dev"):
        probe(
            [path],
            "http://fake",
            tmp_path / "refused.json",
            model="frozen",
            revision="a" * 40,
            reads=1,
            non_public=True,
        )
    assert len(calls) == 2
