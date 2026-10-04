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
    assert output.count("Bulk total: 4120 decisions, 4888 model reads; serial probe: 200") == 6
    assert "24720 bulk decisions, 29328 bulk model reads + 1200 serial HTTP requests" in output
    assert "Prompt variants: min,cygnet,rules" in output
    assert "Qwen/Qwen3.5-4B" not in output
    assert "<set revision via env/--model>" in output
    assert "75 minutes" in output


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


def test_dry_run_variant_subset_and_invalid_lists(capsys):
    assert (
        gpu_runner.main(
            [
                "--dry-run",
                "--manifest",
                str(DEFAULT_MANIFEST),
                "--model",
                "fixture@pin",
                "--prompt-variants",
                "min,rules",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Prompt variants: min,rules" in output
    assert "8240 bulk decisions, 9776 bulk model reads + 400 serial HTTP requests" in output
    for invalid in ("", "min,min", "min,unknown", "min,"):
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
    assert command[command.index("--logprobs-mode") + 1] == "processed_logprobs"
    assert command[command.index("--max-logprobs") + 1] == "26"
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

    def from_pretrained(model, revision):
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
    service = DecisionService(reader, "fake", max_parallel=1, prompt_variant="cygnet")
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


@pytest.mark.parametrize("failure", ["time_cap", "failure"])
def test_live_runner_packs_after_interrupted_install_without_launching_gpu(
    tmp_path, monkeypatch, failure
):
    # Exercise the finalizer on Windows with a fake supervisor; no installs/GPU.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(gpu_runner, "require_free_ports", lambda ports: None)
    commands = []
    stops = []

    class Supervisor:
        def __init__(self, root, deadline, work_deadline):
            self.work_deadline = work_deadline

        def run(self, command, log):
            commands.append(command)
            if command[0] == "nvidia-smi":
                log.write_text("CPU fixture GPU name, driver, memory\n")
                return
            if "_pack" in command:
                gpu_runner.pack_results(Path(command[-2]), Path(command[-1]))
                return
            if failure == "time_cap":
                raise gpu_runner.DeadlineReached("fake time cap")
            raise RuntimeError("fake install failure")

        def stop(self):
            stops.append(True)

    monkeypatch.setattr(gpu_runner, "Supervisor", Supervisor)
    args = SimpleNamespace(
        output=tmp_path / "out",
        archive=tmp_path / "out.tar.zst",
        max_minutes=1,
        vllm_port=8000,
        swift_port=8009,
        prompt_variants=["min", "cygnet", "rules"],
    )
    models = gpu_runner.model_specs(["fixture@pin"], dry_run=False)
    options = gpu_runner.model_options(None, models)
    code = gpu_runner.execute(args, models, {"datasets": []}, time.monotonic(), options)
    assert code == (124 if failure == "time_cap" else 1)
    progress = json.loads((args.output / "progress.json").read_text())
    assert progress["status"] == ("time_cap" if failure == "time_cap" else "failed")
    assert stops and args.archive.is_file()
    env = json.loads((args.output / "fixture/env.txt").read_text())
    assert "CPU fixture GPU name" in env["gpu_name_driver_memory"]
    assert env["requested_revision"] == "pin"
    assert env["resolved_revision"] is None
    assert not any(command[:2] == ["vllm", "serve"] for command in commands)


@pytest.mark.parametrize("first_failure", [None, "health", "parity"])
def test_runner_model_lifecycle_collections_serial_probe_and_cleanup(
    tmp_path, monkeypatch, first_failure
):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(gpu_runner, "require_free_ports", lambda ports: None)
    real_version = gpu_runner.importlib.metadata.version
    monkeypatch.setattr(
        gpu_runner.importlib.metadata,
        "version",
        lambda package: "0.30.0" if package == "vllm" else real_version(package),
    )
    dataset = tmp_path / "public.jsonl"
    dataset.write_text("\n".join(json.dumps(record(index)) for index in range(231)))
    manifest = {
        "group_size": 20,
        "datasets": [
            {"name": "calibration", "paths": [str(dataset)]},
            {"name": "jevbench_public", "paths": [str(dataset)]},
        ],
    }
    events = []
    active = []
    started_models = []

    class Supervisor:
        def __init__(self, root, deadline, work_deadline):
            self.work_deadline = work_deadline

        def remaining(self):
            return 60

        def run(self, command, log):
            if command[0] == "nvidia-smi":
                log.write_text("CPU fixture GPU name, driver, memory")
            elif "_resolve" in command:
                offset = command.index("_resolve")
                Path(command[offset + 3]).write_text(
                    json.dumps(
                        {
                            "model": command[offset + 1],
                            "revision": "a" * 40,
                        }
                    )
                )
            elif "ayaka.swift.collect" in command:
                assert active == ["vllm"]
                assert command[command.index("--concurrency") + 1] == "3"
                events.append(("collect", started_models[-1]))
                variant = command[command.index("--prompt-variant") + 1]
                events.append(("collect_variant", variant))
                collect(
                    iter_dataset([dataset]),
                    FakeReader(),
                    command[command.index("--output") + 1],
                    concurrency=3,
                    model=command[command.index("--model") + 1],
                    revision=command[command.index("--revision") + 1],
                    prompt_variant=variant,
                )
            elif any(str(part).endswith("parity.py") for part in command):
                assert active == ["vllm"]
                assert command[command.index("--n") + 1] == "50"
                assert command[command.index("--max-abs") + 1] == "0.02"
                assert command[command.index("--min-argmax-agreement") + 1] == "0.98"
                events.append(("parity", started_models[-1]))
                Path(command[command.index("--output") + 1]).write_text(
                    json.dumps(
                        {
                            "complete": True,
                            "passed": not (
                                first_failure == "parity" and started_models == ["first"]
                            ),
                        }
                    )
                )
            elif any(str(part).endswith("latency_probe.py") for part in command):
                assert active == ["vllm", "swift"]
                assert command[command.index("--reads") + 1] == "200"
                assert command[command.index("--url") + 1].endswith(":8009")
                events.append(("probe", started_models[-1]))
                variant = command[command.index("--prompt-variant") + 1]
                events.append(("probe_variant", variant))
                Path(command[command.index("--output") + 1]).write_text('{"complete":true}')
            elif "_pack" in command:
                assert not active
                gpu_runner.pack_results(Path(command[-2]), Path(command[-1]))

        def start(self, command, log):
            if command[0] == "vllm":
                assert not active, "previous model's GPU workers must be stopped"
                started_models.append(command[2])
                active.append("vllm")
                events.append(("serve", command[2]))
            else:
                assert active == ["vllm"]
                assert command[command.index("--max-parallel") + 1] == "1"
                assert command[command.index("--prompt-variant") + 1] in args.prompt_variants
                active.append("swift")
            return object()

        def health(self, url, process, timeout):
            if first_failure == "health" and started_models == ["first"]:
                raise RuntimeError("fake first model health failure")

        def stop(self, processes=None):
            if processes:
                assert active == ["vllm", "swift"]
                active.remove("swift")
                return
            if active:
                events.append(("stop", started_models[-1]))
            active.clear()

    monkeypatch.setattr(gpu_runner, "Supervisor", Supervisor)
    models = gpu_runner.model_specs(["first@pin", "second@pin"], dry_run=False)
    args = SimpleNamespace(
        output=tmp_path / "out",
        archive=tmp_path / "out.tar.zst",
        max_minutes=1,
        vllm_port=8000,
        swift_port=8009,
        health_timeout=10,
        concurrency=3,
        prompt_variants=["min", "cygnet", "rules"],
    )
    code = gpu_runner.execute(
        args, models, manifest, time.monotonic(), gpu_runner.model_options(None, models)
    )
    assert code == (1 if first_failure else 0)
    if first_failure:
        assert ("collect", "first") not in events
        assert ("probe", "first") not in events
    assert events.index(("stop", "first")) < events.index(("serve", "second"))
    assert events[-1] == ("stop", "second")
    assert args.archive.is_file()
    progress = json.loads((args.output / "progress.json").read_text())
    assert progress["status"] == ("partial_failure" if first_failure else "complete")
    assert "second/rules/latency" in progress["finished_steps"]
    assert "second/servers_stopped" in progress["finished_steps"]
    for variant in args.prompt_variants:
        rows = load_reads([args.output / f"second/{variant}/jevbench_public.reads.jsonl"])
        assert len(rows) == 231
        assert all(
            row["model"] == "second"
            and row["revision"] == "a" * 40
            and row["prompt_variant"] == variant
            for row in rows
        )
    assert events.count(("serve", "second")) == 1
    second_events = events[events.index(("serve", "second")) :]
    assert second_events.index(("parity", "second")) < second_events.index(("collect", "second"))
    assert [
        event[1] for event in second_events if event[0] == "probe_variant"
    ] == args.prompt_variants
    assert [event[1] for event in second_events if event[0] == "collect_variant"] == [
        variant for variant in args.prompt_variants for _ in manifest["datasets"]
    ]
    assert max(
        i for i, event in enumerate(second_events) if event[0] == "collect"
    ) < second_events.index(("probe", "second"))
    env = json.loads((args.output / "second/env.txt").read_text())
    assert env["resolved_revision"] == "a" * 40
    assert env["versions"]["vllm"] == "0.30.0"
