"""Bounded GPU collection supervisor; --dry-run needs only Python and the inputs."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import ctypes.util
import datetime as dt
import importlib.metadata
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.swift.prompt import PROMPT_VARIANTS  # noqa: E402
from scripts.swift.inputs import (  # noqa: E402
    DEFAULT_MANIFEST,
    PACKED_MANIFEST,
    REPO,
    inventory,
    sha256,
)
from scripts.swift.latency_probe import write_json  # noqa: E402

VLLM_VERSION = "0.30.0"
GEMMA_12B_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
DEFAULT_KWARGS = {"enable_thinking": False}


def prompt_variants(value: str) -> list[str]:
    variants = [variant.strip() for variant in value.split(",")]
    if (
        not variants
        or len(set(variants)) != len(variants)
        or any(variant not in PROMPT_VARIANTS for variant in variants)
    ):
        raise argparse.ArgumentTypeError("use distinct comma-separated variants: min,cygnet,rules")
    return variants


def model_options(path: Path | None, models: list[dict]) -> dict:
    overrides = json.loads(path.read_text(encoding="utf-8")) if path else {}
    if not isinstance(overrides, dict):
        raise ValueError("model options must be an object keyed by model id")
    result = {}
    reserved = {
        "--revision",
        "--tokenizer-revision",
        "--dtype",
        "--max-model-len",
        "--gpu-memory-utilization",
        "--enable-prefix-caching",
        "--no-enable-prefix-caching",
        "--host",
        "--port",
        "--served-model-name",
        "--logprobs-mode",
        "--max-logprobs",
    }
    for model in models:
        options = overrides.get(model["model"], {})
        if not isinstance(options, dict):
            raise ValueError("each model's options must be an object")
        kwargs = options.get("chat_template_kwargs", DEFAULT_KWARGS.copy())
        extra = options.get("vllm_args", [])
        if (
            not isinstance(kwargs, dict)
            or not isinstance(extra, list)
            or not all(isinstance(value, str) for value in extra)
        ):
            raise ValueError("model options need a kwargs object and a list of vllm_args strings")
        if any(value.split("=", 1)[0] in reserved for value in extra):
            raise ValueError("extra vllm_args must not override the pinned serving settings")
        result[model["model"]] = {"chat_template_kwargs": kwargs, "vllm_args": extra}
    return result


class DeadlineReached(Exception):
    pass


def model_specs(requested: list[str] | None, *, dry_run: bool) -> list[dict]:
    if requested is None:
        requested = [
            f"google/gemma-4-12B-it@{GEMMA_12B_REVISION}",
            "google/gemma-4-E4B-it@" + os.environ.get("SWIFT_GEMMA_E4B_REVISION", ""),
        ]
    result = []
    slugs = set()
    for value in requested:
        model, separator, revision = value.rpartition("@")
        if not separator or not model:
            raise ValueError("models must use MODEL@REVISION")
        if not revision:
            if not dry_run:
                raise ValueError(
                    f"{model} needs a revision: --model MODEL@REVISION or SWIFT_GEMMA_E4B_REVISION"
                )
            revision = "<set revision via env/--model>"
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", model).strip("_")
        if not slug or slug in slugs:
            raise ValueError("use distinct models with distinct output slugs")
        slugs.add(slug)
        result.append({"model": model, "requested_revision": revision, "slug": slug})
    return result


def serve_command(model: dict, port: int, options: dict) -> list[str]:
    return [
        "vllm",
        "serve",
        model["model"],
        "--revision",
        model["revision"],
        "--tokenizer-revision",
        model["revision"],
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "0.90",
        "--enable-prefix-caching",
        "--logprobs-mode",
        "processed_logprobs",
        "--max-logprobs",
        "26",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        model["model"],
        *options.get("vllm_args", []),
    ]


def plan(
    models: list[dict],
    manifest: dict,
    concurrency: int,
    minutes: float,
    options: dict,
    variants: tuple[str, ...] | list[str] = PROMPT_VARIANTS,
    *,
    parity_n: int = 50,
    parity_max_abs: float = 0.02,
    parity_min_agreement: float = 0.98,
    parity_hf_device: str = "cpu",
) -> str:
    lines = [
        f"Install: {sys.executable} -m pip install vllm=={VLLM_VERSION}",
        f"Install: {sys.executable} -m pip install --no-deps -e . (no extras)",
        f"Wall-clock cap: {minutes:g} minutes, including install/startup; cleanup/pack reserve <=30s",
        f"Bulk concurrency: {concurrency}; serial Swift HTTP probe: 200 requests/model/variant",
        f"Prompt variants: {','.join(variants)}; reuse one vLLM server per model",
        "Serve each model alone: bf16, context=16384, GPU=0.90, prefix caching enabled",
        'Chat template kwargs default per model: {"enable_thinking": false}',
        f"HF/vLLM parity per variant: N={parity_n}, max-abs<={parity_max_abs:g}, argmax>={parity_min_agreement:g}; HF device={parity_hf_device}",
        f"Parity forwards total: {2 * parity_n * len(models) * len(variants)} (one HF and one vLLM per item)",
    ]
    decisions = sum(dataset["items"] for dataset in manifest["datasets"])
    reads = sum(dataset["reads"] for dataset in manifest["datasets"])
    for model in models:
        lines.append(f"\n{model['model']}@{model['requested_revision']} -> out/{model['slug']}/")
        lines.append(f"  Options: {json.dumps(options[model['model']])}")
        for variant in variants:
            lines.append(f"  Variant {variant} -> {variant}/")
            for dataset in manifest["datasets"]:
                lines.append(
                    f"    {dataset['name']}: {dataset['records']} rows, "
                    f"{dataset['items']} decisions, {dataset['reads']} model reads"
                )
            lines.append(
                f"  Bulk total: {decisions} decisions, {reads} model reads; serial probe: 200"
            )
        lines.append(
            f"  Model total: {len(variants) * decisions} decisions, "
            f"{len(variants) * reads} model reads; serial probes: {len(variants) * 200}"
        )
    lines.extend(
        [
            f"\nAll models: {len(models) * len(variants) * decisions} bulk decisions, "
            f"{len(models) * len(variants) * reads} bulk model reads + "
            f"{len(models) * len(variants) * 200} serial HTTP requests",
            "Health -> HF/vLLM parity -> bulk every dataset/variant -> Swift serial probe per variant -> stop vLLM",
            "Output: parity.json, variant/{dataset.reads.jsonl,latency.json,swift.log}, vllm.log, env.txt, progress.json",
            "Finally: out.tar.zst + out.tar.zst.sha256, including partial results on cap/failure",
        ]
    )
    return "\n".join(lines)


def resolve_model(model: str, requested: str, output: Path, kwargs: dict) -> None:
    """Run only on the GPU host, in a supervised child with a bounded lifetime."""
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    revision = HfApi().model_info(model, revision=requested).sha
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Hub did not resolve the requested revision to a commit SHA")
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    messages = [
        {"role": "system", "content": "Answer with only the option letter."},
        {"role": "user", "content": "Choose: A. yes B. no"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    verification = "rendered successfully at resolved revision"
    if model == "Qwen/Qwen3.5-4B" and kwargs == DEFAULT_KWARGS:
        thinking = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        if rendered == thinking or not re.search(r"<think>\s*</think>\s*$", rendered):
            raise ValueError("pinned Qwen3.5 template did not disable thinking")
        verification = "enable_thinking=false emits an empty closed think block; true differs"
    write_json(
        output,
        {
            "model": model,
            "requested_revision": requested,
            "revision": revision,
            "tokenizer_revision": revision,
            "chat_template_kwargs": kwargs,
            "template_verification": verification,
            "generation_prefix_tail": rendered[-200:],
        },
    )


def pack_results(output: Path, archive: Path, *, deadline: float | None = None) -> None:
    """Pack without depending on a successful pip install (Ubuntu libzstd fallback)."""
    temporary = archive.with_suffix(".tar.tmp")
    try:
        with tarfile.open(temporary, "w") as tar:
            for path in sorted(output.rglob("*")):
                if deadline is not None and time.monotonic() >= deadline:
                    raise DeadlineReached("packing deadline reached")
                if path.is_file() and not path.is_symlink():
                    tar.add(path, arcname="out/" + path.relative_to(output).as_posix())
        try:
            import zstandard
        except ImportError:
            library = ctypes.util.find_library("zstd")
            if not library:
                raise RuntimeError("packing requires Python zstandard or Ubuntu libzstd") from None
            codec = ctypes.CDLL(library)
            codec.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
            codec.ZSTD_compressBound.restype = ctypes.c_size_t
            codec.ZSTD_compress.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            codec.ZSTD_compress.restype = ctypes.c_size_t
            codec.ZSTD_isError.argtypes = [ctypes.c_size_t]
            codec.ZSTD_isError.restype = ctypes.c_uint
            source = temporary.read_bytes()
            target = ctypes.create_string_buffer(codec.ZSTD_compressBound(len(source)))
            size = codec.ZSTD_compress(target, len(target), source, len(source), 3)
            if codec.ZSTD_isError(size):
                raise RuntimeError("libzstd compression failed") from None
            archive.write_bytes(target.raw[:size])
        else:
            with temporary.open("rb") as source, archive.open("wb") as target:
                zstandard.ZstdCompressor(level=3).copy_stream(source, target)
        archive.with_name(archive.name + ".sha256").write_text(
            f"{sha256(archive)}  {archive.name}\n", encoding="ascii"
        )
    finally:
        temporary.unlink(missing_ok=True)


class Supervisor:
    def __init__(self, root: Path, deadline: float, work_deadline: float):
        self.root = root
        self.deadline = deadline
        self.work_deadline = work_deadline
        self.processes: dict[subprocess.Popen, object] = {}

    def remaining(self) -> float:
        remaining = self.work_deadline - time.monotonic()
        if remaining <= 0:
            raise DeadlineReached("wall-clock work budget exhausted")
        return remaining

    def start(self, command: list[str], log: Path) -> subprocess.Popen:
        self.remaining()
        stream = log.open("ab", buffering=0)
        stream.write(("\n$ " + shlex.join(command) + "\n").encode())
        try:
            process = subprocess.Popen(
                command,
                cwd=self.root,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            stream.close()
            raise
        self.processes[process] = stream
        return process

    def stop(self, processes: list[subprocess.Popen] | None = None) -> None:
        selected = list(self.processes) if processes is None else processes
        for process in selected:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        end = min(self.deadline, time.monotonic() + 3)
        # One shared grace period, even if several process groups need stopping.
        for process in selected:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=max(0, end - time.monotonic()))
        for process in selected:
            # Kill the whole group even if its leader exited but GPU workers did not.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=max(0, end - time.monotonic()))
            stream = self.processes.pop(process, None)
            if stream is not None:
                stream.close()

    def run(self, command: list[str], log: Path) -> None:
        process = self.start(command, log)
        try:
            code = process.wait(timeout=self.remaining())
            if code:
                raise RuntimeError(f"command exited {code}; see {log}")
        except subprocess.TimeoutExpired as exc:
            raise DeadlineReached("command reached the work deadline") from exc
        finally:
            self.stop([process])

    def health(self, url: str, process: subprocess.Popen, timeout: float) -> None:
        import urllib.error
        import urllib.request

        end = time.monotonic() + min(timeout, self.remaining())
        while time.monotonic() < end:
            if process.poll() is not None:
                raise RuntimeError(f"server exited {process.returncode} before health: {url}")
            try:
                with urllib.request.urlopen(url, timeout=min(2, self.remaining())) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(min(0.5, self.remaining()))
        self.remaining()
        raise RuntimeError(f"server health timeout: {url}")


def environment(model: dict, options: dict, gpu: str) -> str:
    versions = {}
    for package in ("vllm", "torch", "transformers", "ayaka"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return (
        json.dumps(
            {
                "python": sys.version,
                "gpu_name_driver_memory": gpu,
                "versions": versions,
                "model": model["model"],
                "requested_revision": model["requested_revision"],
                "resolved_revision": model.get("revision"),
                "tokenizer_revision": model.get("revision"),
                "options": options,
                "vllm_pin": VLLM_VERSION,
            },
            indent=2,
        )
        + "\n"
    )


def require_free_ports(ports: tuple[int, ...]) -> None:
    for port in ports:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise ValueError(f"port {port} is already in use; choose different ports") from exc


def execute(
    args: argparse.Namespace,
    models: list[dict],
    manifest: dict,
    started: float,
    options: dict,
) -> int:
    if sys.platform != "linux":
        raise ValueError("live collection requires Linux; --dry-run works on CPU/Windows")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be empty; use --output for a fresh run")
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = started + args.max_minutes * 60
    reserve = min(30, args.max_minutes * 60 * 0.2)
    supervisor = Supervisor(REPO, deadline, deadline - reserve)
    state = {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "max_minutes": args.max_minutes,
        "cleanup_pack_reserve_s": reserve,
        "status": "running",
        "current_step": "install",
        "finished_steps": [],
        "models": {},
        "prompt_variants": args.prompt_variants,
    }
    marker = args.output / "progress.json"
    gpu = "unavailable (install/startup interrupted)"

    def checkpoint(step: str | None = None) -> None:
        if step:
            state["finished_steps"].append(step)
        write_json(marker, state)

    def begin(step: str) -> None:
        supervisor.remaining()
        state["current_step"] = step
        checkpoint()

    def interrupted(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    code = 0
    try:
        checkpoint()
        for model in models:
            directory = args.output / model["slug"]
            directory.mkdir()
            (directory / "env.txt").write_text(environment(model, options[model["model"]], gpu))
            (directory / "vllm.log").touch()
        require_free_ports((args.vllm_port, args.swift_port))
        supervisor.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            args.output / "gpu.txt",
        )
        gpu = (args.output / "gpu.txt").read_text()
        supervisor.run(
            [sys.executable, "-m", "pip", "install", f"vllm=={VLLM_VERSION}"],
            args.output / "install.log",
        )
        supervisor.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "-e", "."],
            args.output / "install.log",
        )
        if importlib.metadata.version("vllm") != VLLM_VERSION:
            raise RuntimeError("installed vLLM version differs from requested pin")
        checkpoint("install")
        for model in models:
            directory = args.output / model["slug"]
            model_options = options[model["model"]]
            kwargs = json.dumps(model_options["chat_template_kwargs"])
            try:
                begin(model["slug"] + "/resolve")
                supervisor.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "_resolve",
                        model["model"],
                        model["requested_revision"],
                        str(directory / "revision.json"),
                        kwargs,
                    ],
                    directory / "resolve.log",
                )
                resolved = json.loads((directory / "revision.json").read_text())
                model["revision"] = resolved["revision"]
                state["models"][model["slug"]] = resolved
                (directory / "env.txt").write_text(environment(model, model_options, gpu))
                checkpoint(state["current_step"])
                begin(model["slug"] + "/vllm_health")
                backend = supervisor.start(
                    serve_command(model, args.vllm_port, model_options), directory / "vllm.log"
                )
                backend_url = f"http://127.0.0.1:{args.vllm_port}"
                supervisor.health(backend_url + "/health", backend, args.health_timeout)
                checkpoint(state["current_step"])
                begin(model["slug"] + "/parity")
                supervisor.run(
                    [
                        sys.executable,
                        str(REPO / "scripts/swift/parity.py"),
                        *dict.fromkeys(
                            path for dataset in manifest["datasets"] for path in dataset["paths"]
                        ),
                        "--model",
                        model["model"],
                        "--revision",
                        model["revision"],
                        "--vllm-url",
                        backend_url,
                        "--chat-template-kwargs",
                        kwargs,
                        "--n",
                        str(getattr(args, "parity_n", 50)),
                        "--max-abs",
                        str(getattr(args, "parity_max_abs", 0.02)),
                        "--min-argmax-agreement",
                        str(getattr(args, "parity_min_agreement", 0.98)),
                        "--hf-device",
                        getattr(args, "parity_hf_device", "cpu"),
                        "--output",
                        str(directory / "parity.json"),
                        "--prompt-variants",
                        *args.prompt_variants,
                    ],
                    directory / "parity.log",
                )
                parity = json.loads((directory / "parity.json").read_text())
                if parity.get("complete") is not True or parity.get("passed") is not True:
                    raise RuntimeError("HF/vLLM parity failed; collection aborted")
                checkpoint(state["current_step"])
                for variant in args.prompt_variants:
                    variant_directory = directory / variant
                    variant_directory.mkdir()
                    for dataset in manifest["datasets"]:
                        begin(model["slug"] + "/" + variant + "/" + dataset["name"])
                        supervisor.run(
                            [
                                sys.executable,
                                "-m",
                                "ayaka.swift.collect",
                                *dataset["paths"],
                                "--output",
                                str(variant_directory / (dataset["name"] + ".reads.jsonl")),
                                "--prompt-variant",
                                variant,
                                "--backend",
                                "vllm",
                                "--vllm-url",
                                backend_url,
                                "--model",
                                model["model"],
                                "--revision",
                                model["revision"],
                                "--concurrency",
                                str(args.concurrency),
                                "--chat-template-kwargs",
                                kwargs,
                                "--group-size",
                                str(manifest["group_size"]),
                            ],
                            variant_directory / "collect.log",
                        )
                        checkpoint(state["current_step"])
                for variant in args.prompt_variants:
                    variant_directory = directory / variant
                    begin(model["slug"] + "/" + variant + "/swift_health")
                    swift = supervisor.start(
                        [
                            sys.executable,
                            "-m",
                            "ayaka.swift.server",
                            "--prompt-variant",
                            variant,
                            "--backend",
                            "vllm",
                            "--vllm-url",
                            backend_url,
                            "--model",
                            model["model"],
                            "--chat-template-kwargs",
                            kwargs,
                            "--host",
                            "127.0.0.1",
                            "--port",
                            str(args.swift_port),
                            "--max-parallel",
                            "1",
                            "--group-size",
                            str(manifest["group_size"]),
                        ],
                        variant_directory / "swift.log",
                    )
                    swift_url = f"http://127.0.0.1:{args.swift_port}"
                    supervisor.health(swift_url + "/health", swift, args.health_timeout)
                    checkpoint(state["current_step"])
                    public = next(d for d in manifest["datasets"] if d["name"] == "jevbench_public")
                    begin(model["slug"] + "/" + variant + "/latency")
                    supervisor.run(
                        [
                            sys.executable,
                            str(REPO / "scripts/swift/latency_probe.py"),
                            *public["paths"],
                            "--url",
                            swift_url,
                            "--output",
                            str(variant_directory / "latency.json"),
                            "--model",
                            model["model"],
                            "--revision",
                            model["revision"],
                            "--prompt-variant",
                            variant,
                            "--reads",
                            "200",
                        ],
                        variant_directory / "latency.log",
                    )
                    checkpoint(state["current_step"])
                    supervisor.stop([swift])
            except (RuntimeError, ValueError) as exc:
                state.setdefault("errors", []).append(
                    {"step": state["current_step"], "error": str(exc)}
                )
                code = 1
                checkpoint()
            finally:
                supervisor.stop()
                checkpoint(model["slug"] + "/servers_stopped")
        state["status"] = "complete" if code == 0 else "partial_failure"
    except DeadlineReached as exc:
        state.update(status="time_cap", error=str(exc))
        code = 124
    except KeyboardInterrupt as exc:
        state.update(status="interrupted", error=str(exc))
        code = 130
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        code = 1
    finally:
        # Ignore further TERM/INT during the reserved, bounded cleanup/packing phase.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        supervisor.stop()
        for model in models:
            directory = args.output / model["slug"]
            if directory.exists():
                (directory / "env.txt").write_text(environment(model, options[model["model"]], gpu))
        state.update(
            elapsed_s=time.monotonic() - started,
            stopped_at_step=state["current_step"],
            current_step="finished",
        )
        checkpoint()
        write_json(args.output / "inputs.manifest.json", manifest)
        # Supervise compression too, so the reserve is part of the hard cap.
        supervisor.work_deadline = deadline
        try:
            supervisor.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "_pack",
                    str(args.output),
                    str(args.archive),
                ],
                args.archive.with_name(args.archive.name + ".pack.log"),
            )
        finally:
            signal.signal(signal.SIGTERM, previous)
            signal.signal(signal.SIGINT, old_int)
        print(f"{state['status']}: packed {args.archive} (+ .sha256)", flush=True)
    return code


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_resolve":
        resolve_model(argv[1], argv[2], Path(argv[3]), json.loads(argv[4]))
        return 0
    if argv and argv[0] == "_pack":
        pack_results(Path(argv[1]), Path(argv[2]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prompt-variants",
        type=prompt_variants,
        default="min,cygnet,rules",
        help="comma-separated variants measured on the same vLLM server",
    )
    parser.add_argument(
        "--model", action="append", help="MODEL@SHA/tag; repeat to replace defaults"
    )
    parser.add_argument("--model-options", type=Path, help="JSON object keyed by model id")
    packed = REPO / PACKED_MANIFEST
    parser.add_argument(
        "--manifest", type=Path, default=packed if packed.exists() else DEFAULT_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=REPO / "out")
    parser.add_argument("--archive", type=Path, default=REPO / "out.tar.zst")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--max-minutes", type=float, default=os.environ.get("SWIFT_MAX_MINUTES", "75")
    )
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--swift-port", type=int, default=8009)
    parser.add_argument("--health-timeout", type=float, default=600)
    parser.add_argument("--parity-n", type=int, default=50)
    parser.add_argument("--parity-max-abs", type=float, default=0.02)
    parser.add_argument("--parity-min-agreement", type=float, default=0.98)
    parser.add_argument(
        "--parity-hf-device", default="cpu", help="CPU avoids the active vLLM GPU allocation"
    )
    args = parser.parse_args(argv)
    try:
        if (
            args.concurrency < 1
            or args.max_minutes <= 0
            or args.health_timeout <= 0
            or not math.isfinite(args.max_minutes)
            or not math.isfinite(args.health_timeout)
        ):
            raise ValueError("concurrency, max minutes, and health timeout must be positive")
        if args.swift_port == args.vllm_port:
            raise ValueError("Swift and vLLM ports must differ")
        if (
            args.parity_n < 1
            or not math.isfinite(args.parity_max_abs)
            or args.parity_max_abs < 0
            or not 0 <= args.parity_min_agreement <= 1
        ):
            raise ValueError("invalid parity count or thresholds")
        if not all(1 <= port <= 65535 for port in (args.swift_port, args.vllm_port)):
            raise ValueError("ports must be in 1..65535")
        if args.archive.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("archive must be outside the output directory")
        args.output = args.output.resolve()
        args.archive = args.archive.resolve()
        if not args.dry_run:
            args.archive.parent.mkdir(parents=True, exist_ok=True)
        models = model_specs(args.model, dry_run=args.dry_run)
        options = model_options(args.model_options, models)
        manifest, _ = inventory(REPO, args.manifest)
        if not any(
            d["name"] == "jevbench_public" and d["items"] >= 200 for d in manifest["datasets"]
        ):
            raise ValueError(
                "manifest must include >=200 distinct jevbench_public items for latency"
            )
        print(
            plan(
                models,
                manifest,
                args.concurrency,
                args.max_minutes,
                options,
                args.prompt_variants,
                parity_n=args.parity_n,
                parity_max_abs=args.parity_max_abs,
                parity_min_agreement=args.parity_min_agreement,
                parity_hf_device=args.parity_hf_device,
            ),
            flush=True,
        )
        return 0 if args.dry_run else execute(args, models, manifest, started, options)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
