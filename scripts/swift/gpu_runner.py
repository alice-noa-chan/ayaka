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

from ayaka.eval.read_artifact import fingerprint  # noqa: E402
from ayaka.swift.prompt import PROMPT_VARIANTS  # noqa: E402
from scripts.swift.inputs import (  # noqa: E402
    DEFAULT_MANIFEST,
    PACKED_MANIFEST,
    REPO,
    inventory,
    sha256,
)
from scripts.swift.latency_probe import write_json  # noqa: E402
from scripts.swift.parity import CENTERED_LOG_MASS_MAX_ABS_NATS, COHORT  # noqa: E402

VLLM_VERSION = "0.30.0"
GEMMA_12B_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
DEFAULT_KWARGS = {"enable_thinking": False}
# Planning estimates, not measured throughput. Loads are additional to priority
# work: each new model/adapter needs an HF reference and a fresh vLLM server.
PRIORITY_MINUTES = {"P0": 10, "P1": 35, "P2": 5, "P3": 20, "P4": 15}
ENVIRONMENT_PREP_MINUTES = 3
HF_LOAD_MINUTES = 2
VLLM_LOAD_MINUTES = 3
# Stop work 120s before the absolute cap to stop process groups and pack partials.
# This reserve is inside max_minutes; cleanup can never extend the cap.
CLEANUP_RESERVE_SECONDS = 120
FIXED_VARIANTS = ("min", "cygnet", "rules")
FIXED_MODELS = ("google/gemma-4-12B-it", "google/gemma-4-E4B-it")
DATASET_ORDER = ("v2_calibration", "v2_dev", "cygnet_calibration", "jevbench_public")


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
        "--tokenizer",
        "--tokenizer-revision",
        "--chat-template",
        "--generation-config",
        "--override-generation-config",
        "--enable-lora",
        "--lora-modules",
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
        "raw_logits",
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


def priority_plan(models, *, with_lora_arm=False):
    primary = models[0]["model"]
    secondary = models[1]["model"] if len(models) > 1 else "<E4B omitted>"
    return [
        {
            "id": "P0",
            "minutes": PRIORITY_MINUTES["P0"],
            "enabled": True,
            "work": f"{primary}: HF GPU bf16 reference -> process exit/free memory -> vLLM load -> exact-ID preflight/parity",
        },
        {
            "id": "P1",
            "minutes": PRIORITY_MINUTES["P1"],
            "enabled": True,
            "work": f"{primary} x min,cygnet,rules: calibration, dev, Cygnet; public last (diagnostic); select on dev only",
        },
        {
            "id": "P2",
            "minutes": PRIORITY_MINUTES["P2"],
            "enabled": True,
            "work": "200 serial HTTP latency reads for the best TWO P1 variants only",
        },
        {
            "id": "P3",
            "minutes": PRIORITY_MINUTES["P3"],
            "enabled": len(models) > 1,
            "work": f"{secondary} x min,cygnet,rules: reference/parity, then same collection",
        },
        {
            "id": "P4",
            "minutes": PRIORITY_MINUTES["P4"],
            "enabled": with_lora_arm,
            "work": "optional ayaka-large LoRA arm on the identical pinned 12B base/tokenizer (--with-lora-arm)",
        },
    ]


def admission_plan(models, minutes, *, with_lora_arm=False, allow_partial=False):
    """Admit only a prefix whose estimated setup, loads, work and cleanup fit."""
    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("max minutes must be finite and positive")
    priorities = priority_plan(models, with_lora_arm=with_lora_arm)
    overhead = ENVIRONMENT_PREP_MINUTES + CLEANUP_RESERVE_SECONDS / 60
    for priority in priorities:
        priority["load_minutes"] = (
            HF_LOAD_MINUTES + VLLM_LOAD_MINUTES if priority["id"] in ("P0", "P3", "P4") else 0
        )
        priority["total_minutes"] = priority["minutes"] + priority["load_minutes"]
    total = overhead + sum(p["total_minutes"] for p in priorities if p["enabled"])
    refused = total > minutes and not allow_partial
    admitted_total = overhead
    prefix_open = not refused
    for priority in priorities:
        admitted = (
            priority["enabled"]
            and prefix_open
            and admitted_total + priority["total_minutes"] <= minutes
        )
        priority.update(
            admitted=admitted,
            status="pending" if admitted else "skipped",
            skip_reason=None if admitted else "not_admitted" if priority["enabled"] else "disabled",
        )
        if admitted:
            admitted_total += priority["total_minutes"]
        elif priority["enabled"]:
            prefix_open = False
    return {
        "environment_prep_minutes": ENVIRONMENT_PREP_MINUTES,
        "cleanup_reserve_seconds": CLEANUP_RESERVE_SECONDS,
        "planned_minutes": total,
        "admitted_minutes": admitted_total if any(p["admitted"] for p in priorities) else 0,
        "refused": refused,
        "priorities": priorities,
    }


def require_admission(admission, minutes):
    if admission["refused"]:
        raise ValueError(
            f"planned total {admission['planned_minutes']:g} minutes exceeds "
            f"--max-minutes {minutes:g}; increase the cap or use --allow-partial "
            "to admit only the priorities that fit"
        )


def plan(
    models,
    manifest,
    concurrency,
    minutes,
    options,
    variants=FIXED_VARIANTS,
    *,
    with_lora_arm=False,
    allow_partial=False,
):
    admission = admission_plan(
        models, minutes, with_lora_arm=with_lora_arm, allow_partial=allow_partial
    )
    lines = [
        f"Prerequisites already installed: vllm=={VLLM_VERSION}; locally cached models/tokenizers (no automatic installs/downloads)",
        f"Absolute wall-clock cap: {minutes:g} minutes; stop unfinished priorities at their capped deadline",
        f"Cleanup/pack reserve: {CLEANUP_RESERVE_SECONDS}s inside the cap; every work deadline <= hard deadline minus reserve",
        "Time budget table / pre-launch admission (minutes; estimates, no GPU measurements):",
        "Priority  Budget  Enabled  Load  Total  Admission           Work",
    ]
    for priority in admission["priorities"]:
        decision = "admitted" if priority["admitted"] else "skipped: " + priority["skip_reason"]
        lines.append(
            f"{priority['id']:8s} {priority['minutes']:6g}  {str(priority['enabled']):7s}  "
            f"{priority['load_minutes']:4g}  {priority['total_minutes']:5g}  {decision:19s} {priority['work']}"
        )
    lines += [
        f"Environment prep: {ENVIRONMENT_PREP_MINUTES} minutes; model load estimate: HF {HF_LOAD_MINUTES} + vLLM {VLLM_LOAD_MINUTES} minutes per new model/adapter",
        f"Enabled plan total: {admission['planned_minutes']:g} minutes; admitted total: {admission['admitted_minutes']:g} minutes",
        "Admission: REFUSED (increase --max-minutes or use --allow-partial)"
        if admission["refused"]
        else "Admission: partial prefix"
        if any(p["enabled"] and not p["admitted"] for p in admission["priorities"])
        else "Admission: complete plan",
        f"Bulk concurrency: {concurrency}; Prompt variants: {','.join(variants)}",
        "Serve one model at a time: bf16, context=16384, GPU=0.90, prefix caching, --logprobs-mode raw_logits",
        f"Fixed parity cohort: {COHORT.name}; 2/20/26 options, skew, typed item and two permutations",
        f"Parity: identical prompt/canonical IDs, complete finite gather; P max-abs <=0.02; argmax >=0.98; centered log-mass <= {CENTERED_LOG_MASS_MAX_ABS_NATS:g} nats",
        "Record HF and vLLM load times; parity failure aborts bulk with comparison_valid=false diagnostic artifact",
    ]
    for model in models:
        lines.append(f"{model['model']}@{model['requested_revision']} -> out/{model['slug']}/")
        lines.append(f"  Options: {json.dumps(options[model['model']])}")
        for dataset in sorted(manifest["datasets"], key=lambda d: d["name"] == "jevbench_public"):
            lines.append(
                f"  {dataset['name']}: {dataset['items']} decisions, {dataset['reads']} model reads per variant"
            )
    decisions = sum(d["items"] for d in manifest["datasets"])
    reads = sum(d["reads"] for d in manifest["datasets"])
    lines.append(
        f"All models: {len(models) * len(variants) * decisions} bulk decisions, {len(models) * len(variants) * reads} bulk model reads + 400 serial HTTP requests"
    )
    lines.append(
        "Output: HF references, parity.json, load_times.json, variant reads, top-two latency, selection, env.txt, progress.json; out.tar.zst + SHA256 includes partials"
    )
    lines.append(f"Per-model/variant inventory: {decisions} decisions, {reads} model reads")
    return "\n".join(lines)


def resolve_model(model: str, requested: str, output: Path, kwargs: dict) -> None:
    """Run only on the GPU host, in a supervised child with a bounded lifetime."""
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    revision = HfApi().model_info(model, revision=requested).sha
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Hub did not resolve the requested revision to a commit SHA")
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)
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
            self.remaining()
            if code:
                raise RuntimeError(f"command exited {code}; see {log}")
        except subprocess.TimeoutExpired as exc:
            raise DeadlineReached("command reached the work deadline") from exc
        finally:
            self.stop([process])

    def health(self, url: str, process: subprocess.Popen, timeout: float) -> None:
        import urllib.error
        import urllib.request

        self.remaining()
        end = min(time.monotonic() + timeout, self.work_deadline)
        while time.monotonic() < end:
            if process.poll() is not None:
                raise RuntimeError(f"server exited {process.returncode} before health: {url}")
            try:
                with urllib.request.urlopen(url, timeout=min(2, self.remaining())) as response:
                    if response.status == 200:
                        self.remaining()
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


def select_model_results(directory: Path) -> dict:
    """Run CPU fitting in a supervised child while the billable backend is live."""
    from ayaka.swift.collect import load_reads
    from ayaka.swift.policy import Policy
    from scripts.swift.select_variant import assert_roles_isolated, select_variants

    calibration, dev = [], []
    for variant in FIXED_VARIANTS:
        for target, name in ((calibration, "v2_calibration"), (dev, "v2_dev")):
            target.extend(load_reads([directory / variant / (name + ".reads.jsonl")]))
    # Check all roles before excluding grouped diagnostics. Cygnet keeps its
    # original split and never enters fitting or dev selection.
    assert_roles_isolated(calibration, dev)
    calibration = [r for r in calibration if r["readout"] != "grouped_approx"]
    dev = [r for r in dev if r["readout"] != "grouped_approx"]
    selection = select_variants(calibration, dev)
    write_json(directory / "variant_selection.json", selection)
    for variant, policy in selection["policies"].items():
        Policy(**policy).save(directory / variant / "policy.json")
    return selection


def execute(args, models, manifest, started, options):
    admission = admission_plan(
        models,
        args.max_minutes,
        with_lora_arm=getattr(args, "with_lora_arm", False),
        allow_partial=getattr(args, "allow_partial", False),
    )
    require_admission(admission, args.max_minutes)
    if sys.platform != "linux":
        raise ValueError("live collection requires Linux; --dry-run works on CPU/Windows")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be empty; use --output for a fresh run")
    args.output.mkdir(parents=True, exist_ok=True)
    hard_deadline = started + args.max_minutes * 60
    work_limit = hard_deadline - CLEANUP_RESERVE_SECONDS
    supervisor = Supervisor(
        REPO, hard_deadline, min(started + ENVIRONMENT_PREP_MINUTES * 60, work_limit)
    )
    state = {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "max_minutes": args.max_minutes,
        "hard_deadline_elapsed_s": hard_deadline - started,
        "admission": {k: v for k, v in admission.items() if k != "priorities"},
        "status": "running",
        "current_step": "preflight_environment",
        "finished_steps": [],
        "models": {},
        "prompt_variants": list(FIXED_VARIANTS),
        "comparison_valid": False,
        "priorities": admission["priorities"],
    }
    marker = args.output / "progress.json"
    gpu = "unavailable (preflight/startup interrupted)"
    backend_url = f"http://127.0.0.1:{args.vllm_port}"
    active_model = None

    def checkpoint(step=None):
        if step:
            state["finished_steps"].append(step)
        write_json(marker, state)

    def begin(step):
        supervisor.remaining()
        state["current_step"] = step
        checkpoint()

    def run(command, log, step):
        begin(step)
        supervisor.run(command, log)
        supervisor.remaining()
        checkpoint(step)

    def reader_flags(model):
        return [
            "--backend",
            "vllm",
            "--vllm-url",
            backend_url,
            "--model",
            model.get("served_model", model["model"]),
            "--revision",
            model.get("adapter_revision", model["revision"]),
            "--tokenizer-model",
            model["model"],
            "--tokenizer-revision",
            model["revision"],
            "--chat-template-kwargs",
            json.dumps(options[model["model"]]["chat_template_kwargs"]),
            *(["--adapter-sha256", model["adapter_sha256"]] if model.get("adapter_sha256") else []),
        ]

    def start_model(model):
        directory = args.output / model["slug"]
        directory.mkdir(exist_ok=True)
        path = directory / "parity.json"
        write_json(
            path,
            {
                "model": model["model"],
                "complete": False,
                "passed": False,
                "comparison_valid": False,
                "hf_load_s": None,
                "vllm_load_s": None,
                "sequence": "hf_reference_exit_then_vllm",
            },
        )
        try:
            prepare_model(model)
        except BaseException as exc:
            diagnostic = json.loads(path.read_text())
            reference_path = directory / "hf_reference.json"
            if reference_path.exists():
                reference = json.loads(reference_path.read_text())
                diagnostic["hf_load_s"] = reference.get("hf_load_s")
            diagnostic.update(
                passed=False,
                comparison_valid=False,
                runner_error=f"{type(exc).__name__}: {exc}",
            )
            write_json(path, diagnostic)
            raise

    def prepare_model(model):
        nonlocal active_model
        supervisor.stop()
        active_model = None
        directory = args.output / model["slug"]
        directory.mkdir(exist_ok=True)
        opts = options[model["model"]]
        if "revision" not in model:
            run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "_resolve",
                    model["model"],
                    model["requested_revision"],
                    str(directory / "revision.json"),
                    json.dumps(opts["chat_template_kwargs"]),
                ],
                directory / "resolve.log",
                model["slug"] + "/resolve",
            )
            model["revision"] = json.loads((directory / "revision.json").read_text())["revision"]
        state["models"][model["slug"]] = model.copy()
        (directory / "env.txt").write_text(environment(model, opts, gpu), encoding="utf-8")
        common = [
            sys.executable,
            str(REPO / "scripts/swift/parity.py"),
            str(COHORT),
            "--model",
            model["model"],
            "--revision",
            model["revision"],
            "--chat-template-kwargs",
            json.dumps(opts["chat_template_kwargs"]),
            "--prompt-variants",
            *FIXED_VARIANTS,
        ]
        if model.get("lora_path"):
            common += ["--lora-path", model["lora_path"], "--served-model", model["served_model"]]
        # Blocking subprocess exit releases the HF model, allocator and CUDA
        # context before vLLM is started; there is no concurrent GPU allocation.
        run(
            [
                *common,
                "--phase",
                "reference",
                "--hf-device",
                "cuda",
                "--dtype",
                "bfloat16",
                "--output",
                str(directory / "hf_reference.json"),
            ],
            directory / "hf_reference.log",
            model["slug"] + "/hf_reference",
        )
        reference = json.loads((directory / "hf_reference.json").read_text())
        timings = {
            "hf_load_s": reference["hf_load_s"],
            "vllm_load_s": None,
            "sequence": "hf_reference_exit_then_vllm",
            "dtype": "bfloat16",
            "device": "cuda",
        }
        write_json(directory / "load_times.json", timings)
        begin(model["slug"] + "/vllm_health")
        loaded = time.monotonic()
        command = serve_command(model, args.vllm_port, opts)
        if model.get("lora_path"):
            command += [
                "--enable-lora",
                "--lora-modules",
                model["served_model"] + "=" + model["lora_path"],
            ]
        try:
            backend = supervisor.start(command, directory / "vllm.log")
            supervisor.health(backend_url + "/health", backend, args.health_timeout)
            timings["vllm_healthy"] = True
        finally:
            vllm_load_s = time.monotonic() - loaded
            timings["vllm_load_s"] = vllm_load_s
            write_json(directory / "load_times.json", timings)
            path = directory / "parity.json"
            diagnostic = json.loads(path.read_text())
            diagnostic.update(hf_load_s=reference["hf_load_s"], vllm_load_s=vllm_load_s)
            write_json(path, diagnostic)
        checkpoint(state["current_step"])
        path = directory / "parity.json"
        try:
            run(
                [
                    *common,
                    "--phase",
                    "compare",
                    "--reference",
                    str(directory / "hf_reference.json"),
                    "--vllm-url",
                    backend_url,
                    "--output",
                    str(path),
                ],
                directory / "parity.log",
                model["slug"] + "/parity",
            )
        finally:
            diagnostic = (
                json.loads(path.read_text())
                if path.exists()
                else {"complete": False, "passed": False, "comparison_valid": False}
            )
            diagnostic.update(hf_load_s=reference["hf_load_s"], vllm_load_s=vllm_load_s)
            diagnostic["comparison_valid"] = (
                diagnostic.get("complete") is True
                and diagnostic.get("passed") is True
                and diagnostic.get("comparison_valid") is True
            )
            write_json(path, diagnostic)
        if not diagnostic["comparison_valid"]:
            raise RuntimeError(
                "exact token-ID HF/vLLM preflight/parity failed; bulk collection aborted"
            )
        state["comparison_valid"] = True
        active_model = model

    def select_model(model):
        directory = args.output / model["slug"]
        run(
            [sys.executable, str(Path(__file__).resolve()), "_select", str(directory)],
            directory / "select.log",
            model["slug"] + "/select",
        )
        selection = json.loads((directory / "variant_selection.json").read_text())
        state["best_two_variants"] = sorted(
            selection["variants"], key=lambda v: (-selection["variants"][v]["composite_A"], v)
        )[:2]
        checkpoint(state["current_step"])

    def collect_model(model, *, select=False):
        if active_model is not model:
            start_model(model)
        directory = args.output / model["slug"]
        # Finish every eligible non-public read before opening the public arm.
        datasets = sorted(manifest["datasets"], key=lambda d: DATASET_ORDER.index(d["name"]))
        for dataset in datasets:
            if select and dataset["name"] == "jevbench_public":
                select_model(model)
            for variant in FIXED_VARIANTS:
                variant_directory = directory / variant
                variant_directory.mkdir(exist_ok=True)
                run(
                    [
                        sys.executable,
                        "-m",
                        "ayaka.swift.collect",
                        *dataset["paths"],
                        "--output",
                        str(variant_directory / (dataset["name"] + ".reads.jsonl")),
                        "--prompt-variant",
                        variant,
                        *reader_flags(model),
                        "--concurrency",
                        str(args.concurrency),
                        "--group-size",
                        str(manifest["group_size"]),
                        *(
                            ["--diagnostic"]
                            if dataset["name"] in ("jevbench_public", "cygnet_calibration")
                            else []
                        ),
                    ],
                    variant_directory / "collect.log",
                    model["slug"] + "/" + variant + "/" + dataset["name"],
                )
                if dataset["name"] in ("jevbench_public", "cygnet_calibration"):
                    write_json(
                        variant_directory / (dataset["name"] + ".diagnostic.json"),
                        {
                            "role": "diagnostic",
                            "used_for_fit_or_selection": False,
                            "public": dataset["name"] == "jevbench_public",
                        },
                    )

    def probe_best_two(model):
        if active_model is not model or len(state.get("best_two_variants", [])) != 2:
            raise RuntimeError("P2 requires P1's predeclared best-two selection")
        public = next(d for d in manifest["datasets"] if d["name"] == "jevbench_public")
        for variant in state["best_two_variants"]:
            directory = args.output / model["slug"] / variant
            begin(model["slug"] + "/" + variant + "/swift_health")
            swift = supervisor.start(
                [
                    sys.executable,
                    "-m",
                    "ayaka.swift.server",
                    "--prompt-variant",
                    variant,
                    *reader_flags(model),
                    "--policy",
                    str(directory / "policy.json"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.swift_port),
                    "--max-parallel",
                    "1",
                    "--group-size",
                    str(manifest["group_size"]),
                ],
                directory / "swift.log",
            )
            try:
                swift_url = f"http://127.0.0.1:{args.swift_port}"
                supervisor.health(swift_url + "/health", swift, args.health_timeout)
                checkpoint(state["current_step"])
                run(
                    [
                        sys.executable,
                        str(REPO / "scripts/swift/latency_probe.py"),
                        *public["paths"],
                        "--url",
                        swift_url,
                        "--output",
                        str(directory / "latency.json"),
                        "--model",
                        model["model"],
                        "--revision",
                        model["revision"],
                        "--prompt-variant",
                        variant,
                        "--reads",
                        "200",
                    ],
                    directory / "latency.log",
                    model["slug"] + "/" + variant + "/latency",
                )
            finally:
                supervisor.stop([swift])

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    code = 0
    try:
        checkpoint()
        if any(p["admitted"] for p in state["priorities"]):
            begin("preflight_environment")
            require_free_ports((args.vllm_port, args.swift_port))
            # Preconditions are read-only. The runner never installs dependencies.
            if importlib.metadata.version("vllm") != VLLM_VERSION:
                raise RuntimeError("preflight requires installed vLLM 0.30.0")
            run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                args.output / "gpu.txt",
                "preflight_environment",
            )
            gpu = (args.output / "gpu.txt").read_text()
        for priority in state["priorities"]:
            if not priority["admitted"]:
                continue
            # A completed priority must also finish stopping its transient
            # processes before its deadline. No fresh work begins in reserve.
            if time.monotonic() >= work_limit:
                raise DeadlineReached("global work deadline reached before next priority")
            priority["status"] = "running"
            priority["started_elapsed_s"] = time.monotonic() - started
            supervisor.work_deadline = min(
                time.monotonic() + priority["total_minutes"] * 60, work_limit
            )
            priority["work_deadline_elapsed_s"] = supervisor.work_deadline - started
            checkpoint()
            if priority["id"] == "P0":
                start_model(models[0])
            elif priority["id"] == "P1":
                collect_model(models[0], select=True)
            elif priority["id"] == "P2":
                probe_best_two(models[0])
            elif priority["id"] == "P3":
                collect_model(models[1])
            else:
                # Release the previous backend before local adapter hashing,
                # which can take time even without launching another model.
                supervisor.stop()
                active_model = None
                arm = {
                    **models[0],
                    "slug": "ayaka_large_lora",
                    "lora_path": str(args.lora_path),
                    "served_model": "ayaka-large-swift",
                    "adapter_revision": args.lora_revision,
                }
                config = json.loads(
                    (args.lora_path / "adapter_config.json").read_text(encoding="utf-8")
                )
                if config.get("base_model_name_or_path") != arm["model"] or config.get(
                    "revision"
                ) not in (None, arm["revision"]):
                    raise ValueError(
                        "LoRA adapter base/revision differs from the pinned comparison base"
                    )
                arm["adapter_sha256"] = fingerprint(
                    {
                        p.relative_to(args.lora_path).as_posix(): sha256(p)
                        for p in sorted(args.lora_path.rglob("*"))
                        if p.is_file()
                    }
                )
                collect_model(arm)
            supervisor.remaining()
            priority.update(status="complete", finished_elapsed_s=time.monotonic() - started)
            checkpoint(priority["id"])
        if state["status"] == "running":
            state["status"] = (
                "partial"
                if any(p["enabled"] and not p["admitted"] for p in state["priorities"])
                else "complete"
            )
    except DeadlineReached as exc:
        state.update(
            status="partial", stop_reason="priority_timeout", error=str(exc), comparison_valid=False
        )
        code = 124
    except KeyboardInterrupt as exc:
        state.update(status="interrupted", error=str(exc), comparison_valid=False)
        code = 130
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", comparison_valid=False)
        code = 1
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            # Reuse the original absolute cap, including if work or SIGTERM
            # arrives late. Neither shutdown nor packing gets a fresh budget.
            supervisor.stop()
            for priority in state["priorities"]:
                if priority["status"] == "pending":
                    priority.update(
                        status="skipped", skip_reason=state.get("stop_reason", state["status"])
                    )
                elif priority["status"] == "running":
                    priority["status"] = (
                        "timeout" if state.get("stop_reason") == "priority_timeout" else "failed"
                    )
            state.update(
                elapsed_s=time.monotonic() - started,
                stopped_at_step=state["current_step"],
                current_step="finished",
            )
            checkpoint()
            write_json(args.output / "inputs.manifest.json", manifest)
            supervisor.work_deadline = hard_deadline
            try:
                supervisor.remaining()
                supervisor.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "_pack",
                        "--deadline",
                        str(hard_deadline),
                        str(args.output),
                        str(args.archive),
                    ],
                    args.archive.with_name(args.archive.name + ".pack.log"),
                )
                supervisor.remaining()
                state["archive_status"] = "complete"
            except Exception as exc:
                timed_out = isinstance(exc, DeadlineReached)
                state.update(
                    archive_status="timeout" if timed_out else "failed",
                    archive_error=f"{type(exc).__name__}: {exc}",
                )
                if state["status"] in ("complete", "partial"):
                    state["status"] = "partial" if timed_out else "failed"
                if not code:
                    code = 124 if timed_out else 1
            state["elapsed_s"] = time.monotonic() - started
            checkpoint()
        finally:
            signal.signal(signal.SIGTERM, previous)
            signal.signal(signal.SIGINT, old_int)
        print(f"{state['status']}: archive {state['archive_status']} at {args.archive}", flush=True)
    return code


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_resolve":
        resolve_model(argv[1], argv[2], Path(argv[3]), json.loads(argv[4]))
        return 0
    if argv and argv[0] == "_pack":
        if argv[1] == "--deadline":
            pack_results(Path(argv[3]), Path(argv[4]), deadline=float(argv[2]))
        else:
            pack_results(Path(argv[1]), Path(argv[2]))
        return 0
    if argv and argv[0] == "_select":
        select_model_results(Path(argv[1]))
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
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="admit only the priority prefix that fits the cap",
    )
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--swift-port", type=int, default=8009)
    parser.add_argument("--health-timeout", type=float, default=600)
    parser.add_argument("--with-lora-arm", action="store_true")
    parser.add_argument(
        "--lora-path", type=Path, help="already cached ayaka-large adapter directory"
    )
    parser.add_argument("--lora-revision", help="immutable adapter receipt SHA")
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
        if tuple(args.prompt_variants) != FIXED_VARIANTS:
            raise ValueError("the minimal plan fixes prompt variants to min,cygnet,rules")
        if (
            args.with_lora_arm
            and not args.dry_run
            and (
                not args.lora_path
                or not args.lora_path.is_dir()
                or not args.lora_revision
                or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", args.lora_revision)
            )
        ):
            raise ValueError(
                "LoRA arm needs a locally cached --lora-path and pinned --lora-revision"
            )
        if not all(1 <= port <= 65535 for port in (args.swift_port, args.vllm_port)):
            raise ValueError("ports must be in 1..65535")
        if args.archive.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("archive must be outside the output directory")
        args.output = args.output.resolve()
        args.archive = args.archive.resolve()
        if not args.dry_run:
            args.archive.parent.mkdir(parents=True, exist_ok=True)
        models = model_specs(args.model, dry_run=args.dry_run)
        if len(models) != 2:
            raise ValueError("fixed plan requires primary 12B and secondary E4B model slots")
        if tuple(model["model"] for model in models) != FIXED_MODELS:
            raise ValueError("fixed plan model slots are gemma-4-12B-it then gemma-4-E4B-it")
        options = model_options(args.model_options, models)
        manifest, _ = inventory(REPO, args.manifest)
        if {dataset["name"] for dataset in manifest["datasets"]} != set(DATASET_ORDER):
            raise ValueError("fixed plan requires calibration, dev, Cygnet and public datasets")
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
                with_lora_arm=args.with_lora_arm,
                allow_partial=args.allow_partial,
            ),
            flush=True,
        )
        return 0 if args.dry_run else execute(args, models, manifest, started, options)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
