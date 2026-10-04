"""Inspect a hash-pinned Linux runtime on CPU, before renting a GPU.

Importing CUDA extensions establishes packaging/ABI compatibility only. Actual
kernel arithmetic, memory, speed and model quality require separate probes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import platform
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

VERSION = "ayaka-direct-linux-runtime-1"
PYTHON = "3.11.14"
CORE = {
    "torch": "2.8.0+cu128",
    "flash-attn": "2.8.3.post1",
    "liger-kernel": "0.8.4",
    "transformers": "5.18.0",
    "tokenizers": "0.23.2",
    "peft": "0.21.2",
}
WHEELS = {
    "torch": "https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl",
    "flash-attn": "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
}


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_lock(path, expected_sha256):
    """Parse the generated binary-only lock; reject unpinned installer directives."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("runtime lock requires a separately trusted SHA256")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("runtime lock differs from its external anchor")
    versions, hashes, current = {}, {}, None
    for line in raw.decode("utf-8").splitlines():
        line = line.strip().removesuffix("\\").strip()
        if not line or line.startswith("#"):
            continue
        hashed = re.fullmatch(r"--hash=sha256:([0-9a-f]{64})", line)
        if hashed:
            if current is None:
                raise ValueError("runtime lock hash has no pinned distribution")
            hashes[current].add(hashed[1])
            continue
        pinned = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        direct = re.fullmatch(r"([A-Za-z0-9_.-]+) @ (https://\S+)", line)
        if pinned:
            current, version = normalize(pinned[1]), pinned[2]
        elif direct:
            current = normalize(direct[1])
            if WHEELS.get(current) != direct[2]:
                raise ValueError("runtime requires the exact official CUDA/ABI wheel")
            # Upstream FA2 encodes build compatibility in the filename, while
            # its distribution METADATA records the base release version.
            filename = unquote(urlparse(direct[2]).path.rsplit("/", 1)[-1])
            version = filename.split("-")[1]
            if current == "flash-attn":
                version = version.split("+")[0]
        else:
            raise ValueError("runtime lock contains an unpinned or unsupported requirement")
        if current in versions:
            raise ValueError("runtime lock repeats a distribution")
        versions[current], hashes[current] = version, set()
    if not versions or any(not values for values in hashes.values()):
        raise ValueError("every runtime distribution requires wheel hashes")
    if any(versions.get(name) != version for name, version in CORE.items()):
        raise ValueError("runtime core pins do not match the audited direct recipe")
    return raw, versions


def _accepts(function, arguments):
    if not callable(function):
        raise ValueError("runtime kernel entry point is not callable")
    try:
        inspect.signature(function).bind(**arguments)
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime kernel API differs from the actual adapter") from exc


def _local_file(module):
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("runtime import escaped the prepared Python prefix")
    return path


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_runtime(lock, expected_sha256):
    """Import actual extensions without querying devices or allocating GPU tensors."""
    raw, pinned = read_lock(lock, expected_sha256)
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or platform.python_implementation() != "CPython"
        or platform.python_version() != PYTHON
    ):
        raise ValueError("runtime requires Linux x86_64 / CPython 3.11.14")
    installed = {name: importlib.metadata.version(name) for name in pinned}
    if installed != pinned:
        changed = sorted(name for name in pinned if pinned[name] != installed[name])
        raise ValueError(f"installed dependencies differ from the anchored lock: {changed}")
    torch = importlib.import_module("torch")
    _local_file(torch)
    if torch.cuda.is_initialized():
        raise ValueError("runtime smoke must not initialize CUDA")
    if torch.__version__ != CORE["torch"] or torch.version.cuda != "12.8":
        raise ValueError("runtime requires the pinned PyTorch CUDA 12.8 build")
    if torch._C._GLIBCXX_USE_CXX11_ABI is not True:
        raise ValueError("runtime FlashAttention wheel requires CXX11 ABI")
    arch = torch._C._cuda_getArchFlags()
    if not isinstance(arch, str) or not arch:
        raise ValueError("runtime PyTorch wheel declares no compiled CUDA architectures")
    for name in ("transformers", "peft"):
        _local_file(importlib.import_module(name))
    extension = importlib.import_module("flash_attn_2_cuda")
    flash = importlib.import_module("flash_attn.flash_attn_interface")
    padding = importlib.import_module("flash_attn.bert_padding")
    hf_flash = importlib.import_module("transformers.modeling_flash_attention_utils")
    liger = importlib.import_module("liger_kernel.transformers.functional")
    for module in (extension, flash, padding, hf_flash, liger):
        _local_file(module)
    for name in ("flash_attn_func", "flash_attn_varlen_func"):
        if not callable(getattr(flash, name, None)):
            raise ValueError("runtime FA2 extension entry point is unavailable")
    for name in ("pad_input", "unpad_input"):
        if not callable(getattr(padding, name, None)):
            raise ValueError("runtime FA2 padding entry point is unavailable")
    _accepts(
        hf_flash._flash_attention_forward,
        {
            "query_states": None,
            "key_states": None,
            "value_states": None,
            "attention_mask": None,
            "query_length": 1,
            "is_causal": True,
            "dropout": 0.0,
            "softmax_scale": 1.0,
            "sliding_window": None,
            "use_top_left_mask": False,
            "target_dtype": None,
            "attn_implementation": "flash_attention_2",
        },
    )
    _accepts(
        liger.liger_rms_norm,
        {
            "X": None,
            "W": None,
            "eps": 1e-6,
            "offset": 0.0,
            "casting_mode": "gemma",
            "in_place": False,
        },
    )
    _accepts(liger.liger_geglu, {"a": None, "b": None})
    if torch.cuda.is_initialized():
        raise ValueError("runtime imports unexpectedly initialized CUDA")
    if Path(lock).read_bytes() != raw:
        raise ValueError("runtime lock changed during inspection")
    return {
        "version": VERSION,
        "status": "linux_cpu_imports_verified",
        "python": platform.python_version(),
        "platform": "linux_x86_64",
        "lock_sha256": expected_sha256,
        "dependencies": installed,
        "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": True,
        "compiled_architectures": arch.split(),
        "libc": list(platform.libc_ver()),
        "flash_extension_sha256": _file_digest(_local_file(extension)),
        "target_gpu_compatibility_verified": False,
        "target_host_compatibility_verified": False,
        "cuda_initialized": False,
        "cuda_kernel_execution_verified": False,
        "native_model_loaded": False,
        "optimizer_steps": 0,
        "gpu_allocated": False,
        "production_ready": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("runtime smoke receipt must be a fresh file")
    result = inspect_runtime(args.lock, args.expected_lock_sha256)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
