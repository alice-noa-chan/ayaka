"""Runtime preparation rejects incompatible pins before any paid model load."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.direct_v2 import runtime

LOCK = Path(__file__).resolve().parents[1] / "scripts/direct_v2/requirements.lock.txt"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_actual_generated_lock_keeps_receipt_versions_and_real_kernel_packages():
    _, versions = runtime.read_lock(LOCK, digest(LOCK))
    assert versions.items() >= runtime.CORE.items()
    assert versions["triton"] == "3.4.0"
    assert "flash-linear-attention" not in versions


@pytest.mark.parametrize(
    "change", ["unpinned", "unhashed", "duplicate", "wrong_abi", "receipt_version"]
)
def test_incompatible_or_unpinned_lock_fails(tmp_path, change):
    text = LOCK.read_text(encoding="utf-8")
    if change == "unpinned":
        text += "\nninja\n"
    elif change == "unhashed":
        text += "\nninja==1.11.1\n"
    elif change == "duplicate":
        text += "\npeft==0.21.2\n --hash=sha256:" + "a" * 64 + "\n"
    elif change == "wrong_abi":
        text = text.replace("cxx11abiTRUE", "cxx11abiFALSE")
    else:
        text = text.replace("peft==0.21.2", "peft==0.21.0")
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        runtime.read_lock(lock, digest(lock))


def test_untrusted_lock_fails_before_dependency_import(monkeypatch):
    monkeypatch.setattr(
        runtime.importlib, "import_module", lambda _: pytest.fail("import too early")
    )
    with pytest.raises(ValueError, match="external anchor"):
        runtime.inspect_runtime(LOCK, "0" * 64)


@pytest.fixture
def local_runtime(monkeypatch, tmp_path):
    _, versions = runtime.read_lock(LOCK, digest(LOCK))
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.11.14")
    monkeypatch.setattr(runtime.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(runtime.sys, "prefix", str(tmp_path))
    binary = tmp_path / "extension.so"
    binary.write_bytes(b"fixture, not a CUDA extension")

    def entry(**_kwargs):
        pytest.fail("CPU runtime smoke must not execute a CUDA kernel")

    modules = {
        name: SimpleNamespace(__file__=str(binary))
        for name in (
            "torch",
            "transformers",
            "peft",
            "flash_attn_2_cuda",
            "flash_attn.flash_attn_interface",
            "flash_attn.bert_padding",
            "transformers.modeling_flash_attention_utils",
            "liger_kernel.transformers.functional",
        )
    }
    torch = modules["torch"]
    torch.__version__ = "2.8.0+cu128"
    torch.version = SimpleNamespace(cuda="12.8")
    torch.cuda = SimpleNamespace(is_initialized=lambda: False)
    torch._C = SimpleNamespace(
        _GLIBCXX_USE_CXX11_ABI=True, _cuda_getArchFlags=lambda: "sm_80 sm_90 sm_120"
    )
    flash = modules["flash_attn.flash_attn_interface"]
    flash.flash_attn_func = entry
    flash.flash_attn_varlen_func = entry
    padding = modules["flash_attn.bert_padding"]
    padding.pad_input = entry
    padding.unpad_input = entry
    modules["transformers.modeling_flash_attention_utils"]._flash_attention_forward = entry
    liger = modules["liger_kernel.transformers.functional"]
    liger.liger_rms_norm = entry
    liger.liger_geglu = entry
    monkeypatch.setattr(runtime.importlib, "import_module", modules.__getitem__)
    return modules, versions


def test_cpu_import_receipt_cannot_claim_training_or_cuda_compatibility(local_runtime):
    report = runtime.inspect_runtime(LOCK, digest(LOCK))
    assert report["dependencies"] == local_runtime[1]
    assert report["optimizer_steps"] == 0
    for name in (
        "production_ready",
        "gpu_allocated",
        "cuda_initialized",
        "native_model_loaded",
        "cuda_kernel_execution_verified",
        "target_gpu_compatibility_verified",
        "target_host_compatibility_verified",
    ):
        assert report[name] is False


@pytest.mark.parametrize(
    "bad",
    [
        "version",
        "cuda",
        "abi",
        "initialized",
        "empty_arch",
        "missing_padding",
        "wrong_liger",
        "escaped_import",
    ],
)
def test_actual_runtime_rejects_binary_and_api_mismatches(local_runtime, bad, tmp_path):
    modules, _ = local_runtime
    torch = modules["torch"]
    if bad == "version":
        torch.__version__ = "2.8.0+cpu"
    elif bad == "cuda":
        torch.version.cuda = "12.6"
    elif bad == "abi":
        torch._C._GLIBCXX_USE_CXX11_ABI = False
    elif bad == "initialized":
        torch.cuda.is_initialized = lambda: True
    elif bad == "empty_arch":
        torch._C._cuda_getArchFlags = lambda: ""
    elif bad == "missing_padding":
        modules["flash_attn.bert_padding"].unpad_input = None
    elif bad == "wrong_liger":
        modules["liger_kernel.transformers.functional"].liger_rms_norm = lambda X, W: None
    else:
        modules["flash_attn_2_cuda"].__file__ = str(tmp_path.parent / "foreign.so")
    with pytest.raises(ValueError):
        runtime.inspect_runtime(LOCK, digest(LOCK))


def test_locked_transitive_mismatch_is_rejected_before_import(local_runtime, monkeypatch):
    versions = dict(local_runtime[1])
    versions["triton"] = "3.8.0"
    monkeypatch.setattr(runtime.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(
        runtime.importlib, "import_module", lambda _: pytest.fail("import too early")
    )
    with pytest.raises(ValueError, match="dependencies differ"):
        runtime.inspect_runtime(LOCK, digest(LOCK))


def test_newer_python_fails_before_import(local_runtime, monkeypatch):
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.14.4")
    monkeypatch.setattr(
        runtime.importlib, "import_module", lambda _: pytest.fail("import too early")
    )
    with pytest.raises(ValueError, match="CPython 3.11.14"):
        runtime.inspect_runtime(LOCK, digest(LOCK))


@pytest.mark.parametrize("arguments", [[], ["relative", "0" * 64], ["/unused", "invalid"]])
def test_preparer_rejects_bad_arguments_before_install(arguments):
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("a real Bash executable is unavailable")
    script = LOCK.with_name("prepare_runtime.sh")
    result = subprocess.run(
        [bash, str(script), *arguments], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 2


def test_import_side_effect_cannot_initialize_cuda(local_runtime, monkeypatch):
    modules, _ = local_runtime
    initialized = False

    def import_module(name):
        nonlocal initialized
        if name == "liger_kernel.transformers.functional":
            initialized = True
        return modules[name]

    modules["torch"].cuda.is_initialized = lambda: initialized
    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    with pytest.raises(ValueError, match="unexpectedly initialized CUDA"):
        runtime.inspect_runtime(LOCK, digest(LOCK))
