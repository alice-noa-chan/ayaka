"""Stage audited direct-student inputs as a verified zstd-3 upload payload.

No cloud allocation, training or dependency installation. Linux/CUDA runtime
and measured full-workflow credit admission remain separate prerequisites.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import zstandard

VERSION = "ayaka-direct-upload-1"
PREFIX = "ayaka-direct/"
MANIFEST = PREFIX + "upload-manifest.json"
BLOCK = 8 * 1024 * 1024


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _relative(name):
    path = PurePosixPath(name)
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or ":" in name
        or path.is_absolute()
        or any(part in {".", ".."} for part in name.split("/"))
        or str(path) != name
    ):
        raise ValueError("upload member must be a safe relative path")
    return name


def _pinned_receipt(path, expected_sha256):
    raw = Path(path).read_bytes()
    if not _sha256_value(expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("CPU audit receipt differs from its external SHA256 before source import")
    receipt = json.loads(raw)
    hashes = receipt.get("source_sha256")
    if (
        receipt.get("version") != "ayaka-direct-cpu-audit-1"
        or not isinstance(hashes, dict)
        or not hashes
    ):
        raise ValueError("CPU audit requires an exact source inventory before import")
    if any(
        not _relative(name).endswith(".py") or not _sha256_value(sha)
        for name, sha in hashes.items()
    ):
        raise ValueError("CPU source inventory requires Python files and SHA256 digests")
    return receipt


def _sha256_value(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _verified_tiny_scope(bundle, expected_sha256, receipt, requested_tiny):
    root = Path(bundle)
    raw = (root / "manifest.json").read_bytes()
    if (
        not _sha256_value(expected_sha256)
        or hashlib.sha256(raw).hexdigest() != expected_sha256
        or receipt.get("bundle_manifest_sha256") != expected_sha256
    ):
        raise ValueError("bundle manifest differs from its external CPU audit anchor before import")
    manifest = json.loads(raw)
    raw_recipe = (root / "recipe.json").read_bytes()
    if (
        manifest.get("files", {}).get("recipe.json") != hashlib.sha256(raw_recipe).hexdigest()
        or manifest.get("source_sha256") != receipt["source_sha256"]
    ):
        raise ValueError("bundle recipe or source differs from its anchored manifest before import")
    recipe = json.loads(raw_recipe)
    tiny = recipe.get("model", {}).get("backbone") == "tiny" and recipe.get("allow_tiny") is True
    if requested_tiny is not tiny:
        raise ValueError("mechanics-only packaging requires an actual anchored tiny recipe")
    return tiny


@contextmanager
def _isolated_bytecode():
    # Existing timestamp-valid/unchecked .pyc can execute different code even
    # when every .py digest matches. An empty prefix forces source compilation.
    previous = sys.pycache_prefix
    with tempfile.TemporaryDirectory(prefix="ayaka-upload-bytecode-") as directory:
        try:
            sys.pycache_prefix = directory
            yield
        finally:
            sys.pycache_prefix = previous


def _source_imports(source_root, source_hashes, *, allow_preloaded=False):
    """Prevent an editable checkout from silently validating a frozen source."""
    source_root = Path(source_root).resolve()
    package_root = source_root / "ayaka"
    if not (package_root / "training/direct_audit.py").is_file():
        raise ValueError("source root must contain the audited Ayaka training source")
    paths = sorted(package_root.rglob("*.py"))
    if (
        any(path.is_symlink() or not path.is_file() for path in paths)
        or {path.relative_to(package_root).as_posix(): _digest(path) for path in paths}
        != source_hashes
    ):
        raise ValueError("frozen source bytes differ from the external CPU receipt before import")
    for name, module in tuple(sys.modules.items()):
        if (name == "ayaka" or name.startswith("ayaka.")) and (
            not allow_preloaded
            or not getattr(module, "__file__", None)
            or not Path(module.__file__).resolve().is_relative_to(package_root)
        ):
            raise ValueError(
                "production packaging requires a fresh process without preloaded Ayaka modules"
            )
    sys.path.insert(0, str(source_root))
    return source_root


class CheckedReader:
    def __init__(self, stream):
        self.stream, self.hasher, self.size = stream, hashlib.sha256(), 0

    def read(self, size):
        block = self.stream.read(size)
        self.hasher.update(block)
        self.size += len(block)
        return block


@_isolated_bytecode()
def build_archive(
    source_root,
    bundle,
    audit_receipt,
    snapshot_record,
    snapshot_path,
    out,
    *,
    expected_bundle_sha256,
    expected_audit_receipt_sha256,
    expected_snapshot_record_sha256,
    allow_tiny=False,
):
    """Stream only pinned training files; original holdout/raw/cache are excluded.

    The frozen source must be the source actually imported for audit. Do not
    infer production readiness from successful packaging or its layout report.
    """
    if type(allow_tiny) is not bool:
        raise ValueError("allow_tiny must be an explicit boolean")
    receipt = _pinned_receipt(audit_receipt, expected_audit_receipt_sha256)
    tiny = _verified_tiny_scope(bundle, expected_bundle_sha256, receipt, allow_tiny)
    source_root = _source_imports(source_root, receipt["source_sha256"], allow_preloaded=tiny)
    from ayaka.training.direct_audit import load_audited_bundle

    # A frozen source may contain another scripts namespace. Inspect with the
    # trusted sibling that we will archive, independently of sys.path order.
    inspector_path = Path(__file__).with_name("native_layout.py").resolve()
    inspector_sha = _digest(inspector_path)
    spec = importlib.util.spec_from_file_location("_ayaka_upload_native_layout", inspector_path)
    inspector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspector)

    bundle, native, out = Path(bundle).resolve(), Path(snapshot_path).resolve(), Path(out).resolve()
    if out.exists() or any(out.is_relative_to(root) for root in (source_root, bundle, native)):
        raise ValueError("upload archive must be a new file outside packaged input directories")
    snapshot = inspector.read_record(snapshot_record, expected_snapshot_record_sha256)
    audited = load_audited_bundle(
        bundle,
        audit_receipt,
        expected_manifest_sha256=expected_bundle_sha256,
        expected_receipt_sha256=expected_audit_receipt_sha256,
        native_path=native,
        allow_tiny=allow_tiny,
    )
    if snapshot.get("native_metadata") != audited.recipe["native_metadata"] or (
        snapshot.get("repo"),
        snapshot.get("revision"),
    ) != (audited.recipe["model"]["backbone"], audited.recipe["model"]["backbone_revision"]):
        raise ValueError("upload native snapshot differs from the audited training model")
    layout = inspector.audit_layout(snapshot, native)
    if layout["inspector_sha256"] != inspector_sha or _digest(inspector_path) != inspector_sha:
        raise ValueError("native inspector changed during upload audit")
    files = {}

    def add(name, path, sha256, size=None):
        name, path = _relative(name), Path(path)
        if name in files or not path.is_file():
            raise ValueError("duplicate or missing upload input")
        files[name] = (
            path,
            {"bytes": path.stat().st_size if size is None else size, "sha256": sha256},
        )

    for name, sha in audited.binding["source_sha256"].items():
        add("ayaka/" + _relative(name), source_root / "ayaka" / name, sha)
    for name, sha in audited.manifest["files"].items():
        add("bundle/" + _relative(name), bundle / name, sha)
    add("bundle/manifest.json", bundle / "manifest.json", expected_bundle_sha256)
    add("cpu-audit.json", audit_receipt, expected_audit_receipt_sha256)
    add("native-snapshot.json", snapshot_record, expected_snapshot_record_sha256)
    for name, entry in snapshot["files"].items():
        add("native/" + _relative(name), native / name, entry["sha256"], entry["bytes"])
    for name in ("LICENSE", "pyproject.toml"):
        add(name, source_root / name, _digest(source_root / name))
    if "survey_sha256" in audited.recipe["model_policy"]:
        name = "docs/experiments/v2_candidates.json"
        add(name, source_root / name, audited.recipe["model_policy"]["survey_sha256"])
    for name in ("native_layout.py", "package.py"):
        path = Path(__file__).with_name(name)
        add("scripts/direct_v2/" + name, path, _digest(path))
    manifest = {
        "version": VERSION,
        "files": {name: entry for name, (_, entry) in sorted(files.items())},
        "anchors": {
            "bundle_manifest_sha256": expected_bundle_sha256,
            "cpu_audit_sha256": expected_audit_receipt_sha256,
            "native_snapshot_record_sha256": expected_snapshot_record_sha256,
        },
        "native_layout": layout,
        "required_audit_dependencies": audited.binding["dependencies"],
        "compression_level": 3,
        "default_action": "audit",
        "linux_cuda_runtime_included": False,
        "production_ready": False,
        "original_test_inputs_included": False,
        "gpu_allocated": False,
        "optimizer_steps": 0,
        "mechanics_only": allow_tiny,
    }
    manifest["audit_command"] = _audit_command(manifest["anchors"], allow_tiny)
    out.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with out.open("xb") as raw:
            created = True
            with (
                zstandard.ZstdCompressor(level=3, threads=2, write_checksum=True).stream_writer(
                    raw, closefd=False
                ) as compressed,
                tarfile.open(
                    fileobj=compressed, mode="w|", bufsize=BLOCK, copybufsize=BLOCK
                ) as archive,
            ):
                for name, (path, expected) in sorted(files.items()):
                    info = tarfile.TarInfo(PREFIX + name)
                    info.size, info.mode = expected["bytes"], 0o644
                    with path.open("rb") as stream:
                        checked = CheckedReader(stream)
                        archive.addfile(info, checked)
                        if (
                            checked.size != expected["bytes"]
                            or checked.hasher.hexdigest() != expected["sha256"]
                            or stream.read(1)
                        ):
                            raise ValueError(f"upload input changed while streaming: {name}")
                payload = _canonical(manifest)
                info = tarfile.TarInfo(MANIFEST)
                info.size, info.mode = len(payload), 0o644
                archive.addfile(info, io.BytesIO(payload))
    except BaseException:
        if created:
            out.unlink()
        raise
    return {
        "path": str(out),
        "bytes": out.stat().st_size,
        "sha256": _digest(out),
        "files": len(files),
        "compression_level": 3,
        "production_ready": False,
        "linux_cuda_runtime_included": False,
        "gpu_allocated": False,
        "optimizer_steps": 0,
    }


def _audit_command(anchors, mechanics_only):
    return [
        "python",
        "-m",
        "ayaka.training.run_direct",
        "audit",
        "--bundle",
        "bundle",
        "--expected-bundle-sha256",
        anchors["bundle_manifest_sha256"],
        "--audit-receipt",
        "cpu-audit.json",
        "--expected-audit-receipt-sha256",
        anchors["cpu_audit_sha256"],
        "--snapshot-record",
        "native-snapshot.json",
        "--snapshot-path",
        "native",
        *(["--mechanics-only"] if mechanics_only else []),
    ]


def verify_archive(path, expected_sha256):
    """Verify every compressed archive member without extracting or loading models."""
    if not _sha256_value(expected_sha256):
        raise ValueError("upload archive differs from its external SHA256")
    seen, manifest = {}, None
    with Path(path).open("rb") as raw:
        compressed = CheckedReader(raw)
        with (
            zstandard.ZstdDecompressor().stream_reader(
                compressed, closefd=False, read_across_frames=True
            ) as decompressed,
            tarfile.open(fileobj=decompressed, mode="r|", bufsize=BLOCK) as archive,
        ):
            for member in archive:
                if not member.isfile() or not member.name.startswith(PREFIX):
                    raise ValueError(
                        "upload archive must contain only regular files under its root"
                    )
                name = _relative(member.name.removeprefix(PREFIX))
                if name in seen or manifest is not None:
                    raise ValueError(
                        "upload archive contains duplicate or trailing unbound members"
                    )
                stream = archive.extractfile(member)
                if member.name == MANIFEST:
                    if member.size > 16 * 1024 * 1024:
                        raise ValueError("upload manifest exceeds its size limit")
                    manifest = json.load(stream)
                    continue
                checked = CheckedReader(stream)
                while checked.read(BLOCK):
                    pass
                seen[name] = {"bytes": checked.size, "sha256": checked.hasher.hexdigest()}
            # Consume the zstd checksum and tar's prefetched buffer. Nonzero
            # bytes after tar EOF, including concatenated frames, are unbound.
            while block := archive.fileobj.read(BLOCK):
                if any(block):
                    raise ValueError("upload archive has trailing unbound data after tar end")
        if compressed.hasher.hexdigest() != expected_sha256:
            raise ValueError("consumed upload archive differs from its external SHA256")
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != VERSION
        or manifest.get("files") != seen
    ):
        raise ValueError("upload inventory or payload differs from its manifest")
    if (
        any(
            manifest.get(name) is not False
            for name in (
                "production_ready",
                "linux_cuda_runtime_included",
                "original_test_inputs_included",
                "gpu_allocated",
            )
        )
        or manifest.get("default_action") != "audit"
    ):
        raise ValueError("upload payload must preserve CPU-only, non-promotable preparation scope")
    required = {
        "bundle_manifest_sha256": "bundle/manifest.json",
        "cpu_audit_sha256": "cpu-audit.json",
        "native_snapshot_record_sha256": "native-snapshot.json",
    }
    anchors = manifest.get("anchors")
    if (
        not isinstance(anchors, dict)
        or set(anchors) != set(required)
        or any(
            not _sha256_value(sha) or seen.get(required[name], {}).get("sha256") != sha
            for name, sha in anchors.items()
        )
        or type(manifest.get("mechanics_only")) is not bool
        or manifest.get("audit_command") != _audit_command(anchors, manifest["mechanics_only"])
    ):
        raise ValueError("upload anchors or CPU-only audit command are unbound")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build")
    for name in (
        "source-root",
        "bundle",
        "audit-receipt",
        "snapshot-record",
        "snapshot-path",
        "out",
    ):
        build.add_argument("--" + name, type=Path, required=True)
    for name in ("bundle", "audit-receipt", "snapshot-record"):
        build.add_argument("--expected-" + name + "-sha256", required=True)
    build.add_argument("--allow-tiny", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--expected-sha256", required=True)
    args = parser.parse_args(argv)
    values = vars(args).copy()
    values.pop("action")
    result = (
        build_archive(**values)
        if args.action == "build"
        else verify_archive(values["archive"], values["expected_sha256"])
    )
    if args.action == "verify":
        result = {
            "verified_files": len(result["files"]),
            "production_ready": False,
            "gpu_allocated": False,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
