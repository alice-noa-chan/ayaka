"""Repack verified local Linux dependencies and clean data at zstd level three."""

import argparse
import hashlib
import io
import json
import posixpath
import tarfile
from pathlib import Path, PurePosixPath

import zstandard

from ayaka.training.prepare_v2 import canonical
from scripts.runpod_v2.clean_recovery import validate_clean_bundle
from scripts.runpod_v2.package import digest


def trusted_base_member(member):
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("ayaka-v2",):
        raise ValueError("base archive path escaped the kit")
    if len(path.parts) < 2 or path.parts[1] not in {"runtime", "hf-cache"}:
        return False
    if path.parts[1] == "hf-cache" and not member.name.startswith(
        "ayaka-v2/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2/"
    ):
        return False
    if not (member.isfile() or member.isdir() or member.issym()):
        raise ValueError("unsupported dependency archive member")
    if member.issym():
        target = posixpath.normpath(posixpath.join(posixpath.dirname(member.name), member.linkname))
        if member.linkname.startswith("/") or not target.startswith("ayaka-v2/runtime/"):
            raise ValueError("nonportable dependency symlink")
    return not member.isdir()


class CheckedReader:
    def __init__(self, stream):
        self.stream, self.hasher = stream, hashlib.sha256()

    def read(self, size):
        block = self.stream.read(size)
        self.hasher.update(block)
        return block


def clean_archive(base_tar, bundle, parent, out):
    base_tar, bundle, parent, out = map(Path, (base_tar, bundle, parent, out))
    if out.exists():
        raise ValueError("clean archive must be new")
    validate_clean_bundle(bundle, parent)
    repo = Path(__file__).resolve().parents[2]
    if any(
        out.resolve().is_relative_to(p.resolve())
        for p in (bundle, parent, repo / "ayaka", repo / "scripts/runpod_v2")
    ):
        raise ValueError("archive must be outside all packaged source directories")
    records, seen = {}, set()
    created = False
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(base_tar, "r:") as source:
            previous = json.load(source.extractfile("ayaka-v2/kit-manifest.json"))["files"]
            raw = out.open("xb")
            created = True
            with (
                raw,
                zstandard.ZstdCompressor(level=3, threads=4, write_checksum=True).stream_writer(
                    raw
                ) as compressed,
                tarfile.open(
                    fileobj=compressed, mode="w|", bufsize=8 * 1024**2, copybufsize=8 * 1024**2
                ) as archive,
            ):
                for member in source:
                    if not trusted_base_member(member):
                        continue
                    name = member.name.removeprefix("ayaka-v2/")
                    if name in seen or name not in previous:
                        raise ValueError("duplicate or unbound base dependency")
                    seen.add(name)
                    expected = previous[name]
                    member.uid = member.gid = 0
                    member.uname = member.gname = "root"
                    if member.issym():
                        if expected != {"symlink": member.linkname}:
                            raise ValueError("dependency symlink differs from original manifest")
                        archive.addfile(member)
                    else:
                        reader = CheckedReader(source.extractfile(member))
                        archive.addfile(member, reader)
                        if (
                            member.size != expected["bytes"]
                            or reader.hasher.hexdigest() != expected["sha256"]
                        ):
                            raise ValueError(
                                "dependency bytes differ from original manifest: " + name
                            )
                    records[name] = expected
                    if member.size > 64 * 1024**2:
                        print(
                            json.dumps({"verified_dependency": name, "bytes": member.size}),
                            flush=True,
                        )
                needed = {
                    name
                    for name in previous
                    if name.startswith(
                        (
                            "runtime/",
                            "hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2/",
                        )
                    )
                }
                if seen != needed:
                    raise ValueError("incomplete offline base dependencies")
                additions = [
                    (repo / "ayaka", "ayaka"),
                    (repo / "scripts/runpod_v2", "scripts/runpod_v2"),
                    (bundle, "bundle"),
                ]
                parent_files = [
                    parent / "ayaka_config.json",
                    parent / "head.safetensors",
                    *sorted((parent / "adapter").glob("*.json")),
                    *sorted((parent / "adapter").glob("*.safetensors")),
                ]
                files = [
                    (repo / "LICENSE", "LICENSE"),
                    *[
                        (p, "v1-checkpoint/" + p.relative_to(parent).as_posix())
                        for p in parent_files
                    ],
                ]
                for root, prefix in additions:
                    files.extend(
                        (p, prefix + "/" + p.relative_to(root).as_posix())
                        for p in sorted(root.rglob("*"))
                        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
                    )
                for path, name in files:
                    if name in records:
                        raise ValueError("duplicate clean overlay file")
                    records[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
                    member = archive.gettarinfo(str(path.resolve()), "ayaka-v2/" + name)
                    member.uid = member.gid = 0
                    member.uname = member.gname = "root"
                    with path.open("rb") as stream:
                        archive.addfile(member, stream)
                manifest = (
                    canonical(
                        {
                            "version": "ayaka-clean-kit-1",
                            "files": records,
                            "gpu_allocated": False,
                            "optimizer_steps": 0,
                            "default_action": "plan",
                            "compression_level": 3,
                        }
                    )
                    + b"\n"
                )
                member = tarfile.TarInfo("ayaka-v2/clean-kit-manifest.json")
                member.size = len(manifest)
                archive.addfile(member, io.BytesIO(manifest))
        result = {
            "bytes": out.stat().st_size,
            "sha256": digest(out),
            "files": len(records),
            "compression_level": 3,
            "optimizer_steps": 0,
            "runtime_scope": "copied and individually verified from the previously tested Linux kit; new Linux execution not yet measured",
        }
        out.with_name(out.name + ".sha256").write_text(
            f"{result['sha256']}  {out.name}\n", encoding="utf-8"
        )
        out.with_name(out.name + ".receipt.json").write_bytes(canonical(result) + b"\n")
        return result
    except BaseException:
        if created and out.exists():
            out.unlink()  # only this newly-created, incomplete file
        raise


def verify_clean_archive(path):
    path, actual, manifest = Path(path), {}, None
    with path.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as decoded:
        with tarfile.open(fileobj=decoded, mode="r|", bufsize=8 * 1024**2) as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if (
                    not parts
                    or parts[0] != "ayaka-v2"
                    or ".." in parts
                    or member.name.startswith("/")
                ):
                    raise ValueError("clean archive escaped the kit")
                name = member.name.removeprefix("ayaka-v2/")
                if member.issym():
                    if not trusted_base_member(member):
                        raise ValueError("unsupported clean archive symlink")
                    record = {"symlink": member.linkname}
                elif member.isfile():
                    stream = archive.extractfile(member)
                    if name == "clean-kit-manifest.json":
                        if manifest is not None:
                            raise ValueError("duplicate clean manifest")
                        manifest = json.load(stream)
                        continue
                    hasher = hashlib.sha256()
                    while block := stream.read(8 * 1024**2):
                        hasher.update(block)
                    record = {"bytes": member.size, "sha256": hasher.hexdigest()}
                else:
                    raise ValueError("unsupported clean archive member")
                if name in actual:
                    raise ValueError("duplicate clean archive path")
                actual[name] = record
                if member.size > 64 * 1024**2:
                    print(json.dumps({"verified_archive_member": name}), flush=True)
        while decoded.read(8 * 1024**2):
            pass
    if (
        manifest is None
        or manifest.get("version") != "ayaka-clean-kit-1"
        or actual != manifest["files"]
    ):
        raise ValueError("clean archive does not match its content manifest")
    result = {
        "verified": True,
        "files": len(actual),
        "archive_sha256": digest(path),
        "zstd_frame_consumed": True,
    }
    expected = path.with_name(path.name + ".sha256").read_text(encoding="utf-8").split()[0]
    if result["archive_sha256"] != expected:
        raise ValueError("clean archive checksum sidecar mismatch")
    path.with_name(path.name + ".verified.json").write_bytes(canonical(result) + b"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base-tar", "bundle", "parent", "out"):
        parser.add_argument("--" + arg, required=True)
    args = parser.parse_args(argv)
    result = clean_archive(args.base_tar, args.bundle, args.parent, args.out)
    print(json.dumps(result), flush=True)
    print(json.dumps(verify_clean_archive(args.out)), flush=True)
    return result


if __name__ == "__main__":
    main()
