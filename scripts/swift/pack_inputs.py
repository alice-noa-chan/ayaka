"""Pack only Swift collection inputs into a deterministic, offline tar.gz."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.swift.inputs import (  # noqa: E402
    DEFAULT_MANIFEST,
    PACKED_MANIFEST,
    REPO,
    inventory,
    sha256,
)


def pack_inputs(
    root: Path,
    manifest: Path,
    output: Path,
    *,
    jevbench_dir: Path | None = None,
) -> dict:
    overrides = {}
    if jevbench_dir is not None:
        for name in ("easy.jsonl", "hard.jsonl", "original.jsonl", "LICENSE"):
            overrides[f"ayaka/eval/data/jevbench_public/{name}"] = jevbench_dir / name
    resolved, files = inventory(root, manifest, overrides=overrides)
    if output.resolve() in {path.resolve() for path in [*files.values(), manifest]}:
        raise ValueError("output must not overwrite an input or the manifest")
    manifest_bytes = (json.dumps(resolved, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive,
    ):
        for name in sorted([*files, PACKED_MANIFEST]):
            info = tarfile.TarInfo(name)
            info.mode = 0o644
            # TarInfo defaults have fixed uid/gid, owner names, and mtime=0.
            if name == PACKED_MANIFEST:
                info.size = len(manifest_bytes)
                archive.addfile(info, io.BytesIO(manifest_bytes))
            else:
                info.size = files[name].stat().st_size
                with files[name].open("rb") as stream:
                    archive.addfile(info, stream)
    output.with_name(output.name + ".sha256").write_text(
        f"{sha256(output)}  {output.name}\n", encoding="ascii"
    )
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=REPO / "scripts/swift/inputs.tar.gz")
    parser.add_argument(
        "--jevbench-dir", type=Path, help="complete audit-clone public dataset fallback"
    )
    args = parser.parse_args(argv)
    try:
        resolved = pack_inputs(
            args.repo, args.manifest, args.output, jevbench_dir=args.jevbench_dir
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for dataset in resolved["datasets"]:
        print(f"{dataset['name']}: {dataset['items']} decisions, {dataset['reads']} model reads")
    print(f"Packed {args.output} (+ .sha256); held-out test.jsonl was not opened")


if __name__ == "__main__":
    main()
