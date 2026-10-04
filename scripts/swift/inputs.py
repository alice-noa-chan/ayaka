"""Shared offline input validation and exact decision/read inventory."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from ayaka.swift.collect import iter_dataset
from ayaka.swift.grouping import validate_option_count

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "scripts/swift/manifest.json"
PACKED_MANIFEST = "scripts/swift/inputs.manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"input paths must be relative to the repo: {value}")
    if path.name == "test.jsonl":
        raise ValueError("test.jsonl is held out and must remain unopened")
    return path.as_posix()


def inventory(
    root: Path,
    manifest: Path,
    *,
    overrides: dict[str, Path] | None = None,
    group_size: int = 20,
) -> tuple[dict, dict[str, Path]]:
    """Validate all input items; never scan the runs directory or held-out set."""
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or not spec.get("datasets"):
        raise ValueError("manifest needs schema_version=1 and nonempty datasets")
    group_size = int(spec.get("group_size", group_size))
    validate_option_count(2, group_size)
    root = root.resolve()
    files: dict[str, Path] = {}

    def locate(value: str) -> Path:
        name = safe_relative(value)
        candidate = (overrides or {}).get(name, root / name).resolve()
        if candidate.name == "test.jsonl":
            raise ValueError("test.jsonl is held out and must remain unopened")
        if name not in (overrides or {}) and not candidate.is_relative_to(root):
            raise ValueError(f"input escapes repository: {name}")
        if not candidate.is_file():
            raise ValueError(f"missing input: {name}")
        files[name] = candidate
        return candidate

    datasets = []
    names = set()
    for dataset in spec["datasets"]:
        name = dataset["name"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in names:
            raise ValueError(f"invalid or duplicate dataset name: {name}")
        names.add(name)
        paths = [safe_relative(path) for path in dataset["paths"]]
        if not paths:
            raise ValueError(f"dataset is empty: {name}")
        sources = [locate(path) for path in paths]
        ids = set()
        reads = records = 0
        for item in iter_dataset(sources):
            if item.id in ids:
                raise ValueError(f"duplicate id in {name}: {item.id}")
            ids.add(item.id)
            count = len(item.question.labels)
            validate_option_count(count, group_size)
            reads += 1 if count <= 26 else (count + group_size - 1) // group_size + 1
        for source in sources:
            with source.open(encoding="utf-8") as stream:
                records += sum(bool(line.strip()) for line in stream)
        if not ids:
            raise ValueError(f"no items in {name}")
        if dataset.get("expected_items", len(ids)) != len(ids):
            raise ValueError(
                f"{name}: expected {dataset['expected_items']} items, found {len(ids)}"
            )
        hashes = {path: sha256(source) for path, source in zip(paths, sources, strict=True)}
        if "sha256" in dataset and dataset["sha256"] != hashes:
            raise ValueError(f"input hash mismatch: {name}")
        datasets.append(
            {
                "name": name,
                "paths": paths,
                "records": records,
                "items": len(ids),
                "expected_items": len(ids),
                "reads": reads,
                "sha256": hashes,
            }
        )
    notices = spec.get("notices", [])
    for notice in notices:
        locate(notice)
    return {
        "schema_version": 1,
        "group_size": group_size,
        "datasets": datasets,
        "notices": notices,
    }, files
