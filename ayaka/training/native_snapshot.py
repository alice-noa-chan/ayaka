"""Hash and inspect local pinned safetensors before any native weight allocation."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from safetensors import safe_open

from ..eval.read_artifact import fingerprint
from .direct_state import file_digest
from .native_metadata import inspect_metadata

VERSION = "ayaka-local-native-snapshot-1"
DEPLOYABLE_VERSION = "ayaka-local-native-snapshot-2"


def _filename(name):
    if not isinstance(name, str) or Path(name).name != name or "\\" in name or "/" in name:
        raise ValueError("native snapshot files must be flat safe filenames")
    return name


def inspect_snapshot(repo, revision, *, path=None, require_tokenizer=False):
    """Read headers and hash bytes, without loading tensors or downloading files.

    A pinned local cache supplies provenance. These byte digests bind the actual
    loader inputs; they do not independently certify publisher weight contents.
    """
    if (
        not isinstance(repo, str)
        or not repo
        or (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)
        )
    ):
        raise ValueError("native snapshot requires a repository and immutable revision")
    if path is None:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=repo, revision=revision, local_files_only=True)
    root = Path(path).resolve()
    config = root / "config.json"
    if not config.is_file():
        raise ValueError("cached native configuration is missing")
    names = {"config.json"}
    index = root / "model.safetensors.index.json"
    mapping = None
    if index.exists():
        mapping = json.loads(index.read_bytes()).get("weight_map")
        if (
            not isinstance(mapping, dict)
            or not mapping
            or any(not isinstance(key, str) or not key for key in mapping)
        ):
            raise ValueError("native safetensors index requires an exact nonempty weight map")
        shards = {_filename(name) for name in mapping.values()}
        if any(not name.endswith(".safetensors") for name in shards):
            raise ValueError("native weights must use safetensors shards")
        names.add(index.name)
    else:
        shards = {"model.safetensors"}
    if {p.name for p in root.glob("*.safetensors")} != shards:
        raise ValueError("native shard inventory differs from the declared weight index")
    names.update(shards)
    metadata = None
    if require_tokenizer:
        metadata, _ = inspect_metadata(repo, revision, path=root)
        names.update(metadata["files"])
    for name in ("generation_config.json",):
        if (root / name).is_file():
            names.add(name)
    if any(not (root / name).is_file() for name in names):
        raise ValueError("cached native weights are incomplete; prepare downloads before GPU use")
    tensors, elements, dtypes = {}, 0, Counter()
    for name in sorted(shards):
        with safe_open(root / name, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 -- safe_open is not a mapping
                if key in tensors:
                    raise ValueError("native tensor appears in multiple weight shards")
                if mapping is not None and mapping.get(key) != name:
                    raise ValueError("native safetensors index disagrees with actual shard headers")
                view = handle.get_slice(key)
                shape, dtype = view.get_shape(), view.get_dtype()
                count = math.prod(shape)
                tensors[key] = {"shape": shape, "dtype": dtype, "file": name}
                elements += count
                dtypes[dtype] += count
    if not tensors or mapping is not None and set(mapping) != set(tensors):
        raise ValueError("native weight index is empty or incomplete")
    files = {
        name: {"bytes": (root / name).stat().st_size, "sha256": file_digest(root / name)}
        for name in sorted(names)
    }
    if metadata is not None and any(
        files[name] != entry for name, entry in metadata["files"].items()
    ):
        raise ValueError("native metadata changed during snapshot inspection")
    if {p.name for p in root.glob("*.safetensors")} != shards:
        raise ValueError("native shard inventory changed during snapshot inspection")
    if metadata is not None and inspect_metadata(repo, revision, path=root)[0] != metadata:
        raise ValueError("native metadata changed during snapshot inspection")
    content = {
        "version": DEPLOYABLE_VERSION if require_tokenizer else VERSION,
        "repo": repo,
        "revision": revision,
        "files": files,
    }
    if require_tokenizer:
        content["native_metadata"] = metadata
    return {
        **content,
        "snapshot_sha256": fingerprint(content),
        "tensor_schema_sha256": fingerprint(tensors),
        "tensor_count": len(tensors),
        "stored_weight_elements": elements,
        "dtype_elements": dict(dtypes),
        "weights_loaded": False,
        "network_downloads": 0,
        "scope": "local pinned loader-byte binding and safetensors headers; no runtime proof",
    }, root


def verify_snapshot(record, *, path=None):
    if record.get("version") not in {VERSION, DEPLOYABLE_VERSION}:
        raise ValueError("unsupported native snapshot record")
    current, root = inspect_snapshot(
        record["repo"],
        record["revision"],
        path=path,
        require_tokenizer=record["version"] == DEPLOYABLE_VERSION,
    )
    if current != record:
        raise ValueError("native loader bytes or tensor headers changed since preparation")
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    source, destination = args.path.resolve(), args.out.resolve()
    if destination.exists() or destination == source or source in destination.parents:
        raise ValueError("native snapshot record must be a new file outside the native directory")
    record, _ = inspect_snapshot(args.repo, args.revision, path=source, require_tokenizer=True)
    raw = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(raw)
    print(
        json.dumps(
            {
                "record_sha256": file_digest(destination),
                "snapshot_sha256": record["snapshot_sha256"],
                "model_weights_loaded": False,
                "network_downloads": 0,
                "out": str(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
