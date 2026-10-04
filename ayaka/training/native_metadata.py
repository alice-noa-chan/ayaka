"""Path-independent bindings for offline native configuration/tokenizer loaders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..backbone import tiny_text_config
from ..eval.read_artifact import fingerprint

VERSION = "ayaka-native-loader-metadata-1"
_SUFFIXES = {".json", ".jinja", ".model", ".txt", ".vocab", ".merges", ".tiktoken"}


def local_root(repo, revision, path=None):
    if (
        not isinstance(repo, str)
        or not repo
        or repo != "tiny"
        and (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)
        )
    ):
        raise ValueError("cached native configuration requires an immutable revision")
    if path is not None:
        root = Path(path).resolve()
        if not root.is_dir():
            raise ValueError("explicit native directory is missing; refuse cache fallback")
        return root
    if repo == "tiny":
        return None
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo, revision=revision, local_files_only=True)).resolve()


def _config_record(root):
    if any(
        (root / name).exists()
        for name in ("adapter_config.json", "adapter_model.bin", "adapter_model.safetensors")
    ):
        raise ValueError("fresh native base must not contain an embedded adapter")
    config = root / "config.json"
    if not config.is_file():
        raise ValueError("native configuration is missing; refuse cache fallback")
    raw = config.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get("model_type"), str):
        raise ValueError("native configuration requires a built-in model_type")
    if "configuration_files" in value:
        raise ValueError(
            "native configuration redirects are unsupported; provide canonical config.json"
        )
    # Hash the complete file before AutoConfig/meta construction, including
    # same-shape rope, scaling, dropout and non-text subconfiguration changes.
    return {"kind": "config_file", "sha256": hashlib.sha256(raw).hexdigest()}


def configuration_binding(cfg, *, path=None):
    root = local_root(cfg.backbone, cfg.backbone_revision, path)
    if root is None:
        return {"kind": "generated_tiny", "sha256": fingerprint(tiny_text_config().to_dict())}, None
    return _config_record(root), root


def _file_record(path):
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def inspect_metadata(repo, revision, *, path=None):
    """Bind a complete fast-tokenizer package without reading any model weights.

    Include versioned tokenizer files and all locally preferred chat templates.
    File symlinks in the HF cache are allowed: digests cover their target bytes.
    Paths, cache names and absolute tokenizer name_or_path are never identity.
    """
    from transformers.utils import CHAT_TEMPLATE_DIR

    root = local_root(repo, revision, path)
    if root is None:
        return None, None
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if any(not (root / name).is_file() for name in required):
        raise ValueError(
            "native upload requires config.json, tokenizer.json and tokenizer_config.json"
        )
    _config_record(root)
    names = {
        p.name
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in _SUFFIXES and not p.name.endswith(".index.json")
    }
    for directory in {CHAT_TEMPLATE_DIR, "chat_templates"}:
        template_dir = root / directory
        if template_dir.exists():
            if not template_dir.is_dir():
                raise ValueError("native chat template assets must be a directory")
            names.update(
                p.relative_to(root).as_posix() for p in template_dir.glob("*.jinja") if p.is_file()
            )
    tokenizer_config = json.loads((root / "tokenizer_config.json").read_bytes())
    if not isinstance(tokenizer_config, dict):
        raise ValueError("native tokenizer configuration must be an object")
    # AutoTokenizer can prefer version-specific fast files. Reject unbound or
    # external paths before letting its loader select one.
    versioned = tokenizer_config.get("fast_tokenizer_files", [])
    if not isinstance(versioned, list) or any(
        not isinstance(name, str)
        or Path(name).name != name
        or "\\" in name
        or "/" in name
        or name not in names
        for name in versioned
    ):
        raise ValueError("versioned tokenizer files must belong to the bound native metadata")
    content = {
        "version": VERSION,
        "repo": repo,
        "revision": revision,
        "files": {name: _file_record(root / name) for name in sorted(names)},
    }
    return {**content, "metadata_sha256": fingerprint(content)}, root


def verify_metadata(record, repo, revision, *, path=None):
    current, root = inspect_metadata(repo, revision, path=path)
    if current != record:
        raise ValueError("native configuration/tokenizer assets changed since preparation")
    return root
