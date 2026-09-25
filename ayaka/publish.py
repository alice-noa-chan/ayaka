"""Upload an Electra export folder to the Hugging Face Hub.

Dry-run by default: it validates the folder and prints what would be
uploaded. Nothing leaves the machine without ``--yes``.

    python -m ayaka.publish --export runs/exports/electra-small --repo <user>/electra-small
    python -m ayaka.publish --export runs/exports/electra-small --repo <user>/electra-small \\
        --with-code --yes                                 # private upload
    ... --public --yes                                   # public upload

Guards:
- the folder must contain a loadable export and a model card
  (README.md) whose ``<code-url>`` placeholder has been filled in
- repos are created private unless ``--public`` is given
- the token comes from ``HF_TOKEN`` / the huggingface-cli login; it is
  never printed or written
"""

from __future__ import annotations

import json
import os
import shutil
from importlib import resources

REQUIRED = (
    "electra_config.json",
    "head.pt",
    "export_meta.json",
    "backbone/model.safetensors.index.json",
)
PLACEHOLDER = "<code-url>"


def _size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total


def check_export(export_dir: str) -> list[str]:
    """Problems that block an upload (empty list = ready)."""
    problems = [f"missing {r}" for r in REQUIRED if not os.path.exists(os.path.join(export_dir, r))]
    card = os.path.join(export_dir, "README.md")
    if not os.path.exists(card):
        problems.append("missing README.md model card (python -m ayaka.modelcard --export ...)")
    else:
        with open(card, encoding="utf-8") as f:
            text = f.read()
        if not text.startswith("---"):
            problems.append("README.md has no YAML front matter (not a generated model card)")
        if PLACEHOLDER in text:
            problems.append(f"model card still contains {PLACEHOLDER}: regenerate with --code-url")
    return problems


def bundle_code(export_dir: str) -> str:
    """Copy the ayaka source + pyproject into <export>/ayaka_src so the
    folder installs offline: pip install ./ayaka_src"""
    pkg = str(resources.files("ayaka"))
    repo_root = os.path.dirname(pkg)
    dest = os.path.join(export_dir, "ayaka_src")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(
        pkg, os.path.join(dest, "ayaka"), ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in ("pyproject.toml", "README.md"):
        src = os.path.join(repo_root, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, name))
    return dest


def plan(export_dir: str, repo: str, public: bool) -> dict:
    files = []
    for root, _, names in os.walk(export_dir):
        for n in names:
            p = os.path.join(root, n)
            files.append((os.path.relpath(p, export_dir).replace(os.sep, "/"), os.path.getsize(p)))
    return {
        "repo": repo,
        "visibility": "public" if public else "private",
        "files": len(files),
        "total_gb": round(_size(export_dir) / 1e9, 2),
        "largest": sorted(files, key=lambda x: -x[1])[:5],
        "problems": check_export(export_dir),
    }


def upload(export_dir: str, repo: str, public: bool) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=not public, exist_ok=True)
    api.upload_large_folder(
        repo_id=repo, repo_type="model", folder_path=export_dir, private=not public
    )
    return f"https://huggingface.co/{repo}"


def main(argv: list[str] | None = None) -> dict:
    import argparse

    ap = argparse.ArgumentParser(
        description="Upload an Electra export to the Hugging Face Hub (dry-run by default)"
    )
    ap.add_argument("--export", required=True, help="export folder (ayaka.export)")
    ap.add_argument("--repo", required=True, help="<user-or-org>/<name>")
    ap.add_argument(
        "--public", action="store_true", help="create the repo public (default: private)"
    )
    ap.add_argument("--with-code", action="store_true", help="bundle ayaka source as ayaka_src/")
    ap.add_argument("--yes", action="store_true", help="actually upload (otherwise dry-run)")
    args = ap.parse_args(argv)

    if args.with_code:
        bundle_code(args.export)
    report = plan(args.export, args.repo, args.public)
    print(json.dumps(report, indent=2), flush=True)
    if report["problems"]:
        raise SystemExit("[publish] blocked: fix the problems above")
    if not args.yes:
        print("[publish] dry-run OK; re-run with --yes to upload", flush=True)
        return report
    report["url"] = upload(args.export, args.repo, args.public)
    print(f"[publish] uploaded -> {report['url']}", flush=True)
    return report


if __name__ == "__main__":
    main()
