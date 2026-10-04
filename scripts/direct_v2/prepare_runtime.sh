#!/usr/bin/env bash
# Build a new portable Linux runtime on CPU before paid allocation.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: bash prepare_runtime.sh /absolute/new/runtime-directory LOCK_SHA256" >&2
  exit 2
fi
RUNTIME_DEST=$1
LOCK_SHA=$2
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ -e "$RUNTIME_DEST" ] || [ -L "$RUNTIME_DEST" ] || [[ "$RUNTIME_DEST" != /* ]]; then
  echo "runtime destination must be a new absolute path" >&2
  exit 2
fi
if [[ ! "$LOCK_SHA" =~ ^[0-9a-f]{64}$ ]]; then
  echo "provide the separately trusted lock SHA256" >&2
  exit 2
fi
if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
  echo "runtime preparation requires Linux x86_64" >&2
  exit 2
fi
printf '%s  %s\n' "$LOCK_SHA" "$HERE/requirements.lock.txt" | sha256sum --check --status
command -v uv >/dev/null
uv python install 3.11.14
PYTHON_BASE=$(uv python find 3.11.14)
PYTHON_BASE=$(readlink -f "$PYTHON_BASE")
mkdir -p "$(dirname -- "$RUNTIME_DEST")"
# CPython terminfo contains case-distinct paths. A normal Windows mount can
# collapse these even though it supports symlinks; use a Linux filesystem.
"$PYTHON_BASE" - "$RUNTIME_DEST" <<'PY'
import sys
import tempfile
from pathlib import Path

parent = Path(sys.argv[1]).parent
with tempfile.TemporaryDirectory(prefix=".ayaka-runtime-case-", dir=parent) as probe:
    Path(probe, "case_probe").touch()
    if Path(probe, "CASE_PROBE").exists():
        raise ValueError("runtime destination requires a case-sensitive Linux filesystem")
PY
cp -a "$(dirname -- "$(dirname -- "$PYTHON_BASE")")" "$RUNTIME_DEST"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
# Exact wheels only: no nvcc/source build can consume paid GPU startup time.
uv pip install --python "$RUNTIME_DEST/bin/python3.11" --break-system-packages \
  --require-hashes --only-binary :all: --link-mode copy \
  -r "$HERE/requirements.lock.txt"
uv pip check --python "$RUNTIME_DEST/bin/python3.11"
cp "$HERE/requirements.lock.txt" "$RUNTIME_DEST/requirements.lock.txt"
cp "$HERE/runtime.py" "$RUNTIME_DEST/runtime_smoke.py"
"$RUNTIME_DEST/bin/python3.11" "$RUNTIME_DEST/runtime_smoke.py" \
  --lock "$RUNTIME_DEST/requirements.lock.txt" --expected-lock-sha256 "$LOCK_SHA" \
  --out "$RUNTIME_DEST/cpu-smoke.json"
