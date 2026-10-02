#!/usr/bin/env bash
# CPU-only Linux preparation; run before renting a GPU.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "usage: bash prepare_runtime.sh /absolute/new/runtime-directory" >&2
  exit 2
fi
DEST=$1
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ -e "$DEST" ] || [[ "$DEST" != /* ]]; then
  echo "runtime destination must be a new absolute path" >&2
  exit 2
fi
command -v uv >/dev/null
uv python install 3.11.14
BASE=$(uv python find 3.11.14)
BASE=$(readlink -f "$BASE")
mkdir -p "$(dirname -- "$DEST")"
cp -a "$(dirname -- "$(dirname -- "$BASE")")" "$DEST"
# The copied prefix belongs to this kit, rather than uv's managed installation.
uv pip install --python "$DEST/bin/python3.11" --break-system-packages \
  --python-platform x86_64-manylinux_2_28 --require-hashes --only-binary :all: \
  --link-mode copy -r "$HERE/requirements.lock.txt"
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" "$DEST/bin/python3.11" -c \
  'import torch, torchvision, transformers, peft; print(torch.__version__, torch.version.cuda, torchvision.__version__, transformers.__version__, peft.__version__)'
