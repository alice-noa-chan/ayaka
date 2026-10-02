#!/usr/bin/env bash
# Run from the extracted kit on /workspace; no pip install or model download.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="$ROOT/hf-cache" HF_HUB_CACHE="$ROOT/hf-cache/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="$ROOT/outputs/kernel-cache"
if [ "$#" -eq 0 ]; then set -- train; fi
mkdir -p "$ROOT/outputs/logs"
LOG="$ROOT/outputs/logs/$(date -u +%Y%m%dT%H%M%S)-$$.log"
"$ROOT/runtime/bin/python3.11" -u "$ROOT/scripts/runpod_v2/launch.py" "$@" 2>&1 | tee "$LOG"
echo "[runpod] job exited successfully. A live Pod continues billing until stopped."
