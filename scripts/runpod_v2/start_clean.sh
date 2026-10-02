#!/usr/bin/env bash
set -euo pipefail
kit_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME="$kit_dir/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$kit_dir"
unset PYTHONHOME
mkdir -p "$kit_dir/outputs"
action="${1:-plan}"
if (( $# )); then shift; fi
exec "$kit_dir/runtime/bin/python3.11" "$kit_dir/scripts/runpod_v2/clean_recovery.py" \
  "$action" --bundle "$kit_dir/bundle" --parent "$kit_dir/v1-checkpoint" "$@"
