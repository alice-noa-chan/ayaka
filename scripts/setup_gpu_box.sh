#!/usr/bin/env bash
# One-time setup on a fresh RunPod / vast.ai instance (PyTorch + CUDA 12 image).
#
#   git clone <this repo> ayaka && cd ayaka && bash scripts/setup_gpu_box.sh
#
# Keeps model weights and datasets on the persistent volume ($WORK) so a
# restarted pod does not download ~10-25 GB again.
set -euo pipefail

WORK=${WORK:-/workspace}
# Ubuntu 24.04 images (e.g. runpod/pytorch torch 2.8) mark the system Python
# as externally managed; torch already lives there and the box is disposable.
export PIP_BREAK_SYSTEM_PACKAGES=1
export HF_HOME=${HF_HOME:-$WORK/hf-cache}
export AYAKA_ARTIFACTS=${AYAKA_ARTIFACTS:-$WORK/runs}
mkdir -p "$HF_HOME" "$AYAKA_ARTIFACTS"

python -m pip install -U pip
python -m pip install -e ".[dev]"
# optional fused kernels; training runs without them (--set liger=true to use)
python -m pip install liger-kernel || echo "[setup] liger-kernel unavailable, continuing"

nvidia-smi --query-gpu=name,memory.total --format=csv
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "bf16", torch.cuda.is_bf16_supported())
print("gpu", torch.cuda.get_device_name(0))
EOF
python -m pytest -q -x   # CPU unit tests on the tiny backbone (~1 min)

cat <<EOF
[setup] done. Persist these in your shell:
  export HF_HOME=$HF_HOME
  export AYAKA_ARTIFACTS=$AYAKA_ARTIFACTS
Next: STAGES="smoke" bash scripts/run_plan.sh
EOF
