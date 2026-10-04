#!/usr/bin/env bash
# Linux GPU entry point. The Python supervisor owns deadlines and process groups.
# --prompt-variants min,cygnet,rules (default) shares one vLLM server per model.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SWIFT_PYTHON:-python3}" "$SCRIPT_DIR/gpu_runner.py" "$@"
