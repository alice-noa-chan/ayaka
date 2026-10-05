#!/usr/bin/env bash
# Start pinned vLLM and the Swift TypeSafe server on one GPU; stop both on exit.
#
#   POLICY=deploy/swift/policy.json bash deploy/swift/serve.sh
#
# Environment (defaults in brackets):
#   MODEL [google/gemma-4-12B-it]  REVISION [707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7]
#   SERVED_NAME [$MODEL; the reader loads its tokenizer by this id]  VLLM_PORT [8890]  PORT [8009]  HOST [127.0.0.1]
#   MAX_MODEL_LEN [16384]  GPU_MEMORY_UTILIZATION [0.90]  POLICY (required)
#   PROMPT_VARIANT (must match the policy; read from it when unset)
#   API_KEY_ENV [AYAKA_API_KEY]  ALLOW_NO_KEY [0]  MAX_INFLIGHT [32]  REQUEST_TIMEOUT_S [60]
#   MODEL_ID [ayaka-swift-<revision prefix>]  MODEL_DESCRIPTION  MODEL_RELEASE_DATE [2026-10-04]
set -euo pipefail

MODEL="${MODEL:-google/gemma-4-12B-it}"
REVISION="${REVISION:-707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7}"
SERVED_NAME="${SERVED_NAME:-$MODEL}"
VLLM_PORT="${VLLM_PORT:-8890}"
PORT="${PORT:-8009}"
HOST="${HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
: "${POLICY:?set POLICY to the fitted policy.json}"
PROMPT_VARIANT="${PROMPT_VARIANT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("prompt_variant","min"))' "$POLICY")}"
SERVICE_ARGS=(--api-key-env "${API_KEY_ENV:-AYAKA_API_KEY}"
  --max-inflight "${MAX_INFLIGHT:-32}" --request-timeout-s "${REQUEST_TIMEOUT_S:-60}"
  --model-id "${MODEL_ID:-ayaka-swift-${REVISION:0:12}}"
  --model-release-date "${MODEL_RELEASE_DATE:-2026-10-04}")
if [[ "${ALLOW_NO_KEY:-0}" == "1" ]]; then SERVICE_ARGS+=(--allow-no-key); fi
if [[ -n "${MODEL_DESCRIPTION:-}" ]]; then SERVICE_ARGS+=(--model-description "$MODEL_DESCRIPTION"); fi
# Check public-bind authentication before starting the GPU backend.
python3 -c 'import os,sys; from ayaka.http_transport import ServiceConfig; ServiceConfig(api_key_env=sys.argv[2], allow_no_key=sys.argv[3] == "1").validate_bind(sys.argv[1])' \
  "$HOST" "${API_KEY_ENV:-AYAKA_API_KEY}" "${ALLOW_NO_KEY:-0}"

# Raw logits for explicitly requested token ids: the Swift reader gathers one canonical
# token per option letter and rejects clipped, missing or duplicate values.
vllm serve "$MODEL" --revision "$REVISION" --tokenizer-revision "$REVISION" \
  --served-model-name "$SERVED_NAME" --host 127.0.0.1 --port "$VLLM_PORT" \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --dtype bfloat16 --enable-prefix-caching \
  --logprobs-mode raw_logits --max-logprobs 26 --chat-template-content-format string &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 600); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vLLM exited during startup" >&2; exit 1; fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null

exec python3 -m ayaka.swift.server --backend vllm \
  --vllm-url "http://127.0.0.1:${VLLM_PORT}" --model "$SERVED_NAME" --revision "$REVISION" \
  --policy "$POLICY" --prompt-variant "$PROMPT_VARIANT" \
  --host "$HOST" --port "$PORT" "${SERVICE_ARGS[@]}"
