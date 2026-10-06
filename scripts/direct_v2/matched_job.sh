#!/bin/bash
# Run only after separate GPU authorization. All downloads/preparation are local first.
# IN contains protocol.json, checkpoint.json, policy.json, procedural/hard JSONL,
# hard_calibration.reads.jsonl and hard_dev.reads.jsonl. PROTOCOL_SHA256 is predeclared.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${IN:?set prepared input directory}"
: "${OUT:?set a fresh result directory outside checkpoint/input}"
: "${PROTOCOL_SHA256:?set protocol SHA fixed before v1 observation}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ -e "$OUT" ] || [ -L "$OUT" ]; then
  echo "OUT must be fresh; refusing old receipts or DONE" >&2
  exit 1
fi
mkdir "$OUT"
shared=(--protocol "$IN/protocol.json" --protocol-sha256 "$PROTOCOL_SHA256"
  --procedural "$IN/procedural.jsonl" --hard-calibration "$IN/hard_calibration.jsonl"
  --hard-dev "$IN/hard_dev.jsonl")
hard=("$IN/hard_calibration.reads.jsonl" "$IN/hard_dev.reads.jsonl")
VP=""
CP=""
cleanup() {
  status=$?
  if [ -n "$CP" ] && kill -0 "$CP" 2>/dev/null; then
    kill "$CP" 2>/dev/null || true
    wait "$CP" 2>/dev/null || true
  fi
  if [ -n "$VP" ] && kill -0 "$VP" 2>/dev/null; then
    kill "$VP" 2>/dev/null || true
    wait "$VP" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# Both CPU gates MUST finish before any GPU service or model load.
python3 -m scripts.direct_v2.matched_preflight "${shared[@]}" --policy "$IN/policy.json" --v2-reads "${hard[@]}"
python3 -m scripts.direct_v2.matched_v1 "${shared[@]}" --checkpoint-receipt "$IN/checkpoint.json" \
  --output "$OUT/v1.rows.jsonl" --receipt "$OUT/preflight.json" --preflight-only

# An old healthy server must never satisfy this job's readiness check.
python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
    probe.bind(("127.0.0.1", 8000))
PY
R=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
vllm serve google/gemma-4-12B-it --revision "$R" --tokenizer-revision "$R" --dtype bfloat16 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --logprobs-mode raw_logits --max-logprobs 26 --chat-template-content-format string \
  --host 127.0.0.1 --port 8000 --served-model-name google/gemma-4-12B-it > "$OUT/vllm.log" 2>&1 &
VP=$!
deadline=$((SECONDS + 900))
while true; do
  kill -0 "$VP" 2>/dev/null || { echo "vLLM exited" >&2; exit 1; }
  if curl -fsS --connect-timeout 5 --max-time 10 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    kill -0 "$VP" 2>/dev/null || { echo "vLLM exited during readiness" >&2; exit 1; }
    break
  fi
  [ "$SECONDS" -lt "$deadline" ] || { echo "vLLM startup timeout" >&2; exit 1; }
  sleep 3
done
python3 -m ayaka.swift.collect "$IN/procedural.jsonl" --output "$OUT/procedural.reads.jsonl" \
  --prompt-variant min --backend vllm --vllm-url http://127.0.0.1:8000 --model google/gemma-4-12B-it \
  --revision "$R" --tokenizer-model google/gemma-4-12B-it --tokenizer-revision "$R" \
  --chat-template-kwargs '{"enable_thinking": false}' --group-size 20 --concurrency 16 &
CP=$!
while kill -0 "$CP" 2>/dev/null; do
  kill -0 "$VP" 2>/dev/null || { echo "vLLM exited during collection" >&2; exit 1; }
  sleep 1
done
wait "$CP"
CP=""
kill -0 "$VP" 2>/dev/null || { echo "vLLM exited before collection completed" >&2; exit 1; }
kill "$VP"
wait "$VP" 2>/dev/null || true
VP=""
python3 -m scripts.direct_v2.matched_v1 "${shared[@]}" --checkpoint-receipt "$IN/checkpoint.json" \
  --output "$OUT/v1.rows.jsonl" --receipt "$OUT/execution.json"
execution_sha=$(sha256sum "$OUT/execution.json")
execution_sha=${execution_sha%% *}
python3 -m scripts.direct_v2.matched_compare "${shared[@]}" --policy "$IN/policy.json" \
  --v2-reads "${hard[@]}" "$OUT/procedural.reads.jsonl" --v1-rows "$OUT/v1.rows.jsonl" \
  --v1-execution-receipt "$OUT/execution.json" --v1-execution-receipt-sha256 "$execution_sha" \
  --output "$OUT/comparison.json"
sha256sum "$OUT/comparison.json" > "$OUT/DONE"
