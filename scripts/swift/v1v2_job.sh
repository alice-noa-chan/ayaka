#!/bin/bash
# GPU job for docs/experiments/V1ON_V2OFF_PROTOCOL_2026-10-05.md.
# Inputs are uploaded files verified offline (no rebuild on the host):
#   $IN/procedural.jsonl $IN/hard_calibration.jsonl $IN/hard_dev.jsonl
# 1. Swift min reads (vLLM, attempt 7 recipe) on the procedural cohort. The hard cohort's
#    min reads already exist from the hard dev run under the same recipe.
# 2. The published v1 system with its frozen worked-steps route on all three files.
# Fails fast; DONE only after exact unique row counts and the archive checksum.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
IN=${IN:-/workspace/in}
OUT=${OUT:-/workspace/v1v2}
declare -A SHA=(
  [procedural]=abf80932e93b1236296af3e79c399b6ce1b517bbe3ffe7f4eda6bbb31db4c92d
  [hard_calibration]=01f17bc9a454e69f771f4afb0d3d73cf415b09609f6dded36517da36b261b58c
  [hard_dev]=8a237769ee38d51adfa89bf6c6fedb5d49e966196497f165dbf5ca2072c3f7ee
)
declare -A EXPECTED=([procedural]=479 [hard_calibration]=390 [hard_dev]=396)
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/steps.log"; }
VP=""
cleanup() {
  status=$?
  if [ -n "$VP" ] && kill -0 "$VP" 2>/dev/null; then kill "$VP"; wait "$VP" 2>/dev/null || true; fi
  [ "$status" -eq 0 ] || log "FAILED (exit $status)"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
count_unique() {
  python3 -c "import json,sys; ids=[json.loads(l)['id'] for l in open(sys.argv[1]) if l.strip()]; print(len(ids) if len(ids)==len(set(ids)) else -1)" "$1"
}
log "start code $(git rev-parse --short HEAD)"
for name in "${!SHA[@]}"; do echo "${SHA[$name]}  $IN/$name.jsonl" | sha256sum -c --quiet; done
log "inputs verified"

vllm serve google/gemma-4-12B-it --revision $R --tokenizer-revision $R --dtype bfloat16 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --logprobs-mode raw_logits --max-logprobs 26 --chat-template-content-format string \
  --host 127.0.0.1 --port 8000 --served-model-name google/gemma-4-12B-it > "$OUT/vllm.log" 2>&1 &
VP=$!
deadline=$((SECONDS + 900))
until curl -fsS --connect-timeout 5 --max-time 10 http://127.0.0.1:8000/health >/dev/null 2>&1; do
  kill -0 "$VP" 2>/dev/null || { log "vLLM exited during startup"; exit 1; }
  [ $SECONDS -lt $deadline ] || { log "vLLM not healthy after 900s"; exit 1; }
  sleep 3
done
log "vLLM healthy"
python3 -m ayaka.swift.collect "$IN/procedural.jsonl" --output "$OUT/swift_min_procedural.reads.jsonl" \
  --prompt-variant min --backend vllm --vllm-url http://127.0.0.1:8000 --model google/gemma-4-12B-it \
  --revision $R --tokenizer-model google/gemma-4-12B-it --tokenizer-revision $R \
  --chat-template-kwargs '{"enable_thinking": false}' --group-size 20 --concurrency 16 \
  > "$OUT/swift_collect.log" 2>&1
[ "$(count_unique "$OUT/swift_min_procedural.reads.jsonl")" -eq 479 ] || { log "swift procedural count wrong"; exit 1; }
log "swift min procedural 479 reads"
kill "$VP"; wait "$VP" 2>/dev/null || true; VP=""

python3 scripts/swift/matched_native.py --prepare-checkpoint --output "$OUT/v1_checkpoint" > "$OUT/v1_prepare.log" 2>&1
CK=$(python3 -c "import json;print(json.load(open('$OUT/v1_checkpoint/checkpoint.json'))['checkpoint_path'])")
log "v1 checkpoint ready"
for name in procedural hard_calibration hard_dev; do
  python3 scripts/swift/v1_on_runner.py "$IN/$name.jsonl" --checkpoint "$CK" \
    --output "$OUT/v1_$name.rows.jsonl" --max-seq-len 8192 > "$OUT/v1_$name.log" 2>&1
  n=$(count_unique "$OUT/v1_$name.rows.jsonl")
  [ "$n" -eq "${EXPECTED[$name]}" ] || { log "v1 $name has $n unique rows, expected ${EXPECTED[$name]}"; exit 1; }
  log "v1 $name $n rows"
done
tar -C "$(dirname "$OUT")" -czf "$OUT.tgz" "$(basename "$OUT")"
sha256sum "$OUT.tgz" > "$OUT.tgz.sha256"
log "DONE"
