#!/bin/bash
# Hard dev reads for docs/experiments/SWIFT_HARD_DEV_PROTOCOL_2026-10-05.md.
# Rebuilds the frozen hard calibration/dev files, refuses them unless both hashes match,
# then reads all four prompt variants with the attempt 7 vLLM recipe. Selects nothing.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
OUT=${OUT:-/workspace/hard}
CAL_SHA=01f17bc9a454e69f771f4afb0d3d73cf415b09609f6dded36517da36b261b58c
DEV_SHA=8a237769ee38d51adfa89bf6c6fedb5d49e966196497f165dbf5ca2072c3f7ee
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/steps.log"; }
log "start code $(git rev-parse --short HEAD)"

python3 scripts/swift/build_hard_dev.py --output "$OUT/data" > "$OUT/build.log" 2>&1 || { log "build FAILED"; exit 1; }
echo "$CAL_SHA  $OUT/data/calibration.jsonl" | sha256sum -c --quiet || { log "calibration hash MISMATCH"; exit 1; }
echo "$DEV_SHA  $OUT/data/dev.jsonl" | sha256sum -c --quiet || { log "dev hash MISMATCH"; exit 1; }
log "hard data verified"

vllm serve google/gemma-4-12B-it --revision $R --tokenizer-revision $R --dtype bfloat16 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --logprobs-mode raw_logits --max-logprobs 26 --chat-template-content-format string \
  --host 127.0.0.1 --port 8000 --served-model-name google/gemma-4-12B-it > "$OUT/vllm.log" 2>&1 &
VP=$!
for _ in $(seq 1 200); do curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1 && break; sleep 3; done
curl -fsS http://127.0.0.1:8000/health >/dev/null || { log "vLLM failed to start"; kill $VP; exit 1; }
log "vLLM healthy"

for variant in min cygnet rules labeled; do
  mkdir -p "$OUT/$variant"
  for role in calibration dev; do
    python3 -m ayaka.swift.collect "$OUT/data/$role.jsonl" \
      --output "$OUT/$variant/hard_$role.reads.jsonl" --prompt-variant $variant \
      --backend vllm --vllm-url http://127.0.0.1:8000 --model google/gemma-4-12B-it --revision $R \
      --tokenizer-model google/gemma-4-12B-it --tokenizer-revision $R \
      --chat-template-kwargs '{"enable_thinking": false}' --group-size 20 \
      --concurrency 16 > "$OUT/$variant/collect_$role.log" 2>&1 \
      && log "$variant/$role $(wc -l < "$OUT/$variant/hard_$role.reads.jsonl") reads" \
      || log "$variant/$role collect FAILED"
  done
done
kill $VP
tar -C "$(dirname "$OUT")" -czf "$OUT.tgz" "$(basename "$OUT")" && sha256sum "$OUT.tgz" > "$OUT.tgz.sha256"
log "DONE"
