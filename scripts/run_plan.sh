#!/usr/bin/env bash
# Staged Electra plan on one GPU box (A100-80GB or H100 recommended).
#
#   STAGES="smoke"                    bash scripts/run_plan.sh   # ~10 min sanity + step-time measurement
#   STAGES="smoke-large"              bash scripts/run_plan.sh   # 12B: 20 steps incl. ~4K-token prompts
#   STAGES="large" LARGE_STEPS=1500 LARGE_MAX_HOURS=8 AUTO_STOP=1 bash scripts/run_plan.sh
#   STAGES="zeroshot small"           bash scripts/run_plan.sh   # small baseline + small LoRA run
#   STAGES="large teacher distill export" bash scripts/run_plan.sh
#
# Knobs (env): SMALL_STEPS LARGE_STEPS BASE_STEPS TEACHER_N WITH_BASE=1 AUTO_STOP=1
#              LARGE_MAX_HOURS -> wall-clock cap on the large optimizer loop (billing
#              guard; calibration and evaluation still run afterwards)
#              ALLOW_RESTRICTED=1 -> also train/label on restricted data (jev_distill
#              with Jev API labels, ANLI, MultiRC, Amazon); default data is license-clean
# AUTO_STOP=1 stops the RunPod pod / vast.ai instance at the end so an idle
# GPU does not keep billing.
set -euo pipefail

export HF_HOME=${HF_HOME:-/workspace/hf-cache}
export AYAKA_ARTIFACTS=${AYAKA_ARTIFACTS:-/workspace/runs}
STAGES=${STAGES:-smoke}
SMALL_STEPS=${SMALL_STEPS:-2000}
BASE_STEPS=${BASE_STEPS:-2000}
LARGE_STEPS=${LARGE_STEPS:-3000}
TEACHER_N=${TEACHER_N:-120000}
WITH_BASE=${WITH_BASE:-0}
LARGE_MAX_HOURS=${LARGE_MAX_HOURS:-0}
CODE_URL=${CODE_URL:-<code-url>}  # git URL of this repo, written into the model cards
R=$AYAKA_ARTIFACTS
P="python -m ayaka.pipeline"
REL=(); TREL=()
if [ "${ALLOW_RESTRICTED:-0}" = 1 ]; then REL=(--set include_restricted=true); TREL=(--allow-restricted); fi

stop_instance() {
  if [ "${AUTO_STOP:-0}" != 1 ]; then return; fi
  if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
    echo "[plan] stopping RunPod pod $RUNPOD_POD_ID"; runpodctl stop pod "$RUNPOD_POD_ID"
  elif [ -n "${CONTAINER_ID:-}" ] && command -v vastai >/dev/null; then
    echo "[plan] stopping vast.ai instance $CONTAINER_ID"; vastai stop instance "$CONTAINER_ID"
  else
    echo "[plan] AUTO_STOP=1 but no runpodctl/vastai context found: stop the instance manually!"
  fi
}
trap stop_instance EXIT

for stage in $STAGES; do
  echo "================ stage: $stage ($(date -u +%H:%M:%S) UTC)"
  case $stage in
    smoke)
      # real E2B, 20 LoRA steps on a small slice: verifies the GPU path and
      # prints per-step time -> use it to refine the cost estimate
      $P train --model electra-small --run smoke --set steps=20 --set log_every=1 \
        --set 'specs=["jev_open","boolq","quality"]' --set limit_per_spec=300 \
        --set 'spec_limits={"jev_open":2000}' --set eval_questions=64 --set eval_every=0 \
        --set calibration_questions=200 --set fidelity_questions=200 ;;
    smoke-large)
      # real 12B, 20 LoRA steps including long-document sources so peak memory
      # and step time reflect ~4K-token prompts -> refine the large-run estimate
      $P train --model electra-large --run smoke-large --set steps=20 --set log_every=1         --set 'specs=["jev_open","quality","synth_long_rules","synth_policy"]'         --set limit_per_spec=300 --set 'spec_limits={"jev_open":2000}'         --set eval_questions=64 --set eval_every=0 --set calibration_questions=200         --set fidelity_questions=0 --set jevbench=false ;;
    zeroshot)
      $P eval --model electra-small --zero-shot ;;
    zeroshot-large)
      $P eval --model electra-large --zero-shot ;;
    small)
      $P train --model electra-small --run small-v1 --set steps="$SMALL_STEPS" "${REL[@]}" ;;
    base)
      $P train --model electra-base --run base-v1 --set steps="$BASE_STEPS" "${REL[@]}" ;;
    large)
      $P train --model electra-large --run large-v1 --set steps="$LARGE_STEPS"         --set max_train_seconds="$(( LARGE_MAX_HOURS * 3600 ))" "${REL[@]}" ;;
    teacher)
      $P teacher --ckpt "$R/large-v1/checkpoint" --out "$R/large-v1/teacher.jsonl" --n-samples "$TEACHER_N" "${TREL[@]}" ;;
    distill)
      $P train --model electra-small --run small-distill --set steps="$SMALL_STEPS" \
        --set teacher_labels="$R/large-v1/teacher.jsonl" "${REL[@]}"
      if [ "$WITH_BASE" = 1 ]; then
        $P train --model electra-base --run base-distill --set steps="$BASE_STEPS" \
          --set teacher_labels="$R/large-v1/teacher.jsonl" "${REL[@]}"
      fi ;;
    export)
      [ -d "$R/large-v1/checkpoint" ] && $P export --ckpt "$R/large-v1/checkpoint" --name electra-large --code-url "$CODE_URL"
      [ -d "$R/small-distill/checkpoint" ] && $P export --ckpt "$R/small-distill/checkpoint" --name electra-small --code-url "$CODE_URL"
      [ -d "$R/base-distill/checkpoint" ] && $P export --ckpt "$R/base-distill/checkpoint" --name electra-base --code-url "$CODE_URL"
      true ;;
    *)
      echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
done
echo "[plan] done. Reports: $R/*/report.json  $R/evals/  exports: $R/exports/"
