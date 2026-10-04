# Ayaka Swift — evaluator package (draft)

**Status: not yet measured.** The policy file, measured figures and the identity check below are filled in only after
the GPU collection described in [`docs/experiments/AYAKA_V3_SWIFT.md`](../../docs/experiments/AYAKA_V3_SWIFT.md).
Do not submit this package until every `TBD` is replaced by a measured value.

Ayaka Swift answers JevBench's `typesafe` wire format (`POST /v1/systemone`) with **frozen**
`google/gemma-4-12B-it` (no fine-tuning) served by unmodified vLLM 0.30.0. Each question is one forward pass:
the options are shown as letters, and the raw logit of one predeclared canonical token per letter is read at the
first answer position. A fitted policy then applies per-primitive temperatures and, only if the calibration split
selected it, a Noul commitment band. One output token per decision; no generation, no reasoning.

## Run

One 48 GB GPU (TBD: cards measured). The model must be readable from the Hugging Face cache or the Hub.

```bash
docker build -f deploy/swift/Dockerfile -t ayaka-swift .
docker run --gpus all -p 8009:8009 -e HOST=0.0.0.0 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" ayaka-swift
```

Without Docker: `pip install vllm==0.30.0 && pip install --no-deps -e .`, then
`POLICY=deploy/swift/policy.json bash deploy/swift/serve.sh`.

Warm up until this returns HTTP 200:

```bash
curl -s http://127.0.0.1:8009/v1/systemone -H 'Content-Type: application/json' -d '{"state": "warm-up",
  "questions": {"decision": {"type": "noul", "instructions": "Is this a warm-up?",
  "criteria": {"false": "no", "true": "yes"}}}}'
```

Then run the harness:

```bash
python3 -m jevbench.cli run \
  --tasks <JEVBENCH>/datasets/public/easy.jsonl,<JEVBENCH>/datasets/public/original.jsonl,<JEVBENCH>/datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8009 --key-env '' \
  --model ayaka-swift --cost-basis self_hosted_gpu --reserve-usd 0 \
  --results <OUT>/results.jsonl --raw-dir <OUT>/raw --ledger <OUT>/ledger.jsonl --manifest <OUT>/manifest.json \
  --run-label ayaka-swift --delay-s 0
```

**Identity check:** TBD (public correct count and tier split from our own runs).

## Status codes

422 for an input the system does not take (over the context limit, an unknown question type, a reasoning budget,
non-uniform Score levels); 502 when vLLM fails; 400 for a body that is not JSON.

## Disclosures

- Frozen weights: `google/gemma-4-12B-it` revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, Apache-2.0 with
  Google's Gemma Prohibited Use Policy. Swift code: MIT.
- The one-pass option-letter readout is NInfer's approach, as used by Cygnet; the readout definition, decision policy,
  fitting and packaging are ours.
- Options are presented as letters in the benchmark's label order with criteria text verbatim; Noul shows `false` first.
- Policy parameters were fitted only on our non-public calibration split; JevBench public items were never used for
  fitting or selection (they are reported as a diagnostic). TBD: fitted values and selected prompt variant.
- Price basis: Google's gemma-4-12B-it at bf16, input tokens only. TBD: mean input tokens on the public set.
- Context limit 16384 tokens.

## Wire-format check without a GPU

`python deploy/swift/harness_smoke.py --jevbench <JEVBENCH>` runs JevBench's own `typesafe` adapter and scorer
against the Swift server with a fake reader. All 231 public items return valid answers (scores are meaningless).
