# Electra (ayaka)

A Jev-class decision model family: hand it a **state** and typed questions
(`noul` / `choice` / `score`) with runtime candidate sets, get back a
calibrated probability distribution per question in one forward pass — no
generation.

Electra runs on pretrained **Gemma 4** instruction-tuned backbones and keeps the
decision contract from the design doc:

| size  | backbone                | bf16 export | int8 export | train GPU         |
| ----- | ----------------------- | ----------- | ----------- | ----------------- |
| small | `google/gemma-4-E2B-it` | ~9.3 GB     | 4.6 GB      | A100-80GB (≥24GB) |
| base  | `google/gemma-4-E4B-it` | ~15 GB      | ~8 GB       | A100-80GB (≥32GB) |
| large | `google/gemma-4-12B-it` | ~24 GB      | ~12 GB      | H100 / A100-80GB  |

## Architecture

```
prefix: instructions + <state>…</state>          encoded ONCE per state (KV cache)
  └─ suffix_q: question + options + answer cue    one isolated branch per question
       ├─ answer-position state h_q ─► label readout  softcap(h_q · E[label_i])
       └─ option-span states r_i ─► Set Mixer (no positions) ─► pointer <W_q h_q, W_k R_i>
logit_i = label_i + g[primitive] · pointer_i       (≤ 26 options; g starts at 0)
logit_i = pointer_i → top-26 shortlist → label re-rank   (larger sets)
p = softmax(logit / T[primitive])                  (T fitted on a held-out split)
```

Guarantees, each covered by `tests/test_model.py`:

- **Question isolation.** Suffixes attend only to the shared prefix and themselves.
  Batched multi-question results equal single-question runs.
- **Candidate permutation equivariance, by construction.** Choice options are
  displayed in a canonical order derived from their content. Any input
  permutation renders the identical prompt, so the distribution moves exactly
  with the candidates.
- **Zero-shot preservation.** With `g = 0` the output is exactly the backbone's
  restricted LM-head readout of the label tokens. Training only moves away from
  that as far as it helps.
- **Exact compute skipping on E2B/E4B.** The last 20 (E2B) / 18 (E4B) Gemma 4
  layers are KV-shared: they read the K/V of earlier layers and never produce
  their own. Electra therefore runs only the answer position through them,
  reads option spans just below them, and encodes a shared prefix only up to
  them. Outputs and gradients are unchanged (tests), and on E2B CPU p50 went
  from 2.4 s to 0.99 s (easy) and from 42 s to 14 s (hard). This is 2.5–3.0×,
  matching the ~70% of per-token FLOPs in E2B's double-wide shared layers.
  12B has no KV sharing, so this does not apply to it.
- **Constant prompt head encoded once.** The instructions before `<state>` are
  identical in every request, and in a causal model their KV does not depend
  on what follows. The server therefore encodes them once and copies the
  cache. SDPA also takes its `is_causal` fast path (flash on GPU) whenever
  no mask is needed. Together on E2B CPU p50: easy 0.99 → 0.69 s, hard
  14.0 → 12.7 s. The result is exact: in fp32 the difference is ≤ 1e-5.
- **Windowed sliding attention.** Gemma 4 sliding layers only see the last
  512 keys, yet masked SDPA computes the full S×S. Queries now run in blocks
  against the key range their mask allows. A guard falls back to the full
  range whenever the mask would allow a key outside the block. Tests match to
  1e-10 in float64, including gradients. On E2B CPU hard items it measured
  1.18× in bf16 (13.8 → 11.8 s) and was neutral in fp32. Short inputs never
  engage it.
- **bf16 vs fp32.** On CPU, bf16 activations moved a few near-tie answers by
  up to 0.2 probability relative to fp32. `ayaka.serve --dtype float32`
  follows the reference probabilities more closely at about 1.3–1.5× CPU
  latency. bf16 stays the default, and training uses bf16 too.
- **Only the text stack is loaded.** Vision and audio towers never reach memory,
  and the 262K-vocab LM head is never materialized.

Training uses LoRA on attention and MLP projections plus the pointer head.
Losses are proper scoring rules: NLL/KL to soft targets, Brier, RPS for score,
a missing-evidence overconfidence penalty, and an auxiliary pointer-only NLL.
Distillation (Large → Base/Small) swaps gold NLL for KL-to-teacher on
teacher-labeled questions.

## Data

By default only license-clean sources are used (`DEFAULT_SPECS` in
`ayaka/training/run.py`). Each source's license is recorded in the run manifest
and in the model card.

- **Typed decisions:** from
  [`SargeDev/jev-distill-corpus-v3`](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3)
  only the `openjev_v2` stream (spec `jev_open`) is used: 74.6K train rows
  (noul/choice), CC0-1.0, whose labels were made without any commercial API.
  It is 52% of the mixture.
- **Not used by default** (`include_restricted=true` / `ALLOW_RESTRICTED=1` opts in):
  - the corpus's `yuri_v3` stream (about 444K rows). Its labels are outputs of
    TypeSafe's Jev API, and TypeSafe's Master Customer Agreement (updated
    2026-09-23) §2.3(b) forbids using "any Output to perform model distillation,
    train a model to imitate the output of the Services, or develop … a similar
    or competing product". OpenRouter's terms §5.1 pass provider terms through
    to its users.
  - `yuri_v1`, labeled by an unidentified "32B teacher".
  - ANLI (CC BY-NC), MultiRC (unclear terms) and Amazon reviews (Amazon's
    terms).
- **Other sources:** multilingual (ko/ja) NLU, high-cardinality intent and
  human soft-label sets, plus QuALITY for long evidence.
- **Calibration and evaluation** use the `openjev_v2` rows of the corpus's
  calibration and `test_set_30k` splits (1,572 / 2,319 rows). Primitives those
  splits lack (score) are topped up from a reserved training slice.

Every training sample that shares a 13-gram with a JevBench public item is
dropped (`ayaka/data/decontam.py`).

Measured prompt lengths (Gemma 4 tokenizer): jev-distill mean 191 tokens (p99
602), JevBench hard mean 1,242 (max 3,892). `max_seq_len` is 4096. To cover
long evidence, QuALITY articles that fit (about 1,100 questions, mean 2.8K
tokens) make up 6% of the mixture. Longer articles are skipped instead of
truncated, because cutting the middle could remove the answer's evidence.

## Evaluation

- **JevBench public tiers** (`python -m ayaka.eval.jevbench`). The 231 public
  items ship with the package (MIT). Results are compared item-for-item with
  the published per-task outcomes of Jev 1.13.0 and other systems on the same
  items. It also reports a chance-corrected Intelligence proxy using the
  official tier weights over the public tiers.
- **Held-out reference set**: accuracy, KL and ECE against the reference
  distributions of `jev_open_test` (Open-Jev rows of `test_set_30k`).
- **Held-out mixture metrics** during training.

## Training (RunPod / vast.ai / Modal / any CUDA box)

One CLI drives every stage (`python -m ayaka.pipeline`), and
`scripts/run_plan.sh` chains the stages:

```bash
git clone <repo> ayaka && cd ayaka && bash scripts/setup_gpu_box.sh   # RunPod / vast.ai
STAGES="smoke"                  bash scripts/run_plan.sh   # ~15 min: GPU path + measured step time
STAGES="zeroshot small"         bash scripts/run_plan.sh   # small baseline, then small LoRA
STAGES="large teacher distill export" AUTO_STOP=1 bash scripts/run_plan.sh
# WITH_BASE=1 also distills Base; STAGES="base" trains Base directly

modal run modal_app.py --cmd "train --model electra-small --run small-v1"            # Modal A100-80GB
modal run modal_app.py --cmd "train --model electra-large --run large-v1" --gpu h100  # Modal H100
```

Speed defaults: activation checkpointing is off, because it recomputes every
layer and costs about 30%. On a CUDA OOM the step is retried with half the
micro-batch, and checkpointing only turns on if that is not enough, so one
config fits 24–80 GB cards. `--set liger=true` enables Liger's fused
RMSNorm/GeGLU after a parity check. Liger targets Gemma 4 31B, not E2B/E4B.
`--set compile=true` compiles each decoder layer (experimental).
FlashAttention is not used. Gemma 4's global-attention layers have head_dim
512, above FA2/FA3's limit of 256, and SDPA already picks the fused kernel
where one applies.

## Export for the benchmark (bf16 + int8)

```bash
python -m ayaka.pipeline export --ckpt runs/small-distill/checkpoint --name electra-small
# -> runs/exports/electra-small/       bf16, LoRA merged
#    runs/exports/electra-small-int8/  int8 weights (half the size) + parity report vs bf16
python -m ayaka.serve --model runs/exports/electra-small-int8 --device cpu   # TypeSafe /v1/systemone
```

Each folder is self-contained: text-only Gemma 4 weights, tokenizer, head,
config and a README. The benchmark runs it through `ayaka.serve`, which speaks
the TypeSafe `/v1/systemone` protocol JevBench already uses. All questions in
a request share one state encoding.

The int8 runtime modes were measured on Gemma 4 E2B zero-shot, JevBench
public tiers, 8-core x86 CPU. The latencies below were taken before the
KV-shared compute skipping. With it, int8 `auto` measures easy 97.9% /
original 88.9% at p50 0.97 s.

| `--linear-mode` | easy | original | p50 | RAM |
|---|---|---|---|---|
| bf16 export | 97.9% | 88.9% | 2.6–3.0 s | ~9.3 GB |
| **int8 `auto`** (int8 weights, bf16 GEMM) | 96–100% | **88.9%** | ~3.0 s | ~6.6 GB |
| int8 `mixed` (dynamic int8 on RMSNorm-fed projections) | 100% | 81.9% | 2.1 s | ~5 GB |
| int8 `dynamic` (dynamic int8 everywhere) | 54% | 33% | 1.5 s | ~4.7 GB |

PyTorch dynamic int8 uses one activation scale per tensor, and Gemma's
activation outliers dominate that scale. That is why the default keeps
GEMMs in bf16. Per-token int8 GEMM (`torch._int_mm`) was about 100× slower
than bf16 on this CPU.

## Public release (Hugging Face)

```bash
STAGES="small large teacher distill" bash scripts/run_plan.sh   # license-clean data (default)
CODE_URL=https://github.com/<you>/ayaka STAGES="export" bash scripts/run_plan.sh
python -m ayaka.publish --export runs/exports/electra-small --repo <you>/electra-small --with-code         # dry-run
python -m ayaka.publish --export runs/exports/electra-small --repo <you>/electra-small --with-code --yes   # private upload
```

- Every export gets a generated `README.md` model card. It has front matter,
  JevBench public-tier results beside the reference systems, held-out
  reference-set metrics, int8 parity, the per-source data and license table
  with a provenance statement, and limitations.
- `ayaka.publish` is a dry-run unless `--yes` is given, and it creates private
  repos unless `--public` is given. It refuses to upload while the card still
  has the `<code-url>` placeholder. `--with-code` bundles the source as
  `ayaka_src/` for an offline `pip install ./ayaka_src`.

## Running locally

```bash
uv pip install -e ".[dev]"
pytest                                    # CPU, random tiny Gemma 4 stack, no downloads
python -m ayaka.eval.jevbench --zero-shot electra-small --device cpu --dtype bfloat16
python -m ayaka.eval.jevbench --export runs/exports/electra-small-int8 --device cpu --dtype bfloat16
```

beam.cloud serverless has only T4/A10G/RTX 4090/RTX 5090. `beam_train.py`
therefore exposes the same pipeline CLI on 4090/5090 for small-model work,
for example `b.pipeline_4090.remote(["eval", "--model", "electra-small", "--zero-shot"])`.

A checkpoint directory contains `electra_config.json`, the LoRA `adapter/`,
`head.pt` (pointer head, gate, temperatures), `meta.json`, and alongside it
`report.json` and `jevbench_report.json`.
