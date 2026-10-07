# Development guide

Evaluation, training, export and release commands for the published v1 recipe.
Run every command from the repository root.

## Evaluation

- **JevBench public tiers** (`python -m ayaka.eval.jevbench`). The 231 public
  items ship with the package (MIT). Results are compared item-for-item with
  the published per-task outcomes of Jev 1.13.0 and other systems on the same
  items. It also reports a chance-corrected Intelligence proxy using the
  official tier weights over the public tiers.
  `summary.leaderboard_estimate` applies the v1.4.2.1 leaderboard axes:
  Speed from the original tier's p50/p95 after the self-hosted ×2 + 0.15 s
  adjustment, and Cost from the board's own estimate for the backbone. The
  composite is the harmonic mean of the four axes, times (axis/50)² for
  Intelligence, Speed or Cost below 50. It is reported only when a
  Calibration value is supplied. The Intelligence proxy is optimistic: there
  is no judge tier, no sealed items and no public-minus-sealed gap penalty.
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

modal run scripts/modal/train_v1.py --cmd "train --model electra-small --run small-v1"            # Modal A100-80GB
modal run scripts/modal/train_v1.py --cmd "train --model electra-large --run large-v1" --gpu h100  # Modal H100
```

Multi-question states are encoded once in training too. Open-Jev states
carry about 15 questions each and HelpSteer2 has 5 ratings per response. Their
questions branch off one prefix KV cache instead of repeating the state per
question. Loss and gradients match full rows (tests), and on the real mixture
this saves 75% / 70% of the compute tokens for those sources and 17.5% overall.
Sharing turns off automatically if activation checkpointing is enabled,
because HF layers drop KV caches under checkpointing.

Mixture quotas apply to **questions**, not states. Multi-label sources cannot
multiply their quota by the number of labels: a state contributes a random
subset when its questions exceed the remaining cell budget. Each step has
exactly `questions_per_step` questions, while selected siblings still share
their prefix. Validation and calibration reserve whole state/lineage groups
across sources, so another question or translation of a held-out state cannot
remain in training. `data_summary.json` records population counts and label
balance; `history.json` records the actual source/family question counts and
long-question exposure per optimizer step.

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
CODE_URL=https://github.com/alice-noa-chan/ayaka STAGES="export" bash scripts/run_plan.sh
python -m ayaka.publish --export runs/exports/electra-large --repo alice-noa-chan/ayaka-large --with-code         # dry-run
python -m ayaka.publish --export runs/exports/electra-large --repo alice-noa-chan/ayaka-large --with-code --yes   # private upload
```

Uploads use the `hf` CLI login (`hf auth login`). The JevBench request
template is in [`docs/JEVBENCH_REQUEST.md`](JEVBENCH_REQUEST.md).

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
ruff check . && ruff format --check .
python -m ayaka.eval.jevbench --zero-shot electra-small --device cpu --dtype bfloat16
python -m ayaka.eval.jevbench --export runs/exports/electra-small-int8 --device cpu --dtype bfloat16
```

beam.cloud serverless has only T4/A10G/RTX 4090/RTX 5090. `beam_train.py`
therefore exposes the same pipeline CLI on 4090/5090 for small-model work,
for example `b.pipeline_4090.remote(["eval", "--model", "electra-small", "--zero-shot"])`.

Tests marked `prepared_corpus` read the local, git-ignored
`runs/v2-pretraining-20261002-ready/` corpus and are skipped with a stated
reason when it is absent. CI (`.github/workflows/ci.yml`) runs the same lint,
format and CPU test commands on every push and pull request, with the
versions pinned in `constraints.txt`.

A checkpoint directory contains `ayaka_config.json` (and the same config as
legacy `electra_config.json`), the LoRA `adapter/`, `head.safetensors`
(pointer head, gate, temperatures), `meta.json`, and alongside it
`report.json` and `jevbench_report.json`. See `ayaka/checkpoint.py` for the
full layout.
