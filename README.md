# Ayaka: an open, license-clean Jev-class decision model

> **Ayaka v2 large is out:** [alice-noa-chan/ayaka-v2-large](https://huggingface.co/alice-noa-chan/ayaka-v2-large).
> On an unseen held-out cohort it gains +10.8 CC over v1 large (Choice and Noul up, Score within noise),
> and it adds a learned reasoning router, generated Choice candidates and image inputs.
> It did not pass the project's predeclared zero-tolerance gate on Score, so the v1 models stay
> available and unchanged. See [Ayaka v2](#ayaka-v2).
> The JevBench public-tier results below describe v1.

<p align="center"><img src="docs/assets/ayaka.png" width="256" alt="Ayaka"></p>

Hand it a **state** and typed questions (`noul` / `choice` / `score`) with
runtime candidate sets. It returns a calibrated probability for every
candidate. Most decisions take a single forward pass with no generation.

## Why this exists

[JevBench](https://github.com/fstandhartinger/jevbench) ranks "decision
models": systems that answer structured questions about a state with a
probability per label, judged on accuracy, calibration, speed and cost. The
best-known system, TypeSafe's Jev, is closed, and its API terms forbid
training on its outputs. Ayaka is an open alternative:

- **MIT-licensed code and weights**, trained only on data whose licence allows
  it (see [Data](#data)). No commercial-LLM outputs, no Jev API labels.
- **Probabilities, not just labels.** Temperatures are fitted per question
  type and prompt length on held-out data, and every candidate gets a
  probability.
- **Fast by default.** A state is encoded once and every question branches off
  it. Only low-confidence calculation questions take the optional
  worked-steps route (below).

The project started as an attempt to build this on Google's ELECTRA encoder.
ELECTRA's 512-token window cannot hold JevBench's long hard-tier documents,
which average about 1.2K and reach about 6K tokens. At that size it also
lacks the capacity for multi-step rules and arithmetic: comparable encoders
(ModernBERT-large) reach only 39-44/111 on the public hard tier. Ayaka keeps the
ELECTRA-era decision contract (question isolation, permutation-invariant
candidates, per-primitive calibration, the pointer head) and puts it on
**Gemma 4** instruction-tuned backbones. Almost all accuracy comes from the
backbone. The learned pointer gate stays near 0 (0.001-0.04 on 12B), and it
is kept for candidate sets larger than the 26-letter label alphabet.

Published Ayaka v1 checkpoints are **text-only**: only the Gemma 4 text stack
is loaded, and states are text or JSON. Ayaka v2 adds experimental native
image inputs that load the pinned vision components separately; see
[Images and generated candidates](docs/MULTIMODAL_AND_CANDIDATES.md).

## Results: Ayaka v2 large against v1 large (held-out)

[Ayaka v2 large](https://huggingface.co/alice-noa-chan/ayaka-v2-large) was
compared with v1 large on the final_test cohort: 836 questions and 431 cases
from validation and test splits that neither model trained on. No fit or
choice saw the cohort, and it was read once after v2's serving policy was
frozen. These are local measurements, not official JevBench scores. Hardware:
one RTX PRO 6000, one request at a time.

| system | equal-type CC | Choice CC / NLL | Noul CC / NLL | Noul abstentions | Score CC / NLL | Speed axis | mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| **[Ayaka v2 large](https://huggingface.co/alice-noa-chan/ayaka-v2-large), frozen policy (default)** | **81.0** | **82.6 / 0.376** | **93.2 / 0.118** | **2** | 67.2 / 0.812 | 58.9 | 3.46 s |
| Ayaka v2 large, single pass | 73.7 | 79.0 / 0.437 | 75.0 / 0.176 | 35 | 67.2 / 0.804 | 85.9 | 0.13 s |
| [Ayaka large](https://huggingface.co/alice-noa-chan/ayaka-large) (v1) + worked-steps route | 70.2 | 75.4 / 0.507 | 67.2 / 0.197 | 50 | 68.0 / 0.792 | 82.4 | 0.41 s |

- **Overall:** v2 gains +10.8 CC over v1 (paired 95% CI [+7.2, +14.2]).
- **Score:** within noise overall (CI [−5.1, +3.9]). HelpSteer2 rubric questions are slightly worse (CI [−8.6, −0.1]).
- **New in v2:**
  - a learned reasoning router, with Noul always reasoned;
  - experimental generated Choice candidates with an Other residual;
  - experimental image inputs.
- **Full report:** [V2_FINAL_RUN_RESULT_2026-10-10](docs/experiments/V2_FINAL_RUN_RESULT_2026-10-10.md). v2 has not yet been scored on the JevBench public tiers below.

## Results (JevBench public tiers, v1 models)

The public items are 48 easy, 72 original and 111 hard. The Intelligence
proxy is chance-corrected with the official tier weights over these public
tiers only. It is not the leaderboard score (see
[Benchmark disclosure](#benchmark-disclosure)).

| system | easy | original | hard | Intelligence proxy |
|---|---|---|---|---|
| **[Ayaka large](https://huggingface.co/alice-noa-chan/ayaka-large) + worked-steps route** (bundled server, self-hosted) | 48/48 | 72/72 | **88/111** | **87.0** |
| [Ayaka large](https://huggingface.co/alice-noa-chan/ayaka-large), single pass | 48/48 | 72/72 | 72/111 | 77.9 |
| [Ayaka base](https://huggingface.co/alice-noa-chan/ayaka-base) (E4B), single pass | 48/48 | 68/72 | 61/111 | 68.6 |
| [Ayaka small](https://huggingface.co/alice-noa-chan/ayaka-small) (E2B), single pass | 48/48 | 65/72 | 55/111 | 62.8 |
| Jev 1.13.0 (TypeSafe) | 48/48 | 71/72 | 81/111 | 82.3 |
| Winnow-12B Q8 | 48/48 | 69/72 | 81/111 | 80.7 |
| SemIf / OpenJev (Qwen3.5-4B) | 48/48 | 71/72 | 68/111 | 74.9 |

- **Probability quality, hard tier.** ECE 0.152 → 0.090 and NLL 0.913 →
  0.542 with the route. The NLL figure is from the offline collection, which
  scored 89/111.
- **Independent check.** On a procedural test set of 500 questions, never
  used for any choice, the route moved accuracy from 348 to 417 (82 fixed,
  13 broken, exact McNemar p ≈ 2e-13).
- **Ordinary decisions.** On Open-Jev test (300), 4% of questions are
  routed and accuracy goes from 273 to 274.
- **Latency** (A100 SXM 80GB, one request at a time):

  | tier | single pass p50 / p95 | with `--reasoning` p50 / p95 |
  |---|---|---|
  | easy | 0.18 / 0.19 s | 0.29 / 0.30 s |
  | original | 0.18 / 0.18 s | 0.29 / 0.30 s |
  | hard | 0.19 / 0.80 s | 9.9 / 31.4 s |

  Single-pass numbers were measured in-process; `--reasoning` numbers went
  through the HTTP server (+~0.1 s). On H100 the single-pass p50 is 0.15 s.
- **Tokens per decision** (what `usage` reports):

  | cohort | input | output |
  |---|---|---|
  | Open-Jev test | 348 | 10 |
  | public original | 146 | 0 |
  | public hard | 1,242 | 160 |

  At the leaderboard's 12B reference price ($0.05 per million input
  tokens) that is $0.007-0.062 per 1,000 decisions.
- **Setup.** Large is Gemma 4 12B with LoRA r64, 860 steps (about 55K
  questions), on one H100. Base (E4B) and small (E2B) use LoRA r32 and
  1,200 steps on one A100 each, trained directly on the same data and code
  (no distillation). They are released as single pass: the worked-steps
  route was measured and frozen only for large. Single-pass p50 on A100 is
  0.07 s for small and 0.18 s for base. Hard-tier ECE is 0.238 for small and
  0.223 for base.
- **Weights:**
  [ayaka-large](https://huggingface.co/alice-noa-chan/ayaka-large),
  [ayaka-base](https://huggingface.co/alice-noa-chan/ayaka-base),
  [ayaka-small](https://huggingface.co/alice-noa-chan/ayaka-small).

## Quick start

Ayaka is released as open weights and code, and there is no hosted API. You
run the bundled server on your own GPU. It speaks the same
`/v1/systemone` protocol JevBench uses to call decision models. "small" and
"large" are model sizes (Gemma 4 E2B and 12B), not service tiers.

```bash
pip install -c constraints.txt -e .   # verified versions; see also the Dockerfile
# single pass (merged export or checkpoint)
python -m ayaka.serve --model <export-dir> --device cuda --port 8000
# with the worked-steps route (checkpoint with the LoRA unmerged)
python -m ayaka.serve --ckpt <checkpoint-dir> --reasoning --device cuda --port 8000
# v2 large with its frozen policy, router and temperatures (read from frozen_tuning.json)
python -m ayaka.serve --ckpt alice-noa-chan/ayaka-v2-large --device cuda --port 8000
# score any running server on the public tiers
python -m ayaka.eval.jevbench --endpoint http://127.0.0.1:8000 --out report.json
```

The server speaks TypeSafe's `POST /v1/systemone` format (see
[`ayaka/serve.py`](ayaka/serve.py)). Its response gives a probability for
every label:

- `noul`: P(true);
- `choice`: `probabilities` per label, the argmax label and Jev `confidence`;
- `score`: `probabilities` per level, the expected score, Jev `confidence` and a
  `legend` preserving the original criteria values.

`usage.output_tokens` counts tokens generated by the worked-steps route.

The official TypeSafe Python SDK is supported (`typesafe-sdk==0.7.2`, available
through `pip install -e ".[sdk-compat]"`). Both `jev-latest` (the SDK default)
and `jev-preview`, as well as the served Ayaka name and versioned id, resolve to
the same local model. The response reports Ayaka's own id, by default
`<served-name>-1.0.0`; set `ayaka.serve --model-id` to override it. These aliases
provide interoperability and do not identify Ayaka as Jev. `GET /v1/models`
returns `{"models": [{"name": ..., "description": ..., "release_date": ...}]}`;
`--model-description` and `--model-release-date` configure that metadata.

Set `AYAKA_API_KEY` before starting the server to require Bearer authentication
on `/v1/systemone` and `/v1/models`, or use `--api-key-env CUSTOM_KEY` to choose
another environment variable. `--max-inflight` (default 32) bounds active and
queued decisions; excess requests receive 429 with `retry-after`. The request's
top-level `ayaka` object carries Ayaka extensions (reasoning settings, image
media and per-question candidate generation); see
[Ayaka v2](docs/experiments/AYAKA_V2.md). The optional
[`ayaka/client.py`](ayaka/client.py) helper wraps the official SDK.

## Worked-steps route

A question is routed only if all three conditions hold:

1. it asks about a quantity. The question text or a choice option contains a
   number or a quantity word. Rubric level numbers such as "0: not helpful" do
   not count;
2. the state has at least three numbers;
3. the single-pass top probability is at most 0.9.

A routed question is answered in two steps:

1. The base model, with the decision LoRA disabled, writes at most 384
   tokens of worked steps: which numbers and dates it uses, conversions,
   rules and exceptions, and the arithmetic. It is told not to name an
   option.
2. The trained model then reads the source with those notes appended. Its
   distribution replaces the single-pass one.

The policy was selected on a procedural dev set (500) and Open-Jev test (300)
and frozen before public or test scoring
(`FROZEN_REASONING_POLICY` in `ayaka/evidence_pipeline.py`).

A zero-shot JSON calculation DSL route was also built and measured, and it
failed: only 24 of 324 dev plans were valid. It remains in the code
(`readout="executed"`) and would need its own LoRA.

## Benchmark disclosure

No JevBench items or labels were included in the training corpus. All 35
training sources were checked against the public items for shared 13-grams,
and 0 samples matched. However, public JevBench results were used as a
development signal. They helped identify weak task families and informed
synthetic-data design, model-size selection, and the decision to develop a
reasoning path. Public-item outputs were also read to diagnose failures (a
zero-shot calculation-plan format, a calibration bug). One prompt example
that structurally mirrored a public item was removed before the reported
runs.

**Not selected on public results:**

- **Reasoning-route threshold, gating and fusion.** This covers the
  calculation gate rules, the confidence cutoff of 0.9 and the fusion
  weight. They were selected on a procedural dev set (500 questions) and
  Open-Jev test (300 questions) only. The policy was frozen and
  hash-recorded before any public scoring.
- **Worked-steps writer.** Using the base model with the adapter off,
  rather than with the adapter on, was decided on the dev set.
- **Checkpoint.** Each model is the final checkpoint of its run. No
  intermediate checkpoint was compared on public items.
- **Training length and hyperparameters.** Steps were set by budget (860
  for large, 1,200 for small and base). Learning rate, LoRA rank and the
  data-mixture quotas were fixed before training and not re-tuned on public
  scores. The synthetic sources themselves were designed as described
  above.
- **Calibration temperatures.** They were fitted on held-out
  training-distribution data.
- **Freeze.** The released models are frozen for this submission, and no
  further changes will be made in response to public results.

## Model family

| size | weights | backbone | for | released as |
|---|---|---|---|---|
| small | [alice-noa-chan/ayaka-small](https://huggingface.co/alice-noa-chan/ayaka-small) | `google/gemma-4-E2B-it` | lowest latency, memory and cost | single pass |
| base | [alice-noa-chan/ayaka-base](https://huggingface.co/alice-noa-chan/ayaka-base) | `google/gemma-4-E4B-it` | a middle ground | single pass |
| large | [alice-noa-chan/ayaka-large](https://huggingface.co/alice-noa-chan/ayaka-large) | `google/gemma-4-12B-it` | accuracy, especially the hard tier | single pass, or `--reasoning` |
| v2 large | [alice-noa-chan/ayaka-v2-large](https://huggingface.co/alice-noa-chan/ayaka-v2-large) | `google/gemma-4-12B-it` | held-out accuracy, fewer abstentions | its frozen policy by default (`frozen_tuning.json`) |

Each repository holds the LoRA adapter (bf16), `head.safetensors` and the
config. The base model is fetched at a pinned revision. Merged bf16/int8
exports can still be produced with `ayaka.pipeline export` (the size
estimates are below), but the worked-steps route needs the unmerged
checkpoint.

| size | bf16 export | int8 export | train GPU |
|---|---|---|---|
| small | ~9.3 GB | 4.6 GB | A100-80GB (≥24GB) |
| base | ~15 GB | ~8 GB | A100-80GB (≥32GB) |
| large | ~24 GB | ~12 GB | H100 / A100-80GB |

## Architecture

```
prefix: instructions + <state>…</state>          encoded ONCE per state (KV cache)
  └─ suffix_q: question + options + answer cue    one isolated branch per question
       ├─ answer-position state h_q ─► label readout  softcap(h_q · E[label_i])
       └─ option-span states r_i ─► Set Mixer (no positions) ─► pointer <W_q h_q, W_k R_i>
logit_i = label_i + g[primitive, length] · pointer_i (≤ 26 options; g starts at 0)
logit_i = pointer_i → top-26 shortlist → label re-rank   (larger sets)
p = softmax(logit / T[primitive, length])          (T fitted on a held-out split)
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
- **Length-specific pointer mixing.** Each primitive learns separate short/long
  gates at the configured 1,024-token boundary. The long compositional data can
  train a different correction strength from short decisions. Existing
  three-value gates load into both buckets, preserving their readout behavior;
  temperatures retain the same migration rule. Accuracy gains require retraining
  and held-out measurement.
- **Sparse candidate pooling.** Gather and sum only candidate span tokens,
  avoiding a full-sequence fp32 cumulative sum and cancellation from subtracting
  long prefix sums. E2B/E4B also normalize only gathered option states below the
  shared layers. Outputs and gradients match independent local means. At
  no-grad evaluation, all-zero active gates with label readouts skip the pointer;
  training, active gates and pointer-only sets retain the branch. An unused
  `DecisionOutput.pointer_logits` is zero on this evaluation path.
  CPU collation prepares token coordinates and segment offsets before device
  transfer, avoiding dynamic index construction and synchronization on CUDA.
- **Exact compute skipping on E2B/E4B.** The last 20 (E2B) / 18 (E4B) Gemma 4
  layers are KV-shared: they read the K/V of earlier layers and never produce
  their own. Ayaka therefore runs only the answer position through them,
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
`ayaka/training/run.py`). Labels come from people or documented procedures,
never from commercial-LLM outputs. Each source's license is recorded in the
run manifest and in the model card. Every training sample that shares a
13-gram with a JevBench public item is dropped.

The source table, mixture quotas, generated data families and the
calibration procedure are in [Training data and calibration](docs/DATA.md).

## Documentation

- [Development guide](docs/DEVELOPMENT.md): evaluation, training, export,
  Hugging Face release and local test commands.
- [Training data and calibration](docs/DATA.md).
- [Experiment index](docs/experiments/README.md): every v2 experiment report
  with its question, result and current status.
- [Images and generated candidates](docs/MULTIMODAL_AND_CANDIDATES.md).
- [Swift evaluator package](deploy/swift/README.md) (frozen 12B readout, draft).
- Cloud kits: [Modal](scripts/modal/README.md), [RunPod](scripts/runpod_v2/README.md),
  [Beam](scripts/beam_v2/README.md).

## Ayaka v2

[Ayaka v2 large](https://huggingface.co/alice-noa-chan/ayaka-v2-large) is a fresh
rank-64 LoRA on Gemma 4 12B, trained on a license-clean corpus with StrategyQA
added, with checkpoint selection and a frozen held-out serving fit. The final
run is reported in
[V2_FINAL_RUN_RESULT_2026-10-10](docs/experiments/V2_FINAL_RUN_RESULT_2026-10-10.md).
On the unseen final_test cohort (836 questions, 431 cases, read once; local CC,
not an official JevBench score):

| System | Equal-type CC | Choice CC | Noul CC | Score CC | Speed axis |
|---|---:|---:|---:|---:|---:|
| v1 large with its worked-steps route | 70.2 | 75.4 | 67.2 | 68.0 | 82.4 |
| v2 large, single pass | 73.7 | 79.0 | 75.0 | 67.2 | 85.9 |
| **v2 large, frozen policy** | **81.0** | **82.6** | **93.2** | 67.2 | 58.9 |

- **Gain over v1:** +10.8 CC, paired 95% CI [+7.2, +14.2].
- **Abstentions:** Noul abstentions fall from 50 to 2.
- **Score:** within noise overall (CI [−5.1, +3.9]). HelpSteer2 rubric questions are slightly worse (CI [−8.6, −0.1]). For that reason v2 failed the predeclared zero-tolerance per-type gate. It is published as a separate model with the result disclosed, and v1 stays frozen.
- **New capabilities (experimental):**
  - generated Choice candidates with an Other residual;
  - native image inputs ([details](docs/MULTIMODAL_AND_CANDIDATES.md)).
- **JevBench public tiers:** not yet measured for v2.

The validated recipe is the default of the training CLIs
([`ayaka/training/v2_recipe.py`](ayaka/training/v2_recipe.py)), and
`python -m ayaka.frozen_tuning package` builds a release folder from a trained
checkpoint and its frozen fit. Controls and API fields are described in
[Ayaka v2](docs/experiments/AYAKA_V2.md); all reports are listed in the
[experiment index](docs/experiments/README.md).
