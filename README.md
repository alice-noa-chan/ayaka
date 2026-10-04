# Ayaka: an open, license-clean Jev-class decision model

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

Ayaka is **text-only**. Gemma 4 is multimodal, but only its text stack is
loaded. The vision and audio towers never reach memory, and states are
text or JSON.

## Results (JevBench public tiers)

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

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient(api_key="local-key", base_url="http://127.0.0.1:8000") as client:
    result = client.system_one(
        state={"ticket": "my parcel never arrived"},
        questions={
            "urgent": Noul(instructions="Is this urgent?"),
            "intent": Choice(instructions={"task": "classify"},
                             criteria={"refund": "refund", "track": None}),
            "severity": Score(instructions=["Rate severity"],
                              criteria=["low", {"level": "medium"}, "high"]),
        },
        extra_body={"ayaka": {}},
    )
    print(result.model, result.choices["intent"].confidence, result.scores["severity"].legend)
    print(client.models.list())
```

Instructions and each criterion accept strings, JSON objects or arrays; Choice
criterion values may also be `null`. Choice accepts 2–255 options and Score
2–10 levels. Invalid requests return 422 with an `error` and `field`. Choice
confidence measures the top probability's excess above uniform; Score confidence
measures concentration around the first most likely level using Jev's uniform
mean absolute deviation. The shared implementation is in
[`ayaka/jev_api.py`](ayaka/jev_api.py).

Set `AYAKA_API_KEY` before starting the server to require Bearer authentication
on `/v1/systemone` and `/v1/models`, or use `--api-key-env CUSTOM_KEY` to choose
another environment variable. An unset or empty key keeps unauthenticated
serving available; health endpoints remain public. `--max-inflight` (default
32) bounds active and queued decisions; excess requests receive 429 with
`retry-after`. Backend overload receives 529 with the same header. Responses
include `x-typesafe-request-id`; request bodies are limited to 24 MiB.

The request's top-level `ayaka` object is reserved and accepted but ignored,
including all its contents: it has no effect on v1 inference. The server's
existing timing extension is now returned as `ayaka.latency_ms`. The optional
[`ayaka/client.py`](ayaka/client.py) helper wraps the official SDK unchanged and
retains that namespace with `AyakaResponse` or `system_one(client, ...)`.

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
`ayaka/training/run.py`). Labels come from people or documented procedures,
never from commercial-LLM outputs. Each source's license is recorded in the
run manifest and in the model card.

| Family (quota) | Sources (license) | ≈ questions available |
|---|---|---|
| Typed decisions (17%) | Open-Jev `openjev_v2` via jev-distill (CC0); Open-Jev `browser-drone-expansion-v1-redistributable` (CC0, ~15 questions per state) | 75K + 109K |
| Judge (10%) | HelpSteer2 (CC BY 4.0, five 0–4 human ratings); hh-rlhf (MIT, human preference); Aegis 2.0 (CC BY 4.0, human safety labels only) | 100K + 30K + 33K |
| Reasoning (10%) | AQuA-RAT (Apache-2.0); HotpotQA yes/no + comparisons over 10 paragraphs (CC BY-SA 4.0); StrategyQA (MIT); ARC-Challenge (CC BY-SA 4.0); CommonsenseQA (MIT) | 30K + 12K + 2K + 1K + 10K |
| Policy / rules (8%) | LegalBench: 118 tasks whose README states CC BY 4.0 or MIT (list vendored in `ayaka/data/legalbench_tasks.json`); generated long reimbursement policies (`synth_policy`, see below) | ≤ 47K + 15K |
| Temporal / numeric (10%) | Generated order records, invoices and month-end / timezone certificates (`synth_temporal`, `synth_numeric`, `synth_calendar`) | 20K + 20K + 60K |
| Compositional (10%) | Generated long procurement cases (`synth_long_rules`), exact rule / ledger / FX reference labels | 30K |
| Probability (5%) | Generated conditional and without-replacement probability decisions (`synth_probability`), exact soft labels | 10K |
| Fact check (5%) | VitaminC train split (CC BY-SA 3.0; supports / refutes / not enough info) | 40K |
| Abstention (in noul) | SQuAD 2.0 answerability (CC BY-SA 4.0) | 30K |
| NLU, multilingual, intent, soft labels, long docs | SNLI, MultiNLI, BoolQ, Banking77, CLINC150, KLUE, KorNLI, MASSIVE ko/ja, JGLUE, GoEmotions, QuALITY | ~250K |

Some sources are not used by default. `include_restricted=true` or
`ALLOW_RESTRICTED=1` opts in to them:

- **jev-distill `yuri_v3` stream** (about 444K rows). Its labels are outputs of
  TypeSafe's Jev API. TypeSafe's Master Customer Agreement (updated
  2026-09-23) §2.3(b) forbids using "any Output to perform model distillation,
  train a model to imitate the output of the Services, or develop … a similar
  or competing product". OpenRouter's terms §5.1 pass provider terms through
  to its users.
- **jev-distill `yuri_v1` stream**, whose labels come from an unidentified
  "32B teacher".
- **ANLI** (CC BY-NC), **MultiRC** (unclear terms) and **Amazon reviews**
  (Amazon's terms).

Excluded outright:

- **Data generated by commercial LLMs.** MuSR was generated with GPT-4.
- **BIG-bench.** Its canary string asks to keep it out of training corpora.
- **30 LegalBench tasks** licensed CC BY-NC.

**Generated data** (`ayaka/data/synthetic.py`, `ayaka/data/hard_synthetic.py`). The trained Small scored
0–7% on JevBench's temporal/numeric items and 26–32% on long policies.
Openly licensed human data for these is scarce, so this repository
generates it. Every label is computed, and `tests/test_synthetic.py`
recomputes them independently. The hard generators have additional independent
oracles in `tests/test_hard_synthetic.py`. They use procedural rules and sampled
facts; no public benchmark scenario text or answer rationale enters these sources.

- `temporal`: order records mixing three date formats and relative dates.
  Questions cover shipping promises, return windows (the delivery day is
  day 0), elapsed days, weekdays, lateness buckets and event order.
- `numeric`: invoices with a discount, tax after the discount, untaxed
  shipping and a budget. Distractor totals come from the usual mistakes
  (forgetting the discount, taxing shipping, and so on).
- `policy`: 11–30 section reimbursement policies. The relevant clauses are
  shuffled in among unrelated ones, alongside an exemption and clauses that
  bind only one category. Each policy comes with a claim whose record may
  leave facts out; a missing fact counts as not established. Questions ask
  whether the claim is permitted, which requirement fails first in section
  order, and whether single requirements are met.
- `long_rules`: scattered vendor aliases, bank-country overrides, dated window
  amendments, per-invoice FX and rounding, same-order ledger aggregation and
  four-tier routing. Distractor records use near-match entities and orders.
  The three questions share the prefix. Entire domain/window/scale/risk
  combinations are held out together rather than splitting their questions.
- `calendar`: month-end and leap-year expiry, fixed-offset conversion, strict
  cutoff comparison and lateness levels, including exact cutoff/60-minute/
  24-hour boundaries. Date distractors vary on both sides of the correct date
  to avoid a fixed correct position after canonical option sorting.
- `probability`: exact Bayes posteriors and without-replacement inspection
  probabilities. Targets remain soft, with balanced most-likely outcomes.
  This teaches uncertainty from the stated process instead of fabricated
  one-hot observed events.

Calibration and evaluation use the `openjev_v2` rows of jev-distill's
calibration and `test_set_30k` splits. Primitives those splits lack (score)
are topped up from reserved training groups, targeting
`calibration_per_bucket` (200) questions per (primitive, prompt length)
bucket. Groups are reserved bucket by bucket rather than drawn at random:
a flat draw once yielded a single long noul question, so long noul prompts
silently inherited the short temperature (T=0.88, which sharpens) and hard
prompts were overconfident. Natural sources are scanned before generated
ones, since the model trains on the generators' own templates and is
unrealistically accurate on them. The actual bucket counts are recorded in
`report.json`; buckets that stay sparse fall back to their primitive's
temperature.

Temperatures are fitted per primitive **and** per prompt-length bucket:
short prompts are under `long_prompt_tokens` (1024 by default), long
prompts are at or above it. The single per-primitive scalar left long, hard
prompts overconfident (JevBench hard ECE 0.26). A bucket with fewer than 20
calibration questions keeps its primitive's temperature. Checkpoints saved
with the old `[3]` temperatures still load, with each value applied to both
buckets.

Every training sample that shares a 13-gram with a JevBench public item is
dropped (`ayaka/data/decontam.py`).

Measured prompt lengths (Gemma 4 tokenizer): jev-distill mean 191 tokens (p99
602), JevBench hard mean 1,242 (max 3,892). `max_seq_len` is 4096 for training. Inference uses `serve_max_seq_len` (8192 by default; `--max-seq-len` on `ayaka.serve` and `ayaka.eval.jevbench`). Without it, longer states lose their middle, while the sealed hard tier describes 2-6K-token policies. To cover
long evidence, QuALITY articles that fit (about 1,100 questions, mean 2.8K
tokens) make up 6% of the mixture. Longer articles are skipped instead of
truncated, because cutting the middle could remove the answer's evidence.

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

modal run modal_app.py --cmd "train --model electra-small --run small-v1"            # Modal A100-80GB
modal run modal_app.py --cmd "train --model electra-large --run large-v1" --gpu h100  # Modal H100
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
template is in [`docs/JEVBENCH_REQUEST.md`](docs/JEVBENCH_REQUEST.md).

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

## Status (2026-10-04)

- **Benchmark request.** The three published models are requested in
  [JevBench issue #132](https://github.com/fstandhartinger/jevbench/issues/132), pinned to code commit
  `475bec3` and the Hugging Face revisions listed there. The models are frozen. At the time of writing, the
  live board (v1.5.6) did not list them yet.
- **TypeSafe SDK compatibility is implemented.** `ayaka.serve` adds Choice/Score `confidence`, Score
  `legend`, the `{"models": [...]}` catalog, `jev-latest` / `jev-preview` aliases and structured
  instructions/criteria. The official `typesafe-sdk` 0.7.2 was tested against a random tiny CPU export
  for Noul/Choice/Score, default-model calls, `models.list()`, 401/422/429 with retries disabled and
  `extra_body={"ayaka": {}}`; the compatibility/export tests passed 44 cases. JevBench's unchanged
  adapter and scorer accepted 231/231 public items through this server with a CPU stub
  (`python -m benchmarks.jevbench_wire --jevbench PATH_TO_CHECKOUT`); this verifies wire validity,
  not model accuracy. Optional Bearer authentication (`--api-key-env`), bounded admission
  (`--max-inflight`), overload status 529 and request ids are covered. The reserved `ayaka` request
  object is ignored in v1. This is a serving-code backport from
  [the v2 design](https://github.com/alice-noa-chan/ayaka/blob/ayaka-v2-experiments/docs/experiments/JEV_API_COMPAT_2026-10-04.md):
  published weights, inference defaults, export formats and the issue #132 code pin remain unchanged.
- **Ongoing work** happens on the experimental
  [`ayaka-v2-experiments`](https://github.com/alice-noa-chan/ayaka/tree/ayaka-v2-experiments) branch. It does not
  supersede v1, and nothing there has been measured or released yet.
