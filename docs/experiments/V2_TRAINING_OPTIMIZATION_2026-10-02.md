# Ayaka v2 training optimization — 2026-10-02

The training implementation now batches supervised work, reuses frozen image
features and prepared CPU inputs, and forecasts completion of the entire fixed
step schedule before the first optimizer update. The equal-work tiny CPU
comparison improved median backward time by **19.50% (1.2423× throughput)**.
Actual pretrained H100 throughput and a four-hour complete run are **unmeasured**.
No new GPU job or pretrained optimizer update was executed.

The machine-readable measurements are
[v2-training-optimization-20261002.json](results/v2-training-optimization-20261002.json).
The previous [readiness record](V2_PRETRAINING_2026-10-01.md) and
[H100 exploration findings](V2_FINDINGS_2026-10-01.md) remain historical records.

## Work removed without dropping supervision

- Trace CE projects only labelled prediction positions, gathered across rows in
  128-token chunks. Per-question averaging, including unlabelled rows in the
  original denominator, is preserved. This avoids many small vocabulary-head
  projections without projecting every context token.
- Candidate proposals use one independently masked, padded forward per
  micro-batch. Proposal text still does not enter the teacher-partition decision
  context. An empty auxiliary loss no longer reduces the entire base embedding
  matrix just to construct zero.
- On supported shared-KV backbones, trace and proposal forwards keep only answer
  and supervised query positions above the shared layer, retaining full causal
  keys/values below it and the original option-span states. Unsupported backbones
  and checkpointed chunks retain the complete native forward.
- Native image rows are batched by output type and missing-evidence flag, up to
  four rows. Those strata preserve the former singleton RPS and missing-evidence
  loss weights. Masks, modalities, multiple images and variable patch counts
  remain independent between rows.
- A bounded 128 MiB feature cache reuses only frozen, evaluation-mode native
  vision/projector outputs. CPU pixel/position content and weight versions bind
  entries; weight/device/dtype changes invalidate them. Cached features own their
  storage. No language activation or training KV cache is reused.
- A bounded 256 MiB CPU preparation cache reuses encoded immutable source rows,
  tokenization and processor output. The sampler order is unchanged, media
  processing stays on CPU until needed, and GPU image features are cached
  separately. Loss metrics use one device-to-host transfer per step.
- Prediction omits training-only trace and proposal CE.

## Recipe compared with v1

| Setting | Previous prepared v2 | Optimized v2 | Recorded v1 Base |
|---|---:|---:|---:|
| Questions/encoded rows per optimizer step | 32 | 64 | 64 |
| Micro-batch token budget | 4,096 | 8,192 | 8,192 |
| Always checkpoint activations | Yes | No | No |
| Image batch row cap | 1 | 4 | No native-image training |
| Frozen image feature cache | None | 128 MiB | Not applicable |
| Prepared CPU row cache | None | 256 MiB | Separate mixture loader |
| Trace CE / proposal CE weight | 0.3 / 0.2 | 0.3 / 0.2 | No generative CE |

The optimized recipe restores v1's batching and initial checkpoint policy; OOM
backoff can still selectively checkpoint and reduce micro-batches. Learning
rates, LoRA targets, model revision, source splits, language mixture and loss
weights are retained. The new recipe explicitly targets 14,400 seconds for the
complete invocation. Changing rows per step changes exposures for a fixed step
count: compare matched supervision/token work rather than treating identical
step counts as identical curricula.

Local release metadata records v1 Base at 1,200 steps × 64 exposures, 11,266.1191
seconds (3h 7m 46s), and v1 Large at 860 × 64, 18,681.7238 seconds (5h 11m 22s).
Both have `stopped_early: false`. V1's mixture sampler draws with replacement;
these completed fixed schedules were not one full pass over every source row.
The corresponding local evidence is `runs/release/ayaka-base/meta.json` and
`runs/release/ayaka-large/meta.json`.

V2 adds trace-token CE, proposal CE and native image processing. Its number of
prepared rows is not a count of independent source questions. Neither the small
500-step exploration nor the CPU benchmark below establishes that the larger
v2 workload will beat v1's end-to-end training time or preserve its quality.

## Measured equal-work CPU comparison

The benchmark used a random tiny Gemma4 model, float32, one CPU thread, LoRA
dropout zero, one warm-up batch and five timed forward/backward batches. Both
paths used checkpointing off and an 8,192-token micro-batch budget. The reference
used full native trace/proposal forwards, singleton images, no feature cache and
32-token CE chunks; the optimized path enabled the new pruning, batching, cache
and 128-token chunks. It is not a replay of every historical v2 configuration.

| Measurement | Reference | Optimized |
|---|---:|---:|
| Median backward batch seconds | 2.6430272 | 2.1275418 |
| Forward chunks per batch | 10 | 5 |
| Supervised rows per batch | 24 | 24 |
| Text tokens / trace labels / proposal labels | 15,646 / 711 / 330 | 15,646 / 711 / 330 |
| Image rows per batch | 8 | 8 |

Total joint-loss absolute difference was 4.7684e-7. There were 44 frozen-feature
cache hits and four misses across warm-up plus timed batches. All trainable
weights stayed unchanged and optimizer steps were zero. This measures the local
mechanics, not pretrained H100 speed, optimizer/save time or downstream accuracy.

To reproduce without model downloads or GPU allocation:

```bash
python -m ayaka.training.benchmark_v2 \
  --out runs/my-new-cpu-speed-check.json --repeats 5
```

## New immutable preparation

Use `runs/v2-pretraining-20261002-e4b-optimized`, prepared from implementation
revision `1436654435a2a9335b1071e56e838fe25ed276bd`. Its manifest SHA-256 is
`f6fe154c5deeee0a70d82a2523cfeb99d6c03870c68b3099e3e3a57ba444bfd3`.
Default audit returned `source_code_matches: true` and zero optimizer steps.
All five JSONL files are byte-identical to the previous validated bundle: 6,080
samples, 6,560 questions and 12,320 prepared rows across the isolated splits.
The source-bound recipe was regenerated; the historical bundle was not changed.

The pinned E4B revision, Apache 2.0 policy, frozen vision, fresh LoRA and head,
parameter count below 14B and 4,096-token training context remain unchanged.
The 15,992,595,884-byte native weight file was re-audited against the new bundle:
SHA-256 `cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503`.
This used already cached weights and allocated no GPU model.

## Completing the schedule within the planned budget

`run_v2 --profile-only` first checks representative train-only gradients and
then measures one warm-up and three production backward batches, including
input preparation, synchronization and the normal OOM policy. It restores RNG,
clears gradients and verifies unchanged weights and empty optimizer state. It
never starts optimizer training. On a CUDA host, with compatible dependencies
and the pinned cache, the next measurement is:

```bash
python -m ayaka.training.run_v2 \
  --bundle runs/v2-pretraining-20261002-e4b-optimized \
  --profile-only --device cuda --max-train-seconds 900 \
  --out runs/my-new-h100-throughput-check
```

This command was **not executed on H100** here. It rejects `--steps`; it records
`preflight.json` and `throughput.json` before returning. No main-training step
count is silently selected.

Actual `--execute` measures throughput before every optimizer run and writes
`completion_plan.json`. Admission estimates all planned steps using:

```text
(maximum observed backward seconds + 1 second per-step optimizer allowance)
    × planned steps × 1.25 safety factor + 120 seconds saving reserve
```

Loading and preflight time are subtracted from the smaller of the explicit job
budget and the recipe's four-hour target. A plan that does not fit is rejected
before any optimizer update. An admitted plan completes the fixed step/LR
schedule; its loop has no time-based partial-curriculum endpoint. Final metadata
and a marker written after saving distinguish full completion.

The short timing sample and optimizer/save allowances are estimates. Heavy later
batches, cache eviction, OOM fallback, repeated checkpoint writes, disk or GPU
contention can change throughput. The outer hard timer remains an emergency
spending cap, so this is not a wall-clock guarantee. Actual H100 timing and the
chosen complete curriculum must be assessed before claiming v1-equivalent time.

## Validation and remaining quality work

Ruff lint and format checks passed for **131 Python files**. The full CPU suite
passed **421 tests**, with one existing Beam-SDK skip, in 53.83 seconds. Added
checks cover loss/gradient parity, both native image architectures, cache on/off,
checkpoint on/off, multi-image padding, cache invalidation and bounds, deterministic
preparation, proposal isolation, unchanged zero-update weights/RNG, OOM cleanup,
budget rejection before updates, and complete fixed-schedule saving. Existing
forced `on + high`, inference isolation and v1 regression tests remain passing.
Float32 batching changes reduction order; parity checks allow small round-off
differences, not bitwise training identity with dropout.

The curriculum is still authored mechanics data. Natural EN/KO/JA preservation,
image accuracy, arbitrary candidate coverage, scoped calibration/router gains
and independent JevBench results still need trained measurements. This work
does not promote the previous regressing adapter, change published v1 defaults,
or establish a leaderboard rank. New GPU seconds and pretrained optimizer
updates for this optimization work are both zero.
