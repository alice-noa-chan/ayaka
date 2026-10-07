# Ayaka v2 preparation with natural rehearsal and measured costs — 2026-10-02

The current immutable bundle is
`runs/v2-pretraining-20261002-ready`, prepared from implementation
`41194b4`. Manifest SHA-256:
`7b11be347d7a319663102b26aa6ec91de5947f23ae9efc6d6ed3aaabad9cad7e`.
The pinned native E4B weights, parameter limit, frozen vision, fresh LoRA/head
initialization, direct/trace/proposal losses and forced reasoning controls remain
as described in the [earlier optimization report](V2_TRAINING_OPTIMIZATION_2026-10-02.md).
No pretrained optimizer update accompanies this preparation.

## Natural rehearsal rather than a mechanics-only training set

Preparation can now add pinned cached original-train human-labelled sources:

| Source | Train samples added | Each reserved split | Targets |
|---|---:|---:|---|
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | 256 | 64 | Five human-rated Score decisions per response |
| [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa) | 256 | 64 | Human-authored five-option Choice |
| [MASSIVE Korean](https://huggingface.co/datasets/AmazonScience/massive) | 512 | 64 | Human-labelled 60-intent Choice |
| MASSIVE Japanese | 512 | 64 | Human-labelled 60-intent Choice |

Source revisions and exact file hashes are recorded in `model_preflight.json`.
These sources publish CC BY 4.0 terms; attribution stays with the prepared data
and result record. Human ratings of model responses are decision supervision;
no commercial teacher rationale/output imitation target is added.

All responses to the same HelpSteer prompt share a source group. MASSIVE
translations/localizations share their original SLURP parent ID across Korean
and Japanese. Deterministic group hashing separates train, router training,
dev, calibration and test. Natural formats and label ontologies are shared across
splits, so this is source-group isolation, not a claim of held-out templates.
Authored mechanics retain their stricter template/rule/document isolation.

The 128 previous evaluation-only examples remain reserved; matching evidence,
prompts and known source lineages, plus public JevBench overlap, are removed
before training selection. This audit removed 64 reserved/public overlaps and
11,040 duplicate evidence/prompt siblings from the source scan. Whole samples
that exceed the real tokenized context are rejected; evidence is never truncated
to fit. The selected samples fit without overflow removal.

| Prepared split | Samples | Questions | Direct/trace/proposal rows | Longest row |
|---|---:|---:|---:|---:|
| train | 2,752 | 3,872 | 5,024 | 2,574 tokens |
| router_train | 1,472 | 1,824 | 2,976 | 1,624 |
| dev | 1,472 | 1,824 | 2,976 | 1,620 |
| calibration | 1,472 | 1,824 | 2,976 | 1,342 |
| test | 1,472 | 1,824 | 2,976 | 1,738 |

The native processor/tokenizer checked every row. All five source/content
isolation audits pass. Model preflight remains meta-only; pinned native weight
bytes were separately re-audited on CPU.

## Complete workload fixed before learning

CPU-only planning shares the exact sampler with production, including pending
multi-question/direct-trace rows:

```bash
python -m ayaka.training.run_v2 \
  --bundle runs/v2-pretraining-20261002-ready \
  --plan-only --planned-steps 1200 --out runs/my-new-workload.json
```

The 1,200-step reference at 64 rows per step yields 76,800 row exposures, visits
all 5,024 prepared train rows and covers 1,780 distinct source lineages.
Natural/authored exposures are 41,961/34,839. Routes are 55,458 direct, 10,745
text trace, 4,956 direct image, 4,956 image trace and 685 proposal-supervised.
The raw row-length sum is 31,271,215 token units, with 294,602 trace labels and
32,450 proposal labels. Padding, prefix sharing and secondary proposal forwards
mean this token sum is not a FLOP count or independently measured throughput.

The language mixture weights refer to source draws. Different row counts per
source produce 55,971 English / 10,347 Korean / 10,482 Japanese row exposures.
Do not label this as an exact 60/20/20 row mix. The schedule hash is
`0854216412c77fad34cdacf36c46eff46c9a60d308a388edd8284e2ebc220105`.
This is a reference complete schedule, not one source epoch and not an implicit
authorization to execute training.

## Costs measured before the first live optimizer step

The zero-update profiler now checks six ordinary production batches and the
largest rows in every language/type/image/trace/proposal/flag stratum. Stress
image batches start with empty frozen-feature caches. If OOM changes the
checkpoint/micro-batch policy, ordinary batches are remeasured. It then profiles
20 evenly spaced steps across the complete deterministic schedule and the
scheduled maxima of six inventory cost proxies: total row length, maximum row
length, trace labels, proposal labels, image row count and summed token work.
Duplicates are measured once. These are actual mixed batches with exact source
and row indices, not 64 artificially longest homogeneous examples. Every selected
image batch starts cold; policy changes require remeasurement under the final
policy. Every sampled backward checks finite gradients as well as finite losses. CUDA profiling retains
equivalent AdamW moment-storage bytes, exposing memory pressure that would
otherwise appear only after the first live optimizer step.

Clipping and AdamW timing use disposable tensors with the actual trainable
shapes, dtypes and optimizer options. Live model weights, RNG, live optimizer
state and scheduled step count stay unchanged. An actual adapter/head plus
populated disposable optimizer-state save measures filesystem cost, including
fsync. The IO probe is explicitly untrained and has no completion marker.

The forecast counts every progress write plus the final checkpoint: 14 writes
for 1,200 steps with interval 100. It applies a 25% time margin and 10% disk
margin. The forecast uses the maximum of the selected actual scheduled batches.
Homogeneous cold stress remains an independent memory check and its all-stress
time estimate is retained explicitly. Cost proxies and sampled maxima are not
mathematical upper bounds. AdamW's measured initial state allocation is counted
once, followed by the maximum of the measured warm optimizer steps. It is never
charged as a cold allocation on every step. There is no fixed one-second optimizer or two-minute total-save
assumption in native v2 admission. A complete plan is rejected before updates
if time or disk capacity is insufficient. Hardware contention remains variable;
measured estimates cannot guarantee a wall-clock deadline.

```bash
python -m ayaka.training.run_v2 \
  --bundle runs/v2-pretraining-20261002-ready \
  --profile-only --planned-steps 1200 --device cuda \
  --max-train-seconds 780 --out runs/my-new-zero-update-profile
```

The short profiling cap is separate from the recipe's four-hour full-training
forecast horizon. Actual optimizer execution still requires explicit `--execute`
and `--steps`; no main training was started here. The bounded remote entry point
is `scripts/modal/preflight_v2.py`: CPU bundle/weight verification precedes GPU allocation;
`H100!` prevents automatic GPU substitution. It reserves 1,200 conservative GPU
seconds inside the original cumulative eight-hour exploration ceiling, has no
training mode or automatic retry, and saves partial reports with a failing exit
if profiling does not complete. The original budget ledger and dated findings
are preserved; profile reservations have their own cumulative ledger.

## Actual native H100 measurement and full-training estimate

The completed [H100 profile](https://modal.com/apps/gaon12/main/ap-glUx5aI9RpTtTLml6BcOxF)
used `NVIDIA H100 80GB HBM3`, the pinned native E4B weights, frozen vision and
fresh trainable parameters. It covered 73 cold largest-row strata and 25 actual
scheduled batches, including step 1112 with the maximum total row length
(45,458 token units). All losses and produced gradients were finite, weights
were unchanged and live optimizer updates were zero.

| Measurement | Result |
|---|---:|
| Actual scheduled batch median / maximum | 6.4898 / 9.0703 seconds |
| Artificial homogeneous stress maximum | 19.4077 seconds |
| First disposable AdamW step / maximum warm step | 0.07469 / 0.005414 seconds |
| One populated-state checkpoint write | 1.2588 seconds, 467,432,314 bytes |
| Effective ordinary / checkpointed micro-token budget | 4,096 / 8,192 |
| Selective activation checkpoint threshold | 1,024 tokens |
| AdamW moment-storage reservation during backward | 303,200,304 bytes |
| Complete 1,200-step plan, median basis plus 25% margin and setup | 10,473.14 seconds: about **2h 55m** |
| Complete plan, every step at sampled maximum plus 25% margin and setup | 14,343.79 seconds: about **3h 59m** |

Both estimates preserve all 76,800 exposures and include model setup, the
zero-update checks/profile, all optimizer steps and 14 checkpoint writes. Disk
reservation is about 7.20 GB against the observed 409.60 GB free. The final
completion gate is `fits: true`; it is an estimate, not an observed complete
training run or a four-hour deadline guarantee. Initial Python/container startup,
GPU queue time and separately CPU-prepared downloads are outside the runner's
timer. Post-training calibration, router selection and independent evaluation
time are also separate and have not been measured here.

The H100 sample was taken from immutable bundle
`runs/v2-pretraining-20261002-scheduled-ready` at `a750cf7`, manifest
`c8cc47d7310b14882e1ff01287dabf462400f3073e514dea9d5883bddc272627`.
Its original completion report remains unchanged and reports a 4h 00m 48s
estimate: it repeated AdamW's initial allocation on every step. The final
`verified_completion_plan.json` is derived on CPU with corrected accounting.
All model/config/data file hashes are identical between the measured and final
bundles. Only `training/throughput.py` differs among package sources; its complete
AST excluding `completion_plan` is identical. Thus no native forward/backward,
sampling, preparation, IO probe or memory-policy code was changed, and no third
GPU job was needed to correct the arithmetic.

The v1 base metadata records 1,200 × 64 exposures and `train_sec: 11266.1191`
(3h 07m 46s), with `stopped_early: false`, using the same pinned backbone. This
is a useful time reference, not an equal-data measured speedup: v2 adds different
supervision, images and proposals, and its full-run time is still forecast.

The two completed zero-update profile functions took 478.99 and 717.59 seconds,
about 19m 57s in total. Their apps stopped and the final container inventory was
empty. The remote prior conservative ledger bound plus two 1,200-second profile
reservations is 24,649.85 seconds (6h 50m 50s), below the original eight-hour
exploration ceiling. Reservations are not an invoice; the older local audit
remains unchanged. No main-training budget was consumed.

Exact measurements, provenance, profile hashes, native-code equivalence audit
and the corrected plan are saved in
[v2-training-ready-20261002.json](results/v2-training-ready-20261002.json).

## Validation and limits requiring trained measurements

The full CPU suite passes 437 tests with one existing Beam-SDK skip in 91.603
seconds. Ruff passes
for the package, tests, scripts and bounded Modal profiler (136 Python files).
Added checks cover deterministic workload accounting, natural source-group
isolation, pinned human-label provenance, old evaluation exclusion, whole-row
overflow rejection, cold stress, zero live updates, actual IO, checkpoint count,
disk admission, finite-gradient failures, exact full-plan batch selection and
full fixed-schedule completion.

Natural rehearsal addresses the missing data path; it does not demonstrate that
the prior Korean/Japanese regressions are solved. Multilingual preservation,
image recognition, calibrated arbitrary generated candidate coverage, router
benefit and JevBench rank require measurements after training. Audio/video/PDF
and generated Score/Noul remain outside the current serving interface. Free-form
candidate probability mass remains conditional on its proposed partition, with
unverified coverage reported explicitly rather than inferred from normalization.
