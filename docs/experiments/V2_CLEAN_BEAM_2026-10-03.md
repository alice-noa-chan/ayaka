# Clean v1 continuation on Beam — 2026-10-03

The fixed 200-step pilot completed and its delivered model passed local integrity
checks. It regressed on the prespecified dev screen. Do not promote this model or
start full training from this result. The independent test remains unopened.
These are local diagnostics, not an official JevBench score or ranking.

[Machine-readable measurements](results/v2-clean-beam-20261003.json) bind the
checkpoint, cohorts, execution, receipt, metrics, budget and cleanup evidence.

## Credit eligibility and actual execution

The initial Beam balance was $7.032868. These credits were eligible for serverless
compute. The quoted reserved A100 80GB SXM offer ($1.518/hour) was rejected because
managed-compute-eligible credit was $0 and its minimum was $25. No A100 machine
was allocated. A nominal price does not establish credit eligibility.

One available serverless RTX5090, two SDK CPU units and 32GiB RAM ran native BF16
with gradient checkpointing and pruned supervised-position CE. The conservative
9700-second compute allowance was $6.69, plus a $0.10 margin. The account cap was
$6.85, below the available balance at admission; automatic top-up and retries
were disabled. The actual final observed balance was $3.856738: **$3.176130** of
credit consumed, including CPU preparation and failed setup attempts. The billing
page displayed **$0.00 cash spend**. No credits were purchased.

GPU task `1b538580-41bc-4048-a5e7-2eaadfebfece` ran from 2026-10-02
15:01:15.011064 UTC to 16:29:57.105868 UTC: **88m42s**. Optimizer steps took
**1664.64s (27m45s)**. Summed decision latency was 2432.25s across baseline,
calibration, pilot dev and three reasoning probes; it is not total evaluation
wall time. GPU staging, integrity checks, backward/stress/IO profiling, fitting,
bootstrap statistics and result packing account for additional elapsed time.
This is a pilot duration, not a measured full-epoch or full-training duration.

## Preserved experiment

The immutable zstd-level-3 kit SHA256 is
`a25b45abf551dc239d52509c90e52e0524bfc6643ed08543234106a651d3ab5e`.
The running wrapper revision was `d4990ba`; delivery verification was subsequently
added locally. Dataset bytes, parent, losses, optimizer and fixed step count were
unchanged. The recorded operational recipe change increased the completion
allowance from 3600 to 5400 seconds. Fresh complete-schedule admission passed
before any optimizer update. This was not a time-truncated training schedule.

The pinned parent is `alice-noa-chan/ayaka-base`; the base is
`google/gemma-4-E4B-it` at `ee0ef6023621cff504d758262d4e04895a5af4a2`.
The parent fingerprint is
`26109de64d2e6417c64a7e0942ac2d18b43208f7d870ffabf7c8fb814ad606d2`.
The pilot fingerprint is
`09ae3c010c2d8ca49acac3e013f95f9f667f59547e155b578826e0b7a33b383d`.

The schedule exposed 12,800 rows (200 × 64), including 8,909 unique prepared rows
and 1,643 source lineages. It processed 5,744,712 text tokens and 898,035 teacher
trace tokens. Repeated rows are not independent examples or evidence of a full
epoch. This pilot trained text only; image/proposal quality was not tested.

## Matched dev comparison

All comparisons use the same 2672 questions and 495 underlying cases. Confidence
intervals use 2000 source-clustered replicates and seed 15. Calibrations were fitted
on the separate 1664-question calibration split, never on dev or test.

| Comparison | Parent CC | Pilot CC | Delta | Paired delta 95% interval |
|---|---:|---:|---:|---|
| Both raw, temperatures removed | 9.9262 | 2.8080 | −7.1182 | [−10.1631, −4.2568] |
| Both separately recalibrated | −6.2255 | 1.7561 | +7.9816 | [+5.3884, +10.6549] |
| Published parent vs calibrated pilot | 13.8787 | 1.7561 | −12.1226 | [−15.5162, −8.8608] |

The apparent calibrated gain is against a parent damaged by NLL temperature
fitting under Noul's fixed abstention thresholds. It does not establish a gain
against published v1, and neither raw nor calibrated comparison passes the
complete screen. Do not select this favorable row as proof of improvement.

| Raw type | Parent CC | Pilot CC | Main observations |
|---|---:|---:|---|
| Choice | 29.9042 | 28.4378 | CC declines; NLL 1.5508 → 1.4847 |
| Noul | −35.2778 | −57.5000 | Abstentions 367 → 497 of 720; NLL improves |
| Score | 35.1522 | 37.4862 | RPS 0.17450 → 0.17152; NLL slightly worsens |

Raw classification fixed/broken counts are **50/160**. Score nMAE improved/worsened
on **541/379** questions, using a 1e-6 diagnostic tolerance. English, Korean and
Japanese raw CC decline by 6.30, 6.96 and 8.25 points respectively.
The temporal/numeric family declines from −10.01 to −26.22 (−16.20 points);
the numeric family declines 0.87 points. Judge improves 3.54 points. Family
measurements diagnose this fixed dev cohort and do not prove generalization.

These measurements show that better proper losses can coexist with worse
thresholded Noul competence. Future work must validate decision thresholds and
abstention behavior on a reserved split, preserve the parent through paired
regression checks, and demonstrate actual date reasoning improvement before
funding longer training. Lower learning rate or more steps alone are unproven
remedies. No new recipe or weights were selected from the unopened test.

## Forced reasoning and delivery

All three predeclared forced-high probes used route `reasoned` and retained the
1024-token maximum. They generated 415, 89 and 89 tokens and ended at normal EOS;
there was no silent effort downgrade. Two probes answered correctly and Noul
failed. Three probes verify execution, not broad reasoning accuracy or its gain.

The result archive is 2,166,633,275 bytes, SHA256
`ba93c10d4fd5065dfb7e45ebd4231587b120b4e2696c3afbbfcd767101e5ce77`.
All 59 manifest files matched checksums. The checkpoint's 200-step completion
seal passed and all 539 head/adapter tensors were finite. Downloads, raw reports,
resume state and billing/cleanup evidence are preserved locally under
`runs/beam-v2-clean-20261002/`. The delivered checkpoint is
`received/result/recovery/pilot/checkpoint/` within that directory; use the pinned
base rather than interpreting the adapter/head as a standalone base model.

The completed serverless task left zero running containers. The unused owned
reserved pool and temporary staging volume were deleted after verified delivery.
Published v1 and unrelated account volumes were not changed. Full training,
release promotion, publication and independent-test evaluation did not occur.

## Implementation validation

Runner changes were committed in separate units after lint/format checks and
relevant tests. The integrated console-entry pytest run passed **499 tests with
one skip** in 192.79 seconds. The 24 Beam-specific tests passed, including credit
class admission, old remote SDK compatibility, checksums and completion seals.
Ruff lint and formatting checks passed. Local hardware tests do not replace the
actual GPU backward/profile, complete training and delivered-checkpoint checks.

Sources: [Beam prices](https://www.beam.cloud/pricing),
[storage and billing](https://docs.beam.cloud/v2/resources/pricing-and-billing),
[reserved pools](https://docs.beam.cloud/v2/scaling/pools). Rates were checked
2026-10-02; the actual account balance and successful/failed allocations, rather
than public prices alone, determined admission.
