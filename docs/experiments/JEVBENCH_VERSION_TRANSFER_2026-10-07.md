# Paired JevBench version transfer, 2026-10-07

Benchmark-version differences do not prevent an approximate comparison.
Models measured in both releases can provide a transfer between score scales.
That transfer is conditional on an input measured on the old benchmark scale;
it does not establish a transfer from Ayaka's separate development corpus.

## Public paired measurements

The sources are the benchmark owner's versioned aggregate APIs:
[v1.5.7](https://www.benchmarkheaven.com/api/jevbench/v1.5.7) and
[v1.6.1](https://www.benchmarkheaven.com/api/jevbench/v1.6.1), downloaded on
2026-10-07. The old release contains 1,624 decisions, while the current
release contains 1,500. The latter also applies the O1S scoring amendment.
The downloaded bytes and their SHA-256 values are retained locally; the
small derived report records the hashes and matching criteria:
[jevbench-version-transfer-20261007.json](results/jevbench-version-transfer-20261007.json).

| Same public system identifier | Old Intelligence | Current Intelligence | Difference |
|---|---:|---:|---:|
| Cygnet | 71.092 | 54.839 | -16.253 |
| Winnow-12B Q8 | 74.435 | 59.534 | -14.901 |
| Jev-Omni | 70.491 | 55.530 | -14.960 |

All three use the Gemma 4 12B family. Their probability-quality axes move
less uniformly: Cygnet Calibration changes 87.007 to 86.968, Winnow changes
84.070 to 82.958, and Jev-Omni changes 82.598 to 87.045. Intelligence and
Calibration therefore need separate transfers rather than one universal
subtraction from a composite.

There are 81 matching complete system identifiers with no disagreement in
supplied repository URLs. For a comparison around the intended quality
range, 38 anchors additionally require self-hosted status in both releases,
old Intelligence at least 50, and either a stable nonempty adapter identifier
or a supplied repository. No reported rank is used as a regression target.

System keys and repository links cannot guarantee that every weight or
serving setting stayed identical: many older rows lack exact model pins.
Some peers also share backbones and recipes. The following is a descriptive
transfer across published systems, with those limitations, rather than a
causal estimate of the scorer change alone.

## Empirical Intelligence transfer

Ordinary least squares on the 38 anchors gives:

`current Intelligence ≈ 1.16217 × old Intelligence − 26.50887`

The old Intelligence range is 50.472 to 84.206. The fit has R² 0.838 and
leave-one-system-out mean absolute error of 3.674 points. The 5th and 95th
percentiles of those held-out residuals are -6.447 and +8.669 points.
These are descriptive peer residuals; they are not a confidence interval
for an Ayaka prediction and do not cover the missing local-to-official
cohort transfer. The anchor median change is -16.419 points.

| Hypothetical old official Intelligence | Current estimate | Peer residual range |
|---|---:|---:|
| 60 | 43.222 | 36.774–51.891 |
| 70 | 54.843 | 48.396–63.513 |
| 80 | 66.465 | 60.018–75.134 |

These are examples of the transfer, not Ayaka measurements. Inputs outside
the fitted range need explicit extrapolation treatment. A separate
Calibration transfer and its residuals are included in the JSON; zero-axis
rows, which use the unsupported/label-only treatment, are excluded from
that probability-quality transfer.

The current open-weights board leads with Capability, the arithmetic mean
of Intelligence and Calibration for systems inside its eligibility caps.
The four-axis Composite is secondary. An estimate of Intelligence alone
does not determine the current headline rank. The official
[board](https://www.benchmarkheaven.com/jev-models) and
[model guide](https://www.benchmarkheaven.com/jev-models/how-to-choose)
describe those axes and conditions.

## What is still needed for v2 Large

The existing Ayaka scores, 57.929 off and 77.874 on, come from 376
development questions comprising 99 underlying cases. They use an older
competence formula, but the corpus contains procedural, legal, judge,
commonsense and intent cases rather than JevBench items. Neither aggregate
API lists an official Ayaka v1 or v2 result.

Applying the transfer directly to these two numbers would silently assume
that this development corpus has the same difficulty, type/tier behavior
and score scale as official v1.5.7. That assumption has not been measured.
Accordingly, the report keeps Ayaka's old official score, transferred current
score and rank unset instead of presenting that substitution as a result.

The practical evaluation path is:

1. Run v2 off/on and frozen reference recipes on the same versioned old
   public JevBench questions, with matching candidates, budgets and scoring.
2. Use references that have official old aggregates to estimate the public
   to official difference, keeping public-only results and inferred full-pool
   results explicitly separate. Sealed old questions are not locally available.
3. Transfer the resulting old-scale Intelligence and Calibration separately,
   propagate both sources of error, and report the current Capability range.
4. Compare that range against the appropriate current board, checking its
   cost/latency eligibility separately from quality. Show an estimated rank
   range alongside the old public measurements, not an official rank claim.

This requires evaluation with the existing weights; it does not require
retraining. This analysis itself made no model calls, updated no weights,
created no GPU instances, and sent no benchmark submission or external message.
