# Saved v2 continuation audit — 2026-10-04

This is a CPU reanalysis of existing matched dev probabilities. It performs no
model forwards, optimization, calibration fitting or new GPU work. It does not
open the independent test split or establish a causal training mechanism.

## Cohort and provenance

The completed Beam pilot's parent is **ayaka-base**, not ayaka-large. The pilot
finished its fixed 200-step schedule; this was not a full training epoch.
Both reports contain the same 2,672 reasoning-off questions in 495 underlying
cases. The comparison requires the complete cohort, not its intersection.

| Input | SHA-256 |
| --- | --- |
| Parent checkpoint fingerprint | `26109de64d2e6417c64a7e0942ac2d18b43208f7d870ffabf7c8fb814ad606d2` |
| Pilot checkpoint fingerprint | `09ae3c010c2d8ca49acac3e013f95f9f667f59547e155b578826e0b7a33b383d` |
| `parent-raw-dev.json` | `812238b807f690d3905ab45c342175b66b6fd8bf0bfa87b1e409c4c4c0d0349b` |
| `pilot-raw-dev.json` | `398496d9000123eba7459d6f80c7952b2904dbdc205b87550b3dc9f9907e1154` |
| `pilot/workload.json` | `7c55577f2010a1146561962505d48e9e12f08b2a2bca3eb38bc41aa234048c0f` |
| `pilot/history.json` | `9279c12a7d83c4b839fcbfda4a8d2ce487c6b3f4695dc5e5e6a7cdfa0cf30885` |

These result inputs are under
`runs/beam-v2-clean-20261002/received/result/recovery/`.
The new local report is `runs/continuation-audit-20261004/raw-dev.json`.
Local run files remain ignored; this document records their identities and findings.

The tool validates report/row checkpoint binding, split, route, generation counters,
IDs, case membership, targets and available metadata. It recomputes typed metrics
from probabilities rather than trusting cached summaries. Source reports omit
full prompts and candidate strings, so matching metadata cannot independently
prove that those full input strings were identical.

## Observed changes

These are local equal-type chance-corrected competence points, not accuracy
percentages or an official sealed-inclusive JevBench composite.

| Metric | Parent | Pilot | Change |
| --- | ---: | ---: | ---: |
| Equal-type competence | 9.9262 | 2.8080 | -7.1182 |
| Choice competence | 29.9042 | 28.4378 | -1.4664 |
| Noul competence | -35.2778 | -57.5000 | -22.2222 |
| Score competence | 35.1522 | 37.4862 | +2.3340 |
| Noul NLL | 0.90035 | 0.83528 | -0.06507 |
| Noul Brier | 0.26115 | 0.22268 | -0.03847 |
| Noul ECE | 0.27068 | 0.16128 | -0.10940 |
| Noul abstentions | 367 | 497 | +130 |

Choice/Noul thresholded correctness improved on 50 rows and worsened on 160.
Score nMAE improved on 541 rows and worsened on 379, with tolerance `1e-6`.
The mean per-row NLL gain across all types was `+0.03719` despite competence
regression. These improvement counts use different definitions for classification
and Score and must not be combined into a single accuracy count.

Case-level paired bootstrap: 2,000 replicates, seed 15, competence delta 95%
interval `[-10.0934, -4.3287]`. Sorting IDs changes deterministic bootstrap draws
relative to the older report's input order, whose interval was
`[-10.1631, -4.2568]`; the new interval is not an exact reproduction of that one.

## Noul decomposition

On the 720 Noul rows:

```text
thresholded credit = argmax credit - credit lost to abstention
delta thresholded credit = +0.0222222 - 0.1333333 = -0.1111111
```

Argmax credit improved by **2.2222 percentage points**, but credit lost to
abstention increased by **13.3333 percentage points**. Thresholded credit fell by
**11.1111 percentage points**; Noul's chance correction doubles that to the
observed 22.2222-point competence loss.

| Parent → pilot state | Rows |
| --- | ---: |
| Abstain → abstain | 350 |
| Abstain → yes | 17 |
| No → abstain | 147 |
| No → no | 74 |
| Yes → yes | 132 |

The regression on this cohort cannot be described simply as more wrong argmax
directions. Better NLL, Brier and ECE did not imply better thresholded decisions.
This does not prove that suppressing abstentions, sharpening temperature or
training longer would improve an independent cohort. Such policies need their
own calibration/dev comparison and final emitted-probability losses.

## Unresolved questions

- Data shift, trace token CE interference and readout training remain confounded.
- Parent-distribution replay has not been tested on a matched training schedule.
- Improvement on synthetic dev alone does not establish natural-document transfer.
- First/last loss windows cover different batches and cannot alone establish convergence.
- The new diagnostic tool has implementation tests, but peer review is still pending.

## Reproduce without inference

```powershell
.venv/Scripts/python.exe -m ayaka.eval.continuation_audit `
  --parent runs/beam-v2-clean-20261002/received/result/recovery/parent-raw-dev.json `
  --candidate runs/beam-v2-clean-20261002/received/result/recovery/pilot-raw-dev.json `
  --workload runs/beam-v2-clean-20261002/received/result/recovery/pilot/workload.json `
  --history runs/beam-v2-clean-20261002/received/result/recovery/pilot/history.json `
  --out runs/continuation-audit-20261004/reproduction.json
```

Choose a fresh output filename; the CLI refuses overwriting an existing receipt.
Related implementation/recovery tests: **14 passed**; lint and formatting checks
passed. The earlier integrated snapshot had **530 passed, 1 skipped** before
the latest fingerprint check and concurrent Swift additions; that result does
not validate subsequent Swift code.
