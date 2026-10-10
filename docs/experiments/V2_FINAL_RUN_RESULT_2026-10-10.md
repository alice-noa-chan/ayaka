# Final v2 run: result (2026-10-10)

This reports the run predeclared in `V2_FINAL_RUN_PREDECLARATION_2026-10-10.md`, including its two amendments. Aggregates only; no dataset rows are committed. The machine-readable reports are in `docs/experiments/results/v2-final-run-20261010/`.

**Decision under the predeclared rule: not adopted.** The frozen v2 policy beats v1 by +10.8 equal-type CC on the unseen final_test cohort (95% CI [+7.2, +14.2]) and passes the Speed axis. It fails the zero-tolerance per-type checks for Score, and the clustered HelpSteer2 subgroup check.

The project owner then decided to publish v2 on Hugging Face as a separate model, with this result stated in its model card. v1 remains `alice-noa-chan/ayaka-large`. That release decision is outside the predeclared rule and does not change this record.

## Rental

| Item | Value |
|---|---|
| Instance | vast.ai 55160604, RTX PRO 6000 S, $1.4538/h |
| Source commit | `5e350ed` (learning-rate amendment) |
| Rental time | 7.57 h, all stages from setup to recovery |
| Compute cost | about $11.0 (estimate; the invoice also includes storage and bandwidth) |
| Earlier attempts | about $5: one stopped at admission, one diverged at lr 1e-4 |
| Recovery | 75 indexed files verified locally; rental destroyed; no other instances |
| Private test split | not opened |

## Training and checkpoint selection

LoRA rank 64 on corpus plan 3 (37,040 rows per epoch, 2 epochs, 2,315 scheduled steps), LoRA lr 3e-5, head lr 1.5e-4.

Selection read the bundle's own router_train split (692 questions) every 250 steps:

| Step | Equal-type NLL | Choice NLL | Noul NLL | Score NLL | Equal-type CC |
|---|---|---|---|---|---|
| 250 | 0.621 | 0.662 | 0.233 | 0.967 | 63.6 |
| 500 | 0.567 | 0.510 | 0.245 | 0.945 | 67.5 |
| 750 | 0.521 | 0.447 | 0.208 | 0.908 | 72.2 |
| 1000 | 0.526 | 0.445 | 0.250 | 0.883 | 74.0 |
| **1250** | **0.495** | 0.456 | **0.196** | **0.832** | **75.4** |
| 1500 | 0.519 | 0.490 | 0.211 | 0.857 | 74.3 |
| 1750 | 0.503 | | | | |
| 2000 | 0.523 | | | | |

- Three reads without improvement stopped training at step 2000. The exported adapter is the step-1250 state.
- The lower learning rate removed the divergence: lr 1e-4 had reached CC −23.7 at step 500.

## Tuning (fixed before final_test was read)

The tuning used the calibration (334 questions), router_train (334) and dev (668) cohorts.

| Policy | router_train CC | dev CC | dev Speed axis |
|---|---|---|---|
| direct | 76.87 | 74.14 | 84.8 |
| router | 84.59 | 80.42 | 68.2 |
| noul_always | 81.87 | 78.10 | 69.5 |
| **noul_always+router** (chosen) | **86.26** | **81.25** | 59.6 |

- The router was promoted on dev at lambda 0.0005 (NLL gain 95% CI [+0.016, +0.136]).
- `frozen.json` (SHA-256 `e896a5e4…`) was sealed before the first final_test reasoned read. The frozen policy then reasoned on 566 of the 836 final_test questions.
- The dev gate over v1, under the older zero-tolerance subgroup rule, failed only `source:commonsense_qa/choice/cc_not_worse`. HelpSteer2 Score passed on dev.

## Final gate (final_test, 836 questions, 431 cases, read once)

| System | Equal-type CC | Choice CC / NLL | Noul CC / NLL / abstentions | Score CC / NLL / Brier | Speed axis |
|---|---|---|---|---|---|
| v1 (`ayaka-large`, frozen reasoning policy) | 70.19 | 75.40 / 0.507 | 67.19 / 0.197 / 50 | **67.99 / 0.792 / 0.448** | 82.4 |
| v2 direct, calibrated | 73.72 | 78.98 / 0.437 | 75.00 / 0.176 / 35 | 67.19 / 0.804 / 0.463 | 85.9 |
| **v2 frozen policy** | **81.01** | **82.57 / 0.376** | **93.23 / 0.118 / 2** | 67.22 / 0.812 / 0.464 | 58.9 |

| Check | Result |
|---|---|
| CC gain ≥ 5 | pass (+10.82) |
| Classification credit gain ≥ 5 | pass |
| Paired CC 95% CI lower bound > 0 | pass ([+7.18, +14.20]) |
| ≥ 200 independent cases | pass (431) |
| Choice and Noul: CC, NLL, Brier not worse | pass |
| Score: CC, NLL, Brier, RPS, nMAE not worse (zero tolerance) | **fail** |
| Per-source and per-language CC, clustered non-inferiority | **fail** for `helpsteer2/score` only |
| Speed axis ≥ 50 | pass (58.9) |

## Post-hoc analysis of the Score failure

This analysis was computed after the decision from the sealed reads. It fits nothing and changes nothing; it explains the failure for the next model. File: `posthoc-score-analysis.json`.

| Score subgroup | Rows (cases) | v1 CC | v2 direct CC | v2 policy CC | Policy reasoned on | Policy − v1 CC, clustered 95% CI |
|---|---|---|---|---|---|---|
| HelpSteer2 | 160 (32 responses) | 66.63 | 65.07 | 62.32 | 56 rows | [−8.56, −0.10] |
| Authored verified | 32 (27) | 74.65 | 77.56 | 91.21 | 19 rows | [+4.47, +33.52] |
| All Score | 192 | 67.99 | — | 67.22 | 75 rows | [−5.11, +3.88] |

- **The Score type as a whole is within noise.** Its −0.8 CC is well inside a 95% interval that spans zero, and NLL, Brier, RPS and nMAE move by similarly small amounts. The per-type checks use zero tolerance, not intervals, so a noise-level difference fails them. The clustered rule applied only to subgroups.
- **HelpSteer2 is a real but borderline regression.** The upper bound of its interval is −0.10. Two causes add up:
  1. v2 direct is 1.6 CC below v1 on HelpSteer2. v1 trained on about 101k HelpSteer2 questions, v2 on 15,000.
  2. The router sends 56 of the 160 HelpSteer2 rows to the reasoned path, which costs another 2.8 CC. On final_test the same routing helped authored Score items (+13.6 over direct) and hurt HelpSteer2. The router does not see the source, so it cannot treat the two differently.
- **The baselines were the same.**
  - Both models read the same 836 questions, with the same scoring and protocol.
  - final_test draws only from validation/test splits, with train-text overlap excluded, so v1's HelpSteer2 training does not inflate its score.
  - v1 answered all 192 Score questions directly.

  The difference is in v2, not in the reference point.

## Lessons for the next model

- Score needs HelpSteer2 data on the scale v1 used, or a Score-specific route rule. The router should not send HelpSteer2-like questions to reasoning without per-source evidence.
- Per-type checks at zero tolerance on 192-row types sit at the statistical resolution of the cohort. A future predeclaration should apply an interval rule at the type level too, or enlarge the Score cohort.
- The recipe defaults this run validated (corpus plan 3 with StrategyQA, rank 64, lr 3e-5 / 1.5e-4, checkpoint selection, token-rate forecast) are now the training code's defaults.
