# v2 base and small run: result (2026-10-11)

This reports the run predeclared in `V2_FAMILY_RUN_PREDECLARATION_2026-10-11.md`. It contains aggregates only; no dataset rows are committed. The machine-readable reports are in `docs/experiments/results/v2-family-run-20261011/`.

**Decision under the predeclared rule:**

- **base: adopted.** The frozen v2 base policy beats v1 base by +15.1 equal-type CC on the unseen final_test cohort (95% CI [+11.1, +19.0]). It passes every per-type and subgroup check, and its Speed axis is 64.0.
- **small: not adopted.** The frozen v2 small policy beats v1 small by +8.5 equal-type CC (95% CI [+3.7, +13.0]). It fails the zero-tolerance Score checks: Score NLL 0.919 against v1's 0.898, and Score Brier 0.499 against 0.494.

The project owner then decided to publish both models on Hugging Face:

- base as `alice-noa-chan/ayaka-v2-base`;
- small as `alice-noa-chan/ayaka-v2-small`, with its failure stated in the model card.

The v1 models are unchanged. Publishing small is outside the predeclared rule and does not change this record.

## Rental

| Item | Value |
|---|---|
| Instance | vast.ai 55302429, one RTX PRO 6000 S, $1.4613/h |
| Source commit | `60e92d5` (predeclaration), code of `a09127f` |
| Rental time | 5.47 h, from setup to recovery, both sizes |
| Compute cost | about $8.0 (an estimate; the invoice also includes storage and bandwidth) |
| Recovery | 143 indexed files verified locally; rental destroyed; no other instances |
| Private test split | not opened |

## Training and checkpoint selection

Both sizes trained at the same time on one GPU. Each phase took about 2.0 h, which includes profiling and the selection reads. Selection read each bundle's router_train split every 250 steps:

| Step | base loss | base CC | small loss | small CC |
|---|---|---|---|---|
| 250 | 0.635 | 58.6 | 0.700 | 49.3 |
| 500 | 0.627 | 62.0 | 0.630 | 57.0 |
| 750 | 0.636 | 65.9 | 0.610 | 61.4 |
| 1000 | **0.598** | 67.5 | 0.610 | 63.8 |
| 1250 | 0.620 | 70.7 | **0.604** | 68.2 |
| 1500 | 0.601 | 69.4 | 0.618 | 66.0 |
| 1750 | 0.603 | 70.5 | 0.606 | 67.2 |
| 2000 | — | — | 0.645 | 68.8 |

- After three reads without improvement, base stopped at step 1750 and small at step 2000.
- The exported adapters are the step-1000 state for base and the step-1250 state for small.

## Tuning (fixed before final_test was read)

| Policy | base router_train CC | base dev CC | base dev Speed | small router_train CC | small dev CC | small dev Speed |
|---|---|---|---|---|---|---|
| direct | 66.84 | 66.11 | 87.0 | 60.50 | 59.03 | 87.1 |
| router | 68.71 | 68.25 | 73.1 | 69.39 | 67.15 | 73.8 |
| noul_always | 75.17 | 73.61 | 70.0 | 69.66 | 65.49 | 70.5 |
| **noul_always+router** (chosen) | **77.04** | **75.33** | 63.1 | **69.81** | 65.49 | 70.5 |

- **Policy choice:** both sizes chose `noul_always+router` on router_train, as the predeclared rule requires.
  - For small this was a 0.15 CC margin over `router`.
  - `router` was better on dev, but dev does not choose the policy.
- **Router promotion:** both routers were promoted on dev at lambda 0.001.
  - base NLL gain 95% CI [+0.005, +0.108].
  - small NLL gain 95% CI [+0.015, +0.068].
- **Routing on final_test:**
  - small's router sent no Choice or Score question to reasoning, so small reasoned on Noul only (384 questions).
  - base reasoned on 481 questions: 384 Noul, 67 Choice and 30 Score.
- **Sealing:** each `frozen.json` was sealed before that size's first final_test read.
  - base SHA-256 `13621c1e…`.
  - small SHA-256 `650a0e7d…`.

### Reasoned reads that fell back to a single pass

Gemma 4 E2B and E4B keep 512 positions in their sliding-window layers. On long HelpSteer2 questions the reasoned read could not roll its cache back, so the question fell back to its single-pass answer. The comparison protocol accepts such fallback rows.

| Read | base | small |
|---|---|---|
| calibration (334) | 1 | 3 |
| router_train (334) | 1 | 3 |
| dev (668) | 2 | 5 |
| final_test (routed) | 0 | 0 |

No final_test read fell back. The fix (`0a97b5e`) now reads the complete final chat when the cache cannot roll back. It landed after this run, which used the predeclared code.

## Final gate (final_test, 836 questions, 431 cases, read once per size)

Each size's final_test reads ran alone on the GPU, one size at a time.

### base

| System | Equal-type CC | Choice CC / NLL | Noul CC / NLL / abstentions | Score CC / NLL / Brier | Speed axis |
|---|---|---|---|---|---|
| v1 (`ayaka-base`, frozen reasoning policy) | 62.98 | 67.19 / 0.673 | 60.42 / 0.281 / 59 | 61.33 / 0.949 / 0.514 | 86.1 |
| v2 direct, calibrated | 70.41 | 73.35 / 0.559 | 70.83 / 0.195 / 39 | 67.05 / 0.852 / 0.464 | 88.0 |
| **v2 frozen policy** | **78.12** | **75.91 / 0.502** | **89.58 / 0.153 / 7** | **68.86 / 0.787 / 0.436** | 64.0 |

| Check | Result |
|---|---|
| CC gain ≥ 5 | pass (+15.14) |
| Paired CC 95% CI lower bound > 0 | pass ([+11.14, +18.96]) |
| ≥ 200 independent cases | pass (431) |
| Per-type checks, zero tolerance | pass |
| Per-source and per-language CC, clustered non-inferiority | pass |
| Speed axis ≥ 50 | pass (64.0) |

### small

| System | Equal-type CC | Choice CC / NLL | Noul CC / NLL / abstentions | Score CC / NLL / Brier | Speed axis |
|---|---|---|---|---|---|
| v1 (`ayaka-small`, frozen reasoning policy) | 54.76 | 68.22 / 0.637 | 35.42 / 0.410 / 105 | 60.64 / **0.898 / 0.494** | 77.4 |
| v2 direct, calibrated | 60.53 | 73.35 / 0.550 | 46.35 / 0.263 / 88 | 61.89 / 0.919 / 0.499 | 89.5 |
| **v2 frozen policy** | **63.31** | **73.35 / 0.550** | **54.69 / 0.282 / 66** | 61.89 / 0.919 / 0.499 | 72.3 |

| Check | Result |
|---|---|
| CC gain ≥ 5 | pass (+8.55) |
| Paired CC 95% CI lower bound > 0 | pass ([+3.69, +12.98]) |
| ≥ 200 independent cases | pass (431) |
| Choice and Noul checks, zero tolerance | pass |
| Score NLL and Brier not worse, zero tolerance | **fail** (`score/nll_not_worse`, `score/brier_not_worse`) |
| Speed axis ≥ 50 | pass (72.3) |

- **Score is close to v1.** v2 small's Score CC is 1.3 points higher than v1's. NLL and Brier are slightly worse, and the policy answered every Score question directly.
- **Report only: v2 policy over v2 direct.** This gate is not part of the decision rule, and small fails it:
  - the paired CC interval spans zero (+2.8, [−0.8, +7.0]);
  - reasoning on Noul lowered Noul CC for the Japanese and Korean MASSIVE sources.

## Lessons

- **Recipe transfer:** the large recipe transfers to base without change. Base is the first v2 size to pass the full predeclared gate, Score included.
- **small Score:** small fails on Score differences of 0.021 NLL and 0.005 Brier over 192 questions. This again shows the zero-tolerance per-type checks at the resolution limit of the cohort, as noted after the large run. No interval was computed for these differences, and the rule stays as predeclared for this run.
- **Reasoning on small Noul:** E2B's Noul reasoning helps overall but hurts ja/ko MASSIVE items. A future small policy should route Noul by language or source evidence rather than always reasoning.
