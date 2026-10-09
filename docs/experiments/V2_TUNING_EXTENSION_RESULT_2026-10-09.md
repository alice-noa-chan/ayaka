# v2 tuning-split extension: result of the predeclared test (2026-10-09)

**Outcome: failed. Nothing is adopted.** This is the single dev test predeclared in `V2_TUNING_EXTENSION_PREDECLARATION_2026-10-09.md` (commit `247b8c0`, made before the extension cohorts were read). No serving default changes, and there is no further dev-driven adjustment on these splits.

- **Reads:** v2 off and v2 on, merged adapter, on `heldout-calibration-ext-20261009` and `heldout-router-train-ext-20261009` (668 questions each, 0 errors). Dev and v1 on are the merged 10-09 reads of `V2_HELDOUT_MERGED_2026-10-09.md`, not re-read.
- **Rental:** one vast.ai A100 (instance 54982698), 5.97 h. The compute estimate is $3.89. Account credit fell from $8.61 to $3.41, which includes bandwidth and storage; this is not the final invoice. Both are within the approved $5.5 ceiling. The rental was destroyed after the results were retrieved and verified.
- **Code:** `heldout_tuning` at `5ab1f2c` (report `ayaka-heldout-tuning-2`), exactly the predeclared command.
- **Aggregate report:** `docs/experiments/results/heldout-tuning-extension-2026-10-09.json`.
- **Safety:** optimizer updates 0; the original private test was not opened.

## The single candidate

The predeclaration sends exactly one candidate to the dev gate: whichever of two policies has the higher router_train equal-type CC.

| Section | Policy chosen on router_train | router_train CC |
|---|---|---|
| base (temperatures only) | `noul_always+router` | 77.19 |
| levers (Noul pool + Score T) | `noul+choice_always+router` | **78.49** |

The candidate is therefore the levers section's `noul+choice_always+router`.

The levers refitted on the 1,002 calibration questions came out the same as on 334: Noul pool weights a = 0.25 (direct) and b = 0.35 (reasoned), and Score direct T = 0.80. The router was promoted with λ = 0.0005.

## Results

| Policy | Split | CC | Choice CC / NLL | Noul CC / NLL / abst. | Score CC / NLL | Speed axis |
|---|---|---|---|---|---|---|
| direct, T only | dev | 62.94 | 78.5 / 0.427 | 51.2 / 0.295 / 65 | 59.1 / 1.050 | 81.7 |
| router, T only | dev | 73.94 | 82.6 / 0.414 | 80.0 / 0.226 / 12 | 59.2 / 1.047 | 68.9 |
| noul_always+router, T only | router_train | 77.19 | 81.5 / 0.387 | 86.7 / 0.185 / 9 | 63.4 / 0.868 | 53.6 |
| noul_always+router, T only | dev | 74.78 | 82.6 / 0.414 | 82.5 / 0.231 / 7 | 59.2 / 1.047 | 55.2 |
| **noul+choice_always+router + levers** (candidate) | router_train | 78.49 | 84.2 / 0.430 | 85.4 / 0.166 / 13 | 65.8 / 0.858 | 50.7 |
| **noul+choice_always+router + levers** (candidate) | dev | **75.03** | 84.0 / 0.407 | 80.6 / 0.227 / 12 | 60.4 / 1.084 | 51.8 |

v1 on (dev): CC 67.47, Choice 79.1 / 0.407, Noul 65.6 / 0.223 / 41 abstentions, Score 57.6 / 1.084.

Dev gates (`quality_hierarchy.RULE`, unchanged):

| Screen | CC delta | 95% CI | Result |
|---|---|---|---|
| over v1 on | +7.56 | +3.08…+12.00 | **failed** (3 checks) |
| over v2 off (same calibration) | +11.61 | +7.63…+15.87 | **failed** (5 checks) |

The candidate fails these checks:

- **Over v1 on:** noul/nll (0.227 vs 0.223), commonsense_qa Choice CC and helpsteer2 Score CC. These are the same three blockers as before the extension.
- **Over v2 off:** score/cc, score/nmae, ayaka-v2-verified Score CC, en Score CC and contract_nli Choice CC.

## Comparison with the 334-question splits

| Tuning splits | Base chosen policy, dev CC | Its fails over v1 | Candidate, dev CC | Its fails over v1 / over v2 off |
|---|---|---|---|---|
| 334 + 334 (`V2_GATE_BLOCKERS_RESULT_2026-10-09.md`) | 75.62 | 3 | 75.07 | 5 / 1 |
| 1,002 + 1,002 (this test) | 74.78 | 5 | 75.03 | 3 / 5 |

## What the read shows

1. **Tripling the tuning data did not make the fits transfer to dev.** The fitted values barely moved: the lever weights are identical, and no path temperature changed by more than 0.18 (the largest change is Choice direct, from 1.215 to 1.039). The dev failures concentrate on the same primitives and sources as before.
2. **The remaining blockers look like model limits, not fitting noise.**
   - Noul NLL: the best dev value (0.226–0.231) stays just above v1's 0.223 under every policy and fit.
   - commonsense_qa Choice and helpsteer2 Score: these trail v1 under every policy.
   - Neither more temperature data nor pooling closes these gaps.
3. **The Score lever trades a v2-off check for a v1 check.** The sharper Score direct temperature (T = 0.80) brings Score NLL level with v1 (1.084), so the v1 screen loses its score/nll failure. It also makes Score CC and nMAE worse than v2 off with the same calibration, so the v2-off screen now fails on Score.
4. **The router chose differently on the larger split.** With 1,002 router_train questions, base dev Choice CC fell from 85.4 to 82.6, and base dev CC from 75.62 to 74.78. The router is refitted on each split, so this is the variance of the whole routing fit, not a defect.

## Conclusions

- The best adoptable state stays as it was: merged `noul_always+router` with temperatures fitted on the original 334-question splits. It scores dev CC 75.62, +8.15 over v1, and fails three v1-gate checks. It is still **not** adopted.
- Free post-hoc work on these reads is exhausted: temperatures, pooling, policy choice and now 3× tuning data. Clearing the three blockers needs a model change on Noul NLL, commonsense_qa Choice and helpsteer2 Score, for example targeted training data or a revised objective. That is a GPU training decision for the project owner.
- The gate rule itself is unchanged. Whether zero-tolerance per-source checks at about 100 questions per source are the right promotion bar is the project owner's decision, made independently of this result.
- Dev has now been gated four times. Every rule was fixed in advance, but a future pass on this dev would be mildly optimistic. The private holdout test remains unopened.
