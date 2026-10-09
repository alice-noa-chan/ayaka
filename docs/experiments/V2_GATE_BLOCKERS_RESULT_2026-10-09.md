# v2 gate blockers: result of the predeclared levers (2026-10-09)

**Outcome: failed. Nothing is adopted.** This is the single dev read predeclared in `V2_GATE_BLOCKERS_PREDECLARATION_2026-10-09.md` (commit `8a3d04a`, made before this read). The levers stay off, and there is no further dev-driven adjustment of this package.

- **Reads:** the merged held-out reads of `V2_HELDOUT_MERGED_2026-10-09.md`. No new GPU read was made.
- **Code:** `heldout_tuning` (report version `ayaka-heldout-tuning-2`), `levers` section.
- **Aggregate report:** `docs/experiments/results/heldout-tuning-levers-2026-10-09.json`.

## Fitted levers (calibration only)

These match the predeclaration exactly:

- Noul pool weights: a = 0.25 (direct), b = 0.35 (reasoned);
- Score direct T = 0.80;
- policy chosen on router_train: `noul+choice_always+router`.

## Results

| Policy | Split | CC | Choice CC / NLL | Noul CC / NLL / abst. | Score CC / NLL | Speed axis |
|---|---|---|---|---|---|---|
| noul_always+router, T only (baseline) | router_train | 80.21 | 80.6 / 0.425 | 90.0 / 0.171 / 0 | 70.0 / 0.819 | 54.1 |
| noul_always+router, T only (baseline) | dev | 75.62 | 85.4 / 0.401 | 82.5 / 0.232 / 7 | 59.0 / 1.050 | 54.4 |
| noul_always+router + levers | router_train | 80.89 | 80.6 / 0.425 | 90.0 / 0.149 / 1 | 72.1 / 0.795 | 54.1 |
| noul_always+router + levers | dev | 75.53 | 85.4 / 0.401 | 80.6 / 0.227 / 12 | 60.6 / 1.087 | 54.4 |
| **noul+choice_always+router + levers** (predeclared) | router_train | 81.82 | 83.3 / 0.431 | 90.0 / 0.149 / 1 | 72.1 / 0.795 | 51.5 |
| **noul+choice_always+router + levers** (predeclared) | dev | **75.07** | 84.0 / 0.408 | 80.6 / 0.227 / 12 | 60.6 / 1.087 | 51.9 |

v1 on (dev): CC 67.47, Choice NLL 0.407, Noul NLL 0.223, Score NLL 1.084.

Dev gates (`quality_hierarchy.RULE`, unchanged):

| Policy | Over v1 on | Over v2 off (same calibration) |
|---|---|---|
| predeclared | +7.59, CI +3.05…+11.99, **failed** (5 checks) | +11.44, CI +7.72…+15.52, **failed** (1 check) |
| secondary, not adoptable | +8.06, CI +3.32…+12.39, failed (4 checks) | +11.90, passed |

The predeclared policy fails these checks:

- **Over v1 on:** choice/nll, noul/nll, score/nll, commonsense_qa Choice CC and helpsteer2 Score CC.
- **Over v2 off:** contract_nli Choice CC.

## What the read shows

1. **The Noul pool did not transfer.** Calibration and router_train both showed lower NLL at equal CC, with at most one abstention. On dev, NLL fell only from 0.232 to 0.227, still above v1's 0.223. Abstentions rose from 7 to 12, and CC fell by 1.9. The CC floor was enforced on calibration, but it does not hold on another split of 320 Noul questions.
2. **The sharper Score temperature traded NLL for CC on dev.** Score CC rose from 59.0 to 60.6, while Score NLL rose from 1.050 to 1.087 and now fails against v1 (1.084). helpsteer2 still trails v1.
3. **Reasoning on every Choice question helped router_train but not dev.** On router_train it took commonsense_qa from 79.2 to 84.4. On dev it lowered Choice CC and made contract_nli worse than direct.
4. **The common cause is split variance.** router_train has 98 Choice, 160 Noul and 76 Score questions, and a source contributes 16–60 questions to it. These sizes are too small to predict per-source dev checks that allow zero regression. helpsteer2 Score CC is 69.4 on router_train, 50.4 on calibration and 54.0 on dev from the same model.

## Conclusions

- Free post-hoc levers fitted on these tuning splits do not clear the three remaining v1-gate blockers. The best adoptable state is still merged `noul_always+router` with temperatures only: +8.15 over v1, failing three checks.
- Closing the gap needs either a better model on the failing primitives, or more tuning data so that fits transfer. One option is larger calibration and router_train cohorts from the same held-out sources (direct reads cost about 10 minutes of GPU, reasoned reads several hours).
- The gate rule itself is unchanged. Whether per-source zero-tolerance checks are the right bar at about 100 questions per source is a decision for the project owner. It cannot be settled after seeing this result.
