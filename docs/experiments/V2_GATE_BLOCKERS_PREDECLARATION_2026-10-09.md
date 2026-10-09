# v2 gate blockers: predeclared levers (2026-10-09)

**Status:** predeclaration. This file is committed **before** the levers below are read on dev. Its dev result is reported separately, whether it passes or fails. Nothing here changes a serving default.

## Context

On merged held-out reads (`V2_HELDOUT_MERGED_2026-10-09.md`), `noul_always+router` beats v1 on dev by +8.15 CC (CI +3.73…+12.44). It still fails three checks of `quality_hierarchy.RULE`:

- `noul/nll_not_worse`;
- `source:commonsense_qa/choice/cc_not_worse`;
- `source:helpsteer2/score/cc_not_worse`.

Every lever below costs no new GPU read. It reuses probabilities that a routed request already computes. Each lever is fitted on `calibration` and selected on `router_train`, as in `heldout_tuning`. Dev has been used only for the existing router λ check and for the gate results already reported. Per-source dev numbers were seen once for diagnosis, before any lever was fitted.

## Levers

1. **Noul log-linear pool.** A reasoned Noul read is replaced by `softmax(a·log p_direct + b·log p_reasoned)`, using the direct read the request has already paid for.
   - `(a, b)` is fitted on calibration Noul pairs over the grid {0, 0.05, …, 1.5}².
   - The fit takes the lowest NLL among the points whose calibration Noul CC is not below that of the temperature-only reasoned read. The constraint keeps the pool from buying NLL with abstentions. Without it, the NLL-optimal pool added 9 abstentions on router_train.
   - **Fitted:** a = 0.25, b = 0.35. The calibration NLL falls from 0.196 to 0.180 at equal CC.
   - **router_train:** Noul NLL 0.171 → 0.149, with CC unchanged at 90.0.
   - The same pool on Choice made router_train NLL worse (0.425 → 0.434), so Choice is not pooled.
2. **Score direct temperature for CC.** Score CC uses the expected-position error, which rewards sharper distributions than the NLL-optimal T = 1.20.
   - T is the highest-CC point on calibration among those whose NLL is within 5% of the minimum, on the grid {0.50, 0.55, …, 1.50}.
   - The 5% tolerance was chosen on router_train among {1%, 2%, 5%}, where the sharpest candidate improved both CC and NLL.
   - **Fitted:** T = 0.80.
   - **router_train v2 off Score:** CC 70.0 → 72.1, NLL 0.819 → 0.795.
3. **Policy `noul+choice_always+router`.** It reasons on every Noul and Choice question, and on Score only where the router says so. The policy is chosen on router_train by the existing rule (highest equal-type CC):

   | Policy (with levers 1 and 2) | router_train CC | Speed axis |
   |---|---|---|
   | noul_always+router | 80.89 | 54.1 |
   | **noul+choice_always+router** | **81.82** | 51.5 |

## Predeclared dev test

There is a single dev read of `noul+choice_always+router` with levers 1 and 2. It is gated by `quality_hierarchy.RULE`, unchanged, against v1 on and against v2 off with the same calibration (temperatures plus lever 2).

- **Adopt** only if both screens pass with zero failed checks.
- **Otherwise:** report the failed checks. No further dev-driven adjustment is made to this package.

The two levers alone, on `noul_always+router`, are reported as a secondary line, for information only. It is not adoptable from this read, since picking it after seeing dev would be dev selection.
