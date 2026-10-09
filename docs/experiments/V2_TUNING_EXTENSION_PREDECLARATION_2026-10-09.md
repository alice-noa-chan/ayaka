# v2 tuning-split extension: predeclared test (2026-10-09)

**Status:** predeclaration. This file is committed before the extension cohorts are read and before any dev result of the extended tuning is computed. Nothing here changes a serving default.

## Why

The levers of `V2_GATE_BLOCKERS_RESULT_2026-10-09.md` fitted well on calibration and router_train but did not transfer to dev. Those splits held 16–60 questions per source, too few to predict per-source checks that allow zero regression. This test triples both tuning splits and changes nothing else.

## Data

| Split | Existing | Extension | Total |
|---|---|---|---|
| calibration | 334 (`heldout-calibration-20261007`) | 668 (`heldout-calibration-ext-20261009`, `70f59317…`) | 1,002 |
| router_train | 334 (`heldout-router-train-20261007`) | 668 (`heldout-router-train-ext-20261009`, `a9a2431e…`) | 1,002 |
| dev | 668, unchanged, not re-read | — | 668 |

Both extension cohorts use dev's per-source composition. They exclude every earlier cohort by lineage and normalized state. Their manifests are in `docs/experiments/results/heldout-*-ext-cohort-20261009-manifest.json`.

## Reads (one A100 rental, approved estimate about $3.9, ceiling $5.5)

v2 off and v2 on for both extension cohorts, with the merged adapter (`--adapter merged`). The checkpoint, backbone, runtime and reading recipe are the same as in `V2_HELDOUT_MERGED_2026-10-09.md`. Each extension gets its own protocol from `make_protocol`.

## Analysis (fixed now)

```
python -m ayaka.eval.heldout_tuning --results <merged 10-09 results with dev/v1_on> \
  --extension <extension results> --adapter merged --out report.json
```

- The code (`heldout_tuning`, report `ayaka-heldout-tuning-2` with `--extension`) and every rule, grid and tolerance stay unchanged from the previous test.
- Path temperatures and both lever fits use the 1,002 calibration questions. The router regression and every policy choice use the 1,002 router_train questions. The router's λ and promotion still use dev, as `fit_router` is designed.
- **Single adoptable candidate:** exactly one candidate goes to the dev gate, and it is chosen on router_train only. It is whichever has the higher router_train equal-type CC of:
  - the base section's `policy_chosen_on_router_train`, with temperatures only;
  - the levers section's `policy_chosen_on_router_train`, with the Noul pool and Score temperature.
- **Adopt** only if that candidate passes both `quality_hierarchy.RULE` screens (over v1 on, and over v2 off with the same calibration) with zero failed checks. Every other dev line in the report is informational and not adoptable.
- **If it fails:** report the failed checks. There is no further dev-driven adjustment on these splits.

## Caveat

Dev has now been gated several times, for the 10-08 comparison, the merged re-read and the levers. Its rules were fixed in advance each time, and no fit uses dev except the router's λ check. Even so, repeated dev looks make a pass mildly optimistic. A pass is evidence for a promotion decision, made by the project owner, not a final result; the private holdout test stays unopened.
