# v2 held-out comparison, reasoned calibration and router (2026-10-08)

**Scope:** diagnostic development evidence on held-out cohorts. This is not an official JevBench rank. Nothing is promoted. The private test split was not opened, and no optimizer update was made.

## Why this run

The 10-07 dev cohort reused train rows that v1 was trained on (see `V2_GAP_ANALYSIS_2026-10-07.md` and the held-out cohort manifest), so the earlier v1 vs v2 comparison favored v1. This run builds three cohorts from validation/test splits only, with train-text overlap excluded:

| Split | Questions | Cases | Role |
|---|---|---|---|
| `dev` | 668 | 342 | scored comparison; `fit_router` lambda and promotion check |
| `calibration` | 334 | 173 | path temperatures only |
| `router_train` | 334 | 173 | router regression; choice between routing policies |

The three cohorts share no lineage. Their manifests are in `docs/experiments/results/heldout-*-cohort-20261007-manifest.json`.

## Execution

- **Hardware:** one vast.ai A100 SXM4 40GB (instance 54732175), $0.70/h, with source commit `2d680df`.
- **Elapsed and cost:** the rental ran 8.94 h. The billed instance charge was $6.24.
- **Integrity:** all 8 runs are complete with zero errors. The result archive has SHA-256 `a440c06e…`, and every one of its 29 files was verified by size and SHA-256 before the rental was destroyed.
- **Backbone:** `google/gemma-4-12B-it` at revision `707f0a3b` was downloaded on the host and checked against the pinned per-file SHA-256 values. Its weights are `5a84cb31…`, identical to the local native snapshot.
- **Reasoning speed:** reads with reasoning on took about 20 s per question, generating at about 10 tokens/s (budget 384; 41 of 668 dev reads stopped at the budget). This is far below what a 12B model should reach on an A100, and it is the main operational follow-up.

Reproduce the analysis from the retrieved folder:

```
python -m ayaka.eval.heldout_tuning --results <job>/retrieved/results --out report.json
```

The aggregate report is `docs/experiments/results/heldout-tuning-2026-10-08.json`. Rows are not committed because some sources are CC BY-SA.

## 1. Fair dev comparison (no tuning)

| System | CC (equal types) | Choice | Noul | Noul abstentions | Score | Mean latency |
|---|---|---|---|---|---|---|
| v1 on (published) | 67.47 | 79.1 | 65.6 | 41 | 57.6 | 0.85 s |
| v2 off | 63.57 | 78.5 | 52.5 | 63 | 59.7 | 0.34 s |
| v2 on | 75.44 | 81.9 | 83.7 | 3 | 60.6 | 19.9 s |

- **v2 off vs v1 on:** −3.90 (paired 95% CI −9.79 to +1.24). Choice and Score are level or better. The whole gap is **Noul abstention**:
  - decisive-wrong counts are equal (13 vs 14) and v2's Noul AUC is 0.945;
  - the 63 abstained P(true) values sit just inside the 0.2/0.8 band (gold-false around 0.22–0.25, gold-true around 0.6–0.77).
- **v2 on vs v2 off:** +11.87 (CI +7.21 to +16.92), with Noul abstentions falling from 63 to 3. The gain holds on natural sources:

  | Source | v2 off | v2 on |
  |---|---|---|
  | hotpotqa | 60 | 85 |
  | massive_ja | 59 | 84 |
  | strategyqa | 59 | 91 |

  However, the reasoned reads are over-confident (Score ECE 0.36, Choice 0.11), so every NLL/Brier check fails.

## 2. E2: Noul label position

v2 off was re-read with the Noul labels in reversed order (`true_first`):

- mean P(true) moved by −0.022;
- AUC changed from 0.945 to 0.947;
- abstentions changed from 63 to 53, and 27 argmax decisions flipped.

Averaging the two orders gives Noul CC 53.75 (+1.25). **Label position bias is small and is not the cause of the abstentions.**

## 3. Path temperatures (fitted on calibration only)

| Path | T |
|---|---|
| choice / direct | 1.19 |
| choice / reasoned | 1.73 |
| noul / direct | 1.02 |
| noul / reasoned | 2.56 |
| score / direct | 1.17 |
| score / reasoned | 2.61 |

Reasoned reads need strong softening. With temperatures, v2 on's dev ECE improves from 0.36 to 0.15 (Score), 0.11 to 0.06 (Choice) and 0.07 to 0.03 (Noul).

Noul direct is already NLL-calibrated (T≈1). An NLL fit therefore does **not** sharpen it, and does not reduce its abstentions.

## 4. Router (fitted on calibrated router_train pairs)

- **Validation:** promoted, with λ = 0.001. The dev NLL gain is 0.0316 (95% CI 0.0032 to 0.0618) over 342 independent cases, and budget 384 is validated.
- **Routing on dev:** the router reasons on 95 of 668 questions (55 Choice, 40 Noul, 0 Score). Calibrated reasoning lowers Score CC (58.9 → 56.6), so leaving Score direct is the right call.
- **Caveat:** λ selection and promotion use dev pairs, as `fit_router` is designed, so the router's dev reading is mildly optimistic.

## 5. Routing policies

The policy choice is made on router_train (highest CC), not on dev.

| Policy | router_train CC | dev CC | dev Noul abstentions | dev latency p50 / p95 | Speed axis |
|---|---|---|---|---|---|
| direct (+T) | 69.47 | 63.07 | 64 | 0.24 / 0.84 s | 79.4 |
| router | 77.24 | 71.01 | 37 | 0.25 / 22.8 s | 65.2 |
| noul_always | 77.80 | 73.70 | 6 | 1.46 / 20.8 s | 58.9 |
| **noul_always+router** (chosen) | **80.58** | **76.01** | 6 | 10.4 / 25.4 s | 49.7 |

Dev gates under the predeclared rule (`quality_hierarchy.RULE`):

| Policy | Over calibrated v2 off | Over v1 on |
|---|---|---|
| router | +7.94, CI +4.96…+11.31, **passed** | +3.54, CI −1.56…+8.16, failed |
| noul_always | +10.62, CI +7.66…+13.86, **passed** | +6.23, CI +1.03…+10.81, failed |
| noul_always+router | +12.94, CI +9.37…+16.72, **passed** | +8.54, CI +4.05…+12.80, failed |

For noul_always+router over v1 on, only four checks fail:

- choice/nll;
- noul/nll (0.227 vs 0.223);
- commonsense_qa Choice CC;
- helpsteer2 Score CC.

## Conclusions

1. **The v2 deficit was abstention, not knowledge.** Reasoning removes the Noul abstentions and the gain survives held-out natural sources.
2. **Calibration is solved for reasoned reads by path temperatures**, fitted on a separate split.
3. **The chosen policy is noul_always+router, but it is too slow as served.** At about 10 tokens/s its median is 10 s, which puts the Speed axis below 50. Since Speed enters the leaderboard as a harmonic-mean component with a penalty below 50, faster generation is now the highest-value fix. A 3× speed-up would bring p50 to about 3–4 s.
4. **Remaining v1-gate blockers:** small NLL deficits on Choice/Noul, commonsense_qa Choice and helpsteer2 Score.

## Next

- Profile and speed up reasoned generation. The candidates are a merged LoRA for generation, a static cache with CUDA graphs, and vLLM.
- Re-read only Choice/Score sources that block the v1 gate, with no training.
- Add a v1 calibration read (direct, about 10 minutes). Without it, a Noul temperature for v1 cannot be fitted on the calibration split.
