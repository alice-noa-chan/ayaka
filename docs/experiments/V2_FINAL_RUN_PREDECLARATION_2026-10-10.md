# Final v2 run: predeclared training, tuning and decision (2026-10-10)

**This is the last v2 experiment.** It is written before any GPU rental, before the new model exists and before any read of the final_test cohort. Whatever the outcome, no setting below is changed afterwards, no further fit is made on any split, and no follow-up experiment is run. The result decides whether v2 is released.

## Why this run

`V2_BLOCKER_CAUSES_2026-10-09.md` found three reasons why the best v2 state (dev CC 75.62, +8.15 over v1) failed the v1 gate:

1. **Training scale.** v2 had 74 optimizer updates over 2,368 questions at LoRA rank 32. v1 had 860 updates at rank 64 over ~35 sources, including ~9.6k CommonsenseQA rows and ~101k HelpSteer2 questions. v2 never saw StrategyQA.
2. **Confident wrong answers on reasoned MASSIVE reads.** Temperature cannot fix them.
3. **A gate at its statistical resolution.** About 26 zero-tolerance checks are applied to subgroups of 24–96 cases.

This run addresses each:

- Scale: more data from the same license-clean sources, StrategyQA added, rank 64, two epochs.
- Overfitting: validation-based checkpoint selection with early stopping.
- MASSIVE: more MASSIVE training data, so the direct read is sharper and the router can prefer it.
- Gate: the clustered non-inferiority rule chosen by the project owner, applied once to a cohort no fit has seen.

## Code

All of it is on `ayaka-v2-experiments`. The rental runs the exact commit recorded in its anchors.

| Commit | Change |
|---|---|
| `a37f2f1` | StrategyQA as an opt-in direct training source (corpus plan3) |
| `83a15de` | Validation-based checkpoint selection with early stopping |
| `7d77b1c` | Clustered non-inferiority rule for subgroup CC checks |
| `ffd02bd` | `final_test` held-out cohort split |
| `c8234c5` | final_test settings and manifest |
| `ff25c52` | Frozen tuning fit and the once-only final gate |
| `57b7f9c` | Reasoned final_test read limited to routed questions |
| `7dbb00e` | Clearer natural quota error |
| `3479bc5` | Routing policies are eligible only with a promoted router |
| `4d4628d` | Final corpus plan3 settings |

Prepared bundle anchors:

- corpus plan `8d65fcaa…`;
- bundle manifest `e4795ccd…`;
- CPU audit receipt `04e7df75…`.

The training bundle and its CPU audit were prepared from a `git archive` export of `4d4628d`, so the audited source bytes are the bytes the rental runs. The working tree's line endings play no part.

## 1. Training corpus

`docs/experiments/direct_final_corpus_settings.json` defines corpus plan `ayaka-direct-corpus-plan-3`. All rows come from original train files.

| Source | License | Train samples | Train questions | Each of router_train / dev / calibration |
|---|---|---|---|---|
| HelpSteer2 | CC BY 4.0 | 3,000 responses | 15,000 Score | 32 responses |
| CommonsenseQA | MIT | 6,400 | 6,400 Choice | 128 |
| MASSIVE ko | CC BY 4.0 | 3,000 utterances | 6,000 Noul | 48 |
| MASSIVE ja | CC BY 4.0 | 3,000 utterances | 6,000 Noul | 48 |
| ContractNLI | CC BY 4.0 | 108 documents | 1,836 Choice | 4 documents |
| StrategyQA | MIT | 1,036 | 1,036 Noul | 48 |
| Authored verified (train voice) | repository | 256 per type | 768 | 32 per type |
| **Total** | | | **37,040 rows** | 692 questions each |

- **Prepared size:** 12.93M text tokens per epoch, 25.87M for the two-epoch schedule. ContractNLI was lowered from 128 to 108 documents because only 109 remain in its train bucket; StrategyQA was set to 1,036 to keep whole batches.
- **Exclusions:** the reserved inventory holds the natural questions of every held-out cohort:
  - dev, calibration and router_train;
  - the 10-09 calibration and router_train extensions;
  - final_test.

  Any training sample that shares a source group, a lineage or 13 normalized words of state evidence with them is excluded.
- **Authored items:** these come from the same generator as the cohorts' synthetic questions, and share template wording. They are kept apart by voice and index instead:
  - training uses the train voice;
  - the bundle's internal splits use indices 0–31 of their own voices;
  - the cohorts use indices 32 and above;
  - final_test uses the dev voice at indices 64–95.

  The previous v2 training used the same generator in the same way.

## 2. Training

| Setting | Value |
|---|---|
| Backbone | `google/gemma-4-12B-it` @ `707f0a3b`, pinned native BF16 weights |
| Kernels | native PyTorch attention, no Liger: the variant the previous run trained with after its deterministic-probe fix (`6933493`). FlashAttention 2 and Liger failed its loss or probability parity probes |
| Adapter | LoRA rank 64, alpha 128, dropout 0.05, all attention and MLP projections |
| Schedule | 2 epochs, 32 rows per step: 2,315 optimizer steps, cosine with 3% warmup |
| Optimizer | AdamW, LoRA lr 1e-4, head lr 5e-4, weight decay 0.01, grad clip 1.0 |
| Losses | as in the previous run: NLL 1.0, Brier 0.5, RPS 0.35, missing 0.25 |
| Checkpoint selection | bundle router_train (692 questions), every 250 steps and at the last step; equal-type mean raw NLL; patience 3; min_delta 0 |
| Export | the selected weights, with the pipeline's own calibration on the bundle calibration split |

Training stops early after three reads in a row without improvement. The exported adapter is the best read, not the last step.

## 3. Held-out reads (merged adapter)

| Cohort | Questions | v2 direct | v2 reasoned | v1 |
|---|---|---|---|---|
| calibration | 334 | all | all | — |
| router_train | 334 | all | all | — |
| dev | 668 | all | all | all (re-read under the new protocol) |
| final_test | 836 | all | only the questions the frozen policy routes | all |

The 10-09 extension cohorts are not re-read; tripling the tuning splits did not change the fits. v1 is the published `alice-noa-chan/ayaka-large` with its frozen reasoning policy.

## 4. Tuning (fixed before final_test is read)

```bash
python -m ayaka.eval.heldout_tuning --results results --adapter merged \
  --out tuning.json --frozen-out frozen.json
```

- Path temperatures are fitted on calibration.
- The router is fitted on router_train, with lambda and promotion from dev.
- The policy is the base-section choice on router_train among `direct`, `router`, `noul_always` and `noul_always+router`. The two routing policies are eligible only if `fit_router` promotes the router, because serving never routes with an unpromoted one. The lever section is reported, not used.
- `frozen.json` is written, and its SHA-256 recorded, before any final_test reasoned read starts.
- `final_gate route-ids` then lists the routed final_test questions from the direct reads alone.

## 5. Decision (one read of final_test)

```bash
python -m ayaka.eval.final_gate gate --results results --frozen frozen.json --out final.json
```

**v2 is adopted if and only if both of these hold:**

1. **The v1 gate passes on final_test**, comparing the frozen policy with v1 on the same 836 questions:
   - CC gain of at least 5 and a classification credit gain of at least 5 points;
   - a paired 95% CC interval (whole cases, seed 15) whose lower bound is above 0;
   - at least 200 independent cases (final_test has 431);
   - no per-type regression in CC, NLL, Brier, RPS or Score nMAE (zero tolerance);
   - no significant per-source or per-language CC regression: the upper bound of each subgroup's clustered 95% interval is at least 0.
2. **The policy's Speed axis is at least 50** on the rental's hardware, so the JevBench Speed penalty does not apply.

The gate over calibrated v2 direct and the dev gates are reported but do not decide.

- **If adopted:**
  1. Add the one-command train-to-serve pipeline on `ayaka-v2-experiments`, reproducing the frozen fit.
  2. Publish to Hugging Face.
  3. Merge into `main`.
- **If not adopted:**
  - v1 stays the released model;
  - the result is reported as it is;
  - nothing is refitted, re-gated or retried.

## 6. Budget

One vast.ai RTX PRO 6000 (96 GB), price cap $1.60/h; the previous v2 training ran at $1.33/h on this class.

| Stage | Expected time |
|---|---|
| Setup, uploads, backbone fetch | 0.7 h |
| Training (25.87M tokens at the previous 1,700 tokens/s), selection reads, calibration, export | 5.0 h |
| Direct reads, all cohorts (2,172 questions) | 0.3 h |
| v1 reads, dev and final_test (1,504 questions) | 0.7 h |
| v2 reasoned reads, calibration, router_train, dev (1,336 questions) | 4.3 h |
| v2 reasoned reads, routed final_test questions (about 420) | 1.4 h |
| Tuning, final gate, archive, download | 0.4 h |
| **Total** | **about 12.5 h, about $17 at $1.33/h plus storage and bandwidth** |

- **Guardian:** stops compute at 16 h, which is at most $25.6 at the price cap.
- **Credit check:** run_direct's whole-workflow admission applies a 1.25 safety factor to its stage allowances. It therefore needs about $26 of credit at $1.33/h, or about $31 at $1.60/h, before it starts training.
- **Approval:** the rental needs explicit approval and a credit top-up before it starts.

## Integrity

- The bundle's private test split is not opened.
- final_test is read once, after `frozen.json` exists.
- Dev has been gated four times before; it is now only a tuning split for the router's lambda and promotion.
- No dataset rows are committed; reports contain aggregates only.

## Amendment before any optimizer update (2026-10-10)

The first rental (instance 55120004, RTX PRO 6000 S) stopped at run_direct's whole-workflow admission, **before its first optimizer update**. No model was trained and no cohort was read.

- **Why it stopped:**
  - The admission charged each of the 2,315 steps the slowest sampled scheduled batch: 53.4 s, against a 3.1 s median. That batch held long ContractNLI documents and had backed off after CUDA OOM.
  - The forecast was therefore 34 h of training and $74 for the workflow, against $32.66 of credit.
- **Realistic rate:** the 22 schedule samples processed 1,836 tokens/s in aggregate. At that rate the 25.87M-token schedule needs about 3.9 h, in line with the original estimate.
- **Change:** commit `8cc7433` adds `run_direct --forecast-basis token_rate`. It divides every scheduled token by the sampled scheduled token rate. The 1.25 safety factor and the credit admission are unchanged.
- **What did not change:** corpus, plan, schedule, model, selection, tuning, final_test and decision rule. Only the source commit, and so the bundle and audit anchors, change.
- **Rental time limit:** raised from 16 h to 20 h, to cover the rental time already used. The compute ceiling becomes about $27.5 at $1.38/h.

## Amendment: learning rate, after a training-internal divergence (2026-10-10)

The second rental (instance 55134917) trained with the declared learning rates: LoRA 1e-4, head 5e-4. The checkpoint-selection reads on the bundle's own router_train split showed a divergence:

| Step | Choice NLL | Noul NLL | Score NLL | Equal-type CC |
|---|---|---|---|---|
| 250 | 0.602 | 0.256 | 1.008 | 65.5 |
| 500 | 1.479 | 0.697 | 1.438 | −23.7 |

A CC below chance is a divergence, not overfitting. The previous v2 run (74 steps) never trained at the peak learning rate for long.

- **Change:** the run was stopped at step ~600. It restarts from the backbone with both learning rates scaled by 0.3: LoRA 3e-5, head 1.5e-4. The new rates are set in the payload's `training-config.json`.
- **What did not change:** corpus, bundle and audit anchors, schedule, warmup fraction, selection rule, tuning, final_test and the decision rule.
- **Integrity:** the evidence for this change is the training-internal selection split only. No held-out cohort and no final_test question had been read by the new model.
