# v2 base and small run: predeclared training, tuning and decision (2026-10-11)

This document is written before any GPU rental, before either model exists and before either reads the final_test cohort. It applies the recipe of the large final run (`V2_FINAL_RUN_PREDECLARATION_2026-10-10.md`, with its two amendments) to the two smaller sizes. No setting below is changed after a held-out read, and no fit is repeated on any split.

## Why this run

v2 large was trained, tuned and gated on 2026-10-10 (`V2_FINAL_RUN_RESULT_2026-10-10.md`). It gained +10.8 equal-type CC over v1 large on final_test but failed the zero-tolerance Score checks. It is published as `alice-noa-chan/ayaka-v2-large`. Base and small have no v2 counterpart yet.

## What is the same as the large run

- **Corpus:** plan `ayaka-direct-corpus-plan-3` from `docs/experiments/direct_final_corpus_settings.json`, with StrategyQA and ContractNLI, and the same reserved manifest (`7988a84f…`). Every held-out cohort question stays excluded by source group, lineage and 13-word overlap.
- **Training:**
  - two epochs, 32 rows per step, 2,315 steps, cosine schedule with 3% warmup;
  - LoRA learning rate 3e-5 and head learning rate 1.5e-4 (the amended large rates);
  - losses NLL 1.0, Brier 0.5, RPS 0.35, missing 0.25;
  - checkpoint selection on the bundle's router_train every 250 steps, patience 3;
  - native attention, no Liger, and the token-rate credit forecast.
- **Held-out reads:** the same four cohorts (calibration 334, router_train 334, dev 668, final_test 836 questions), the merged v2 adapter, the routed-only reasoned final_test read.
- **Tuning:** `heldout_tuning` with the same fits and policy choice. `frozen.json` is sealed before final_test is read.
- **Decision rule:** `final_gate`, unchanged:
  1. the v1 gate passes on final_test (CC and credit gains of at least 5, paired 95% CC interval above 0, at least 200 cases, zero-tolerance per-type checks, clustered subgroup non-inferiority);
  2. the policy's Speed axis is at least 50.

## What differs

| Setting | small | base |
|---|---|---|
| Backbone | `google/gemma-4-E2B-it` @ `3e22461f` | `google/gemma-4-E4B-it` @ `ee0ef602` |
| LoRA | rank 32, alpha 64, dropout 0.05 | rank 32, alpha 64, dropout 0.05 |
| Head | pointer 256, 2 mixer layers, 4 heads | pointer 384, 2 mixer layers, 6 heads |
| v1 compared against | `alice-noa-chan/ayaka-small` @ `9277bdcc` | `alice-noa-chan/ayaka-base` @ `064eac5e` |

- **Size settings:** each size keeps the backbone, LoRA rank and head dimensions of its v1 counterpart (`ayaka.training.v2_recipe`, `--size small|base`). The learning rates were validated only on large.
- **Comparison:** v2 small is compared with v1 small and v2 base with v1 base, each on its own backbone (`checkpoint_comparison --model-size`). The two sizes are decided independently.
- **Survey:** E2B was added to the pinned backbone survey (`a09127f`).

## Code

| Commit | Change |
|---|---|
| `f7a9eb2` | base and small model configs in the v2 recipe |
| `ccea169` | protocols and v1 checks for each size's own v1 and backbone |
| `a09127f` | Gemma 4 E2B in the pinned backbone survey |

Each size's training bundle and CPU audit were prepared from a `git archive` export of `a09127f`. The rental runs that source.

| Anchor | small | base |
|---|---|---|
| corpus plan | `b7584488…` | `55f3dc0b…` |
| bundle manifest | `67c52342…` | `ec369324…` |
| CPU audit receipt | `f0885f71…` | `9ed957a9…` |
| native snapshot | `93159957…` | `18cc9621…` |

Both bundles select the same 37,040 training rows per source as the large bundle, with the same exclusions (947 reserved or public overlaps, 24 duplicates, 1 context overflow). Their membership digests differ from large's because each size encodes its rows with its own backbone's chat template.

## Schedule on one GPU

Both sizes share one RTX PRO 6000 (96 GB):

1. **Training, concurrently.** Base starts first. Small starts once base has profiled its schedule, so the two memory stress checks do not race. Training latency feeds no decision.
2. **Tuning reads, possibly concurrently.** Each size reads calibration, router_train and dev (direct and reasoned) and v1 dev, as soon as its training ends, even while the other size still trains or reads. Their latencies feed no fit or choice: temperatures, router and policy use probabilities only. Dev latencies in the reports are therefore measured on a shared GPU.
3. **final_test reads, one at a time.** After both sizes have sealed `frozen.json`, each size's final_test reads (v2 direct, v1, routed v2 reasoned) run with the GPU otherwise idle. Their latencies set the Speed axis of the gate, so they are measured as in the large run.

A failure of one size does not stop the other.

## If a size passes or fails

- **Passes:** it is published as `alice-noa-chan/ayaka-v2-<size>` with its frozen fit.
- **Fails:** the result is reported as it is. Nothing is refitted, re-gated or retried. As with large, the project owner may still publish it as a separate model with the failure stated.

## Budget

One vast.ai RTX PRO 6000 S at about $1.46/h, from the current credit of $16.42 without a top-up.

| Stage | Expected time |
|---|---|
| Setup, uploads, two backbone fetches (26 GB) | 0.4 h |
| Training of both sizes, concurrently | 2.5 h |
| Tuning reads after training (both sizes, sharing the GPU) | 1.5–2 h |
| final_test reads, one size at a time | 1.2 h |
| Tuning, gates, archive, download | 0.3 h |
| **Total** | **about 6–7.5 h, about $9–11** |

- **Basis:** the large run measured a 3.9 h training forecast and 3.0 h of reads for 12B. The smaller backbones need less compute per token, but generation was overhead-bound (38 ms per token on 12B), so reads are expected to shrink less than training.
- **Guardian:** stops compute at 10 h, at most $14.9 at the $1.49 price cap.
- **Credit check:** run_direct's admission (safety factor 1.25) fits the stage allowances of `launch.py` within the current credit.

## Integrity

- Neither bundle's private test split is opened.
- final_test is read once per size, after that size's `frozen.json` is sealed.
- No dataset rows are committed; reports contain aggregates only.
