# Qwen3.5-4B pilot: predeclared backbone comparison (2026-10-11)

This document is written before any GPU rental and before the pilot model exists. It fixes the training, the reads and the decision rule of a cheap pilot. The pilot asks one question: does the v2 recipe work better on Qwen3.5-4B than on Gemma 4 E4B?

## Why this pilot

- **Results so far:** v2 base on Gemma 4 E4B passed the full release gate on 2026-10-11 (`V2_FAMILY_RUN_RESULT_2026-10-11.md`).
- **The candidate:** Qwen3.5-4B is pinned in the backbone survey (`v2_candidates.json`, revision `851bf6e8`, Apache-2.0). The 2026-10-01 selection deferred it for budget, so it has never been measured.
- **Why it is interesting:**
  - it has 4.7B parameters against E4B's 8B;
  - it is strong on multilingual text;
  - another Jev-style decision model (Jeeves) uses a Qwen3.5 backbone.
- **Scope:** only the backbone changes. A pass does not release anything; it only justifies a predeclared full run.

## What is the same as v2 base

- **Corpus:** plan `ayaka-direct-corpus-plan-3`, the same reserved manifest (`7988a84f…`) and the same exclusions. Rows are encoded with Qwen's own chat template.
- **Training:**
  - two epochs, 32 rows per step, 2,315 steps, cosine schedule with 3% warmup;
  - LoRA learning rate 3e-5 and head learning rate 1.5e-4;
  - losses NLL 1.0, Brier 0.5, RPS 0.35, missing 0.25;
  - checkpoint selection on the bundle's router_train every 250 steps, patience 3.
- **Size settings:** LoRA rank 32, alpha 64, dropout 0.05; head pointer 384, 2 mixer layers, 6 heads (the v2 base settings).

## What differs

- **Backbone:** `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, text stack only. The vision tower and the multi-token-prediction weights are not loaded.
- **LoRA targets:** the survey's targets for this architecture. These are the attention and MLP projections plus the linear-attention projections (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`). Three of every four Qwen3.5 layers use linear attention. With only the Gemma targets, those layers would get no adapter on their attention.
- **Kernels:** flash-linear-attention 0.5.2 (`fla-core` and `flash-linear-attention`, pinned PyPI wheels) is added to the pinned runtime. Without it transformers falls back to a reference PyTorch path that it documents as over ten times slower. The job checks the kernel on CUDA before training and stops if it is missing.

## Reads

- **Cohorts:** calibration (334 questions) and dev (668), read in a single pass with the LoRA merged. router_train (334) is also read, so that a full run could reuse it.
- **Not read:**
  - no reasoning read;
  - no v1 read;
  - no tuning fit beyond path temperatures;
  - final_test is not in the payload and is not read.
- **Code:** the reads use `checkpoint_comparison` with the new `v2_backbone` protocol field (`87f5c00`).

## Decision rule

- **Calibration:** each model's dev read is calibrated with path temperatures fitted on its own calibration read (`pilot_compare.py`, the same functions as `heldout_tuning`).
- **Reference:** the E4B v2 base reads of rental 55302429, which were made on the same GPU class.

  On those reads, the comparison script reproduces the tuning table's calibrated direct dev CC of 66.11.
- **Comparison:** Qwen minus E4B equal-type CC on dev, paired by question, resampling whole cases (2,000 replicates).
- **Rule:** **proceed** to a predeclared Qwen v2 run if the lower bound of the 95% interval is above 0. Otherwise stop, and v2 stays on Gemma 4.
- **Also reported, not deciding:** per-type CC, NLL and Brier, and single-pass latency.

A full Qwen run would need its own predeclaration, including how final_test is used. v2 base, v2 large and v2 small have each read final_test once already.

## Budget

One vast.ai RTX PRO 6000 S at about $1.46/h, from the current credit of $8.43 without a top-up.

| Stage | Expected time |
|---|---|
| Setup, upload, backbone fetch (9.3 GB), kernel check | 0.3–0.4 h |
| Training | 1.0–1.5 h |
| Single-pass reads | 0.1 h |
| Archive, download | 0.1 h |
| **Total** | **about 1.5–2 h, about $2.5–3** |

- **Training estimate:** E4B trained in about 2.0 h while sharing the GPU with E2B.
- **Guardian:** stops compute at 4 h, at most about $5.9.
- **Admission:** run_direct's credit admission (safety factor 1.25) fits within the credit.

## Integrity

- The bundle's private test split is not opened.
- final_test is not read.
- No dataset rows are committed; reports contain aggregates only.
