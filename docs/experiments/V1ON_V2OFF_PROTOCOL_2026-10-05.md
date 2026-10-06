# v1 reasoning on vs v2 reasoning off: fixed comparison protocol (frozen 2026-10-05, amended before any v1 row)

## Goal

The goal is to test the project claim "v2 reasoning off beats v1 with its reasoning route"
(`V2_DIRECT_OVER_V1_REASONING_2026-10-04.md`). Both systems run on the same non-public items and are scored by one
scorer. The success rule is the one that plan already stated. It is applied once and not changed after the result.

## Systems

| System | What runs | Generated tokens |
|---|---|---|
| **v2 off** | Frozen gemma-4-12B-it (`707f0a3b…`), Swift `min` prompt, canonical-letter raw readout, saved production policy (`gate/google_gemma-4-12B-it/policy.json`) with the reasoning route removed | 0 |
| **v1 on** | Published `alice-noa-chan/ayaka-large` at `267605ee…`, unmerged LoRA, `reasoning_decision` with `FROZEN_REASONING_POLICY` (calculation gate, baseline cutoff 0.9, 384 new tokens). This is the same path as `ayaka.eval.jevbench --ckpt … --reasoning` | worked steps where gated |
| v1 off (diagnostic) | v1's own single pass, recorded inside the same run | 0 |

v1 runs with `--max-seq-len 8192`, a longer context than its 4,096-token historical evaluation. The longer context
is meant to avoid truncating v1. Its net effect on v1 is not measured here, so no direction is claimed.

## Cohort (1,265 decisions, all non-public, none used to fit either system's parameters)

| File | Decisions | sha256 | Notes |
|---|---:|---|---|
| `procedural.jsonl` | 479 | `abf80932e93b1236296af3e79c399b6ce1b517bbe3ffe7f4eda6bbb31db4c92d` | 160 samples of the six repository generators at seed 20261005 (`build_procedural_cohort.py`) |
| hard `calibration.jsonl` | 390 | `01f17bc9…b261b58c` | QuALITY, HotpotQA, HelpSteer2 validation (hard dev protocol) |
| hard `dev.jsonl` | 396 | `8a237769…a07ee` | same |

How the procedural items were checked:
- v1 large trained on these generators with seed 0. Each generator's full seed-0 draw was compared with the new items
  and no state collided.
- 13-gram decontamination against the JevBench public items dropped nothing.
- The items come from the generator family v1 trained on, and they are where v1's calculation gate fires. The
  effect of that on the comparison is not measured, so no direction is claimed.

How the hard items relate to both systems:
- v2's policy was fit on v2 calibration only. The hard set was used for the `cygnet` prompt decision: `cygnet` was
  rejected and `min` stayed. The hard part is therefore **exposed to v2 prompt selection**. This is a matched
  comparison on an exposed cohort, **not a fresh independent confirmation**.
- v1 never saw these items.
- HotpotQA rows that share a paragraph are statistically dependent. Rows that share a case or any HotpotQA paragraph
  title, across both hard files, are closed into one component, and the bootstrap resamples components (rule 3).

## Success rule (decides; applied once)

All four must hold on the full cohort. Δ means v2 off minus v1 on.

1. Thresholded Choice/Noul gold credit: **Δ ≥ +5 points**. Choice uses argmax; Noul uses ≤0.2 / ≥0.8, with values
   in between counted wrong.
2. Equal-type chance-corrected score, the mean of the per-type CC: **Δ ≥ +5**.
3. The bootstrap 95% CI lower bound of (2) is **above 0**. The bootstrap uses B = 2000, seed 15, and resamples
   dependence components: each procedural sample is one component; hard rows are closed by shared case or HotpotQA
   paragraph title. On the frozen hard files this gives 507 components from 533 cases.
4. Score normalized RPS for v2 off is **not worse** than for v1 on.

**Also reported, deciding nothing:**
- tier-weighted I, C, by-source and by-type results;
- v1 off;
- v1 route counts and errors.

**Speed and Cost are not compared:** v1 runs on HF eager and v2 on vLLM, so their latencies are not comparable.

## Execution

- Script: `scripts/swift/v1v2_job.sh`. The host gets offline-verified input files, which are hash-checked again
  before use.
- The host first collects Swift `min` reads on the procedural file with the attempt 7 recipe. The implementation
  hashes are unchanged.
- It then prepares the v1 checkpoint and runs `scripts/swift/v1_on_runner.py` on all three files.
- The job fails fast and logs DONE only after exact unique counts.
- Scoring: `scripts/swift/v1v2_compare.py`. It refuses a missing, extra, duplicate or mismatched row:
  - every row of both systems must bind the pinned cohort's state, full question, gold, soft gold, source, tier and
    case;
  - every v2 row must be a frozen Swift `min` direct read;
  - every v1 row must come from the single pinned run contract (checkpoint revision and files, context 8192, frozen
    policy).

  The runner refuses resuming a row whose run or input differs. The success field is named
  `v2_off_beats_v1_on_on_this_cohort`.
- **What a failure means:** v2 off has not yet beaten v1 on, and the gap is reported as measured. The rule, the
  cohort and the systems are not changed afterwards.

## Amendment log (before any v1 row)

- 2026-10-06, after Codex review (replies §34–35): no measurement existed yet.
  - Removed the directional claims about context length and in-distribution items.
  - Named the cohort exposed (not a fresh confirmation).
  - The bootstrap now resamples shared-paragraph components instead of case clusters.
  - The comparator and runner check every row's semantic binding and run contract.
  - The v1 runner takes the pinned checkpoint receipt.
  - The success thresholds, cohort files and systems are unchanged.
