# JevBench v1.5 board: scoring details that drive the Swift design — 2026-10-04

This note collects the scoring facts and board observations behind the
[Swift readout plan](AYAKA_V3_SWIFT.md). Everything comes from public sources: the live board snapshot
`runs/jevbench-audit-20261004/live-v1.5.6-20261004.json` (v1.5.6, captured 2026-10-04), `docs/METHOD-v1.5*.md`
in the JevBench repository and public GitHub issues. None of it is an Ayaka measurement.

## Headline and axes (METHOD v1.5 + headline amendment A)

- **Composite A**: the equal-weight harmonic mean of Intelligence, Calibration, Speed and Cost. If Intelligence,
  Speed or Cost is below 50, the composite is multiplied by (axis/50)².
- **Intelligence**: 50 % open and 50 % sealed. Choice, Noul and Score each weigh 1/3, and the tier weights are
  easy .10, standard .20, judge .30, hard .40. Values are chance-corrected per item. The overfit penalty applies to
  the open−sealed gap only beyond the field median (G_med 5.19) plus 8 points.
- **Noul**: P(yes) ≤ .20 counts as No and P(yes) ≥ .80 as Yes. **Everything in between is an abstention, scored as
  wrong.**
- **Score**: the prediction is the *expected position* of the returned distribution, scored as nMAE against the
  uniform-guess nMAE.
- **Calibration**: Choice, Noul and Score each weigh 1/3, pooled over open and sealed items. Values below were
  reverse-engineered from board rows and reproduced to four decimals in `ayaka/swift/score.py`:
  - Choice part = (100·(1 − ECE/0.5) + 100·(1 − mean TVD)) / 2. **The ECE uses hard-tier Choice items only**
    (`n_ece_hard` 277 = 169 open + 108 sealed). The TVD uses the 50 items that carry an exact gold distribution.
  - Noul part = 100·(1 − ECE/0.5), with ECE on P(yes) in 10 bins.
  - Score part = (100·(1 − nRPS) + 100·(1 − top-label ECE/0.5)) / 2.
- **Speed**: the mean of score(p50) and score(p95), where score(s) = 100 − 20·log10(s/0.1) after the self-host
  adjustment (2×s + 0.15 s). **The sample is a serial run over open standard and judge items only.**
- **Cost**: 100 − 30·log10($ per 1,000 decisions / $0.001). Tokens are pooled over all decisions. Prices use a
  30-day public list price or, failing that, a base-model reference estimate.

## Board on 2026-10-04 (headline A)

| # | System | A | I | C | S | Cost |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Cygnet (frozen gemma-4-12B-it, letter readout, T = 3.4) | 73.70 | 71.1 | 87.0 | 91.0 | 56.4 |
| 2 | Winnow-12B Q8 | 73.23 | 74.4 | 84.1 | 86.1 | 56.6 |
| 3 | Jev 1.13.0 | 72.13 | 72.0 | 88.0 | 83.8 | 54.7 |
| 4 | JevK5 v0.3 (4B) | 71.90 | 56.3 | 88.3 | 93.6 | 63.1 |
| 5 | Vansa-3.4 | 71.59 | 58.0 | 87.6 | 91.3 | 61.4 |

Ayaka v1 (issue #132) was not on the board yet. Queued requests include Hopper 12B (LoRA on gemma-4-12B-it with a
letter readout) and Nadir Scout 12B (Cygnet recipe reproduction).

## Observations that shaped Swift

1. **Base size is bounded by Cost.** Larger bases sit at or below the Cost gate: Qwen3.8-Flash-Next ≈ 42 (swanOne
   has I 71.2 and still ranks 31st), 26B-A4B ≈ 49. Qwen3.5-9B rows range from 45.7 to 58.9, depending on their
   tokens per decision. 4B-class bases get Cost ≈ 63, but
   the best measured 4B Intelligence is 62, and frozen Qwen3.5-4B (SemIf) reaches only 51.3.
2. **Every top system abstains on Noul.** Abstention rates are 13–28 %: Cygnet .139, Winnow .131, Jev .228,
   Vansa .283. On Cygnet's published public reads, moving in-band values to .199/.801 raised Noul CC from 59.5 to
   75.7, and ECE fell from .070 to .054. The sample was 74 items, so this is a diagnostic only. The same rule
   *worsened* Noul ECE on ayaka-base and v2 pilot reads
   ([saved-policy probe](V2_SAVED_POLICY_PROBE_2026-10-04.md)). That is why Swift fits the commit band rather than
   fixing it.
3. **The Speed sample excludes the hard tier,** which carries 40 % of Intelligence. A reasoning route that fires on
   hard-style calculation items but rarely on standard and judge items can raise Intelligence without moving
   p50/p95.
4. **Prompt scaffolding matters.** Cygnet's shim records that a bare prompt cost about 10 Intelligence points.
   Swift therefore selects among prompt variants on non-public dev reads.
5. **Choice calibration is half TVD.** On the 50 exact-distribution items, top systems sit at TVD .29–.32. It is a
   large, shared weakness, and temperature alone moves it little.
6. **Only Speed and Cost depend on the serving GPU.** The evaluator measures on its own pod (Cygnet: RTX PRO 6000),
   so local latency only estimates the Speed axis.

## How to submit

Requests are GitHub issues on `fstandhartinger/jevbench` titled `[bench request]: …`. They give a pinned package,
run commands, the public-set result from the JevBench CLI and its identity check, and the disclosures; Cygnet's
issue #71 is the model. The evaluator makes benchmark claims only after its own run.
`deploy/swift/README.md` follows this format, and its measured fields stay TBD until our GPU collection.
