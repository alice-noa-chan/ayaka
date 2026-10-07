# Why v2 direct trails v1 and where the abstentions come from, 2026-10-07

CPU reanalysis of the saved observations behind the
[trained checkpoint comparison](TRAINED_CHECKPOINT_COMPARISON_2026-10-07.md).
No model was run, nothing was fitted, and no threshold, prompt or temperature
changed. All numbers are local development CC on 376 questions from 99
independent cases, not official JevBench scores.

Machine-readable report:
[trained-checkpoint-gate-breakdown-20261007.json](results/trained-checkpoint-gate-breakdown-20261007.json).

## Summary

1. **The v2 direct deficit comes almost entirely from 27 questions (7%) where
   v1 used worked steps.** The declared comparison "v2 off beats v1 on" asks a
   single direct read to beat reasoning on calculation questions. On the 349
   questions both systems read directly, v2 already scores higher than v1.
2. **Most v2 direct abstentions are calculation questions that a direct read
   cannot compute.** Their probabilities sit near 0.5. This is honest
   uncertainty, which is why the calibration-only trial could not remove them.
3. **Reasoning gains are concentrated in the synthetic source the reasoning
   traces were trained on.** On natural sources reasoning is flat or worse,
   and its probabilities are overconfident on every type.
4. **The remaining abstentions are Korean and Japanese intent questions with a
   "false" lean**, the same direction measured for the frozen reader on
   October 5.
5. **The development cohort is too small to confirm the gains we are looking
   for.** One Noul question moves the equal-type headline by about 1 point,
   and the screen requires 200 independent cases while this cohort has 99.

## 1. Split by v1's frozen reasoning gate

v1's published policy reads 349 questions directly and reasons on 27 that its
calculation gate selects. The two strata behave very differently:

| Stratum | Questions | v1 on | v2 off | v2 on |
|---|---:|---:|---:|---:|
| v1 read directly | 349 | 63.03 | **64.23** | 76.16 |
| v1 reasoned | 27 | 62.22 | **-2.22** | 100.00 |
| All (declared result) | 376 | 63.09 | 57.93 | 77.87 |

Equal-type CC within each stratum. The reasoned stratum contains only Choice
and Noul questions.

Within the direct stratum, v2 gains 9.5 Choice CC (80.24 against 70.75) and
loses 7.4 Noul CC (48.15 against 55.56); Score is 64.31 against 62.78.
Classification credit on the direct stratum is 86.3% for v2 against 79.8%
for v1 on Choice, and 74.1% against 77.8% on Noul.

On the reasoned stratum, v2 direct abstains on 7 of the 10 Noul questions.
v1's worked steps answer 8 of them correctly. v2's own forced reasoning
answers all 27 questions correctly.

### Composed diagnostic: v2 direct plus reasoning where v1's gate reasons

Replacing v2 direct reads with v2 reasoned reads on exactly the 27 gated
questions gives:

| System | Equal-type CC | Choice | Noul | Score | Noul abstentions |
|---|---:|---:|---:|---:|---:|
| v1 on (published) | 63.09 | 70.24 | 56.25 | 62.78 | 7 |
| v2 composed | **67.47** | 81.85 | 56.25 | 64.31 | 11 |

Against v1 on: CC +4.38, paired 95% interval [-2.61, +10.96], classification
credit +6.06 points, mean per-question NLL gain +0.177 (2,000 replicates,
seed 15, 99 cases). Noul NLL improves from 0.343 to 0.255 and Noul ECE from
0.122 to 0.078.

This is not a deployable v2 policy and not a proven gain: it borrows v1's
gate decisions, which depend on v1's own baseline confidence, and its interval
crosses zero. It does show that the structure "v2 direct plus a selective
reasoning route" removes most of the declared deficit, while forced
reasoning on every question costs about 17 s per question at p50.

## 2. What the v2 direct abstentions are

v2 direct abstains on 18 of 64 Noul questions. By source:

| Source | v1 on | v2 off | v2 composed |
|---|---:|---:|---:|
| ayaka-v2-verified (synthetic procedural) | 0 | 11 | 4 |
| MASSIVE Korean (intent) | 3 | 5 | 5 |
| MASSIVE Japanese (intent) | 4 | 2 | 2 |

The 11 verified abstentions are leap-year, business-day, time-zone,
month-end and rounding questions. Their P(true) values range from 0.30 to
0.60, and 3 of the 11 argmax directions are wrong. A single forward pass
cannot do this arithmetic reliably, and the model reports that. Raising
confidence would convert some abstentions into wrong assertions; this
matches the rejected calibration-only trial, where abstentions went from 18
to 19.

## 3. Reasoning helps the synthetic source, not natural sources

Classification credit (Choice/Noul) or 1 - nMAE (Score), and mean NLL:

| Source | Type | n | v2 off credit / NLL | v2 on credit / NLL |
|---|---|---:|---:|---:|
| ayaka-v2-verified | Choice | 32 | 84.4% / 0.39 | 100.0% / 0.01 |
| ayaka-v2-verified | Noul | 32 | 59.4% / 0.44 | 100.0% / 0.00 |
| ayaka-v2-verified | Score | 32 | 91.6% / 0.26 | 99.7% / 0.01 |
| CommonsenseQA | Choice | 32 | 68.8% / 0.68 | 78.1% / 0.94 |
| ContractNLI | Choice | 136 | 89.0% / 0.27 | 86.0% / 0.55 |
| HelpSteer2 | Score | 80 | 84.3% / 1.12 | 84.5% / 1.97 |
| MASSIVE Japanese | Noul | 16 | 81.2% / 0.24 | 93.8% / 0.14 |
| MASSIVE Korean | Noul | 16 | 62.5% / 0.32 | 68.8% / 2.01 |

The verified source is generated by the same templates as the verified
traces v2 was trained on, so its near-perfect reasoned score is in-distribution.
On ContractNLI and HelpSteer2, the largest natural sources, reasoning does not
improve accuracy and doubles NLL. On Korean intent questions it produces
confident wrong answers (NLL 2.01). Half of all reasoned reads put at least
0.99 on one candidate (v2 off: 9%), and reasoned Score ECE is 0.31.

This agrees with the October 5
[paired teacher diagnostic](V2_PAIRED_TEACHER_DIAGNOSTIC_2026-10-05.md):
raw worked steps help synthetic and verified questions and hurt natural
HelpSteer2. The reasoned route reuses direct-read temperatures that were never
validated for it.

## 4. Remaining abstentions: Korean and Japanese intent

In the composed system, 7 of the 11 abstentions are MASSIVE Korean/Japanese
questions, and all 7 have gold "true". v2 direct's mean P(true) is 0.42 over
64 Noul questions whose true rate is 0.50. The frozen `min` reader showed the
same lean on October 5 (mean 0.43 against 51% true;
[decomposition](SWIFT_NOUL_DECOMPOSITION_2026-10-05.md)). v1 also abstains on
these questions, so they are not a v2 regression, but they are the main
abstention source left once calculation questions are routed.

## 5. Statistical resolution of the development cohort

- Noul contributes one third of the equal-type headline from only 64
  questions. One Noul question changes Noul CC by 3.125 points and the
  headline by about 1.04 points.
- Paired intervals on this cohort have half-widths of about 7 to 9 CC. The
  declared target, a 5-point direct gain, is smaller than that.
- The screen requires at least 200 independent cases; this cohort has 99, so
  it cannot pass regardless of the result.
- The cohort is 36% ContractNLI. Choice CC, and therefore one third of the
  headline, mostly measures one legal source.

## 6. Latency observation

Per-question collection latency at p50 was 0.18 s for v1 direct reads and
2.13 s for v2 direct reads, on the same GPU and the same native HF runtime.
v2 reads use the `swift_canonical` input contract, which skips the shared
prompt-head cache; that alone does not explain a 12x difference. The cause is
not established from saved data. It matters for the Speed axis and needs a
profiled run before any serving claim.

## Recommendations, in order

1. **Change the development question from "v2 direct beats v1 with
   reasoning" to two matched comparisons:** v2 direct against v1 direct, and
   v2 with a selective route against v1 with its selective route. The
   current rule compares different inference budgets on the calculation
   questions.
2. **Fit and validate a v2 reasoning router** on reserved calibration reads.
   `auto` mode currently never reasons without a validated router. The
   composed diagnostic suggests this is where most of the remaining gain is,
   at about 7% of questions reasoned instead of 100%.
3. **Fit reasoned-route temperatures** on reserved reasoned calibration reads
   at the deployed budget before any reasoning result is used for
   Calibration-axis claims.
4. **Build a development cohort with at least 200 independent cases** and a
   Noul share large enough that one question does not move the headline by a
   full point. Report natural and synthetic sources separately.
5. **Target the Korean/Japanese Noul false lean with data, not thresholds.**
   The October 5 decomposition showed a threshold lever does not transfer.
6. **Profile v2 direct latency** before the next GPU comparison.

## Reproduce

From the repository root, with the retained raw observations:

```bash
R=.dev/v1v2-compare-20261007/retrieved/results
python -m ayaka.eval.checkpoint_comparison breakdown \
  --samples $R/dev.jsonl --protocol $R/protocol.json \
  --v1-on $R/v1_on.jsonl --v2-off $R/v2_off.jsonl --v2-on $R/v2_on.jsonl \
  --replicates 2000 --out gate-breakdown-recomputed.json
```

The command validates every row against the canonical cohort and protocol
before scoring and refuses to overwrite an existing output.
