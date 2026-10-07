# Direct Noul calibration trial, 2026-10-07

The trained v2 Large still abstains too often relative to its matched v1
control. A calibration-only trial improved probability quality but did not
reduce development abstentions or increase the headline score. The candidate
was rejected for serving. No model weights, saved head temperatures, or
optimizer state were updated, and no GPU instance was rented for this trial.

## Frozen fit and independent validation

The checkpoint identity remains
`752534912fd83f3753b9b8d25e97cd66bc68a686faf371e321417818f20e3f01`.
Its adapter was trained for 74 updates on 2,368 gold-only direct questions.
The existing Noul temperature is approximately 0.68266922. Refitting that
same scalar on the same data would not introduce a new correction.

The new fit uses only the original reserved calibration split: 64 Noul
questions from 43 independent cases. The canonical binary margin is
`raw_logit_true - raw_logit_false`; the correction is
`sigmoid(scale * margin + bias)`. The slope stays positive, while the bias
can correct a false/true asymmetry. Choice and Score retain their scalar
temperatures and their dynamic candidate semantics.

Two fixed regularization strengths, 0.01 and 0.05, were checked with five
source folds. Every question from one case remains in the same fold.
Selection required lower NLL, non-increasing Brier, non-decreasing served
credit, and fewer abstentions at the unchanged 0.2/0.8 thresholds.

| Calibration out-of-fold result | NLL | Brier | Served credit | Abstentions |
|---|---:|---:|---:|---:|
| Temperature reference | 0.338029 | 0.105160 | 60.9375% | 23 / 64 |
| Affine, regularization 0.01 | 0.326962 | 0.095862 | 62.5000% | 22 / 64 |
| Affine, regularization 0.05 | 0.325753 | 0.099573 | 59.3750% | 24 / 64 |

Only 0.01 passed all selection conditions. It was refitted on the complete
calibration split, producing scale **1.5430069725604798** and bias
**0.5011419993194093**. Both values were frozen before applying the candidate
to development outputs. Development targets were never used to fit or
adjust the parameters. Calibration and development case identities are
disjoint.

## Development outcome

This is exact CPU postprocessing of the previously saved GPU probabilities,
using the original per-question full input lengths and the saved head
temperatures. It does not require another 12B model run. Raw comparison rows
were checked against their original canonical cohort and checkpoint protocol
before replay. It measures probability changes; it is not a new latency
measurement or an official JevBench run.

| Noul, same 64 development questions | Existing policy | Frozen candidate |
|---|---:|---:|
| NLL, lower is better | 0.359715 | 0.335121 |
| Brier, lower is better | 0.108997 | 0.098873 |
| ECE, lower is better | 0.111311 | 0.080224 |
| Abstentions | 18 | 19 |
| Served credit | 65.625% | 65.625% |
| Diagnostic argmax credit | 87.500% | 87.500% |
| Local Noul competence | 31.250 | 31.250 |
| Overall equal-type local competence | 57.929 | 57.929 |

The candidate repaired two thresholded classification results and broke two.
The overall score change is zero, with a source-bootstrap 95% interval of
[-4.233, 4.104] points (2,000 replicates, 99 independent development cases).
Fourteen of the original 18 abstentions have a correct diagnostic argmax;
that diagnosis does not prove that increasing every confidence is calibrated.

The independent validation rule rejects this candidate because abstentions
increase. Its rejected artifact retains the frozen parameters and audit
results but acts as an identity transformation when loaded. The original
serving checkpoint is unchanged. A separate small JSON report contains the
measured summaries and provenance:
[noul-calibration-20261007.json](results/noul-calibration-20261007.json).

The original matched scores consequently remain:

| System | Local competence |
|---|---:|
| Published v1 with its frozen reasoning policy | 63.088 |
| Trained v2 Large, reasoning off | 57.929 |
| Trained v2 Large, forced medium reasoning | 77.874 |

The direct correction never changes valid reasoned reads. The goal of v2 off
outperforming v1 on remains unmet. More confidence alone is not an adequate
remedy for that result.

## Automatic behavior for future training

Both the general training runner and the v2 direct runner fit temperatures
from reserved raw calibration logits, then evaluate this Noul correction
automatically. The general runner reserves calibration first from the user's
own corpus, then a separate evaluation split, keeping whole state/lineage
groups together and retaining training data even for a small corpus.

The fitted candidate must pass source-fold selection and a separate holdout
validation. Validation can reject it but never changes its parameters. At
least 20 independent cases are required in each fitting fold and in the
validation holdout. Sparse data, failed guards and unvalidated routes keep
the existing temperature policy. Reports include raw and calibrated metrics,
unfitted types, source-group counts, and the correction status.

Checkpoint save/load and native serving apply an accepted correction once,
without a separate user command. The correction binds to exact checkpoint
weights, saved temperatures, configuration and the serving recipe. Explicit
path calibration takes precedence. Reasoned and image reads do not reuse a
direct-text correction. Resumed training clears the old correction; new
weights require a fresh fit. Merged standalone exports retain the artifact,
while lossy adapter compaction or quantization requires a new calibration.

Gold NLL and Brier objectives already train useful probabilities. Proper
losses, representative data, and source-separated evaluation can reduce
the required correction, but no training objective guarantees calibration
under every future data distribution. Automatic fitting removes the manual
workflow without making that unsupported guarantee.

## JevBench rank and next quality work

This checkpoint has no official rank measurement. The latest
[Benchmark Heaven board](https://www.benchmarkheaven.com/jev-models) checked
on 2026-10-07 uses JevBench v1.6.1, with 1,200 sealed and 300 public decisions
per self-hosted system. Its scoring round uses option B / O1S. The four-axis
Composite combines Intelligence, Calibration, Speed and Cost; Capability
combines Intelligence and Calibration within the published eligibility caps.
Our 376-question development diagnostic uses the older local competence
method and a different cohort, so neither 57.929 nor 77.874 can be translated
into a leaderboard position. The
[official model guide](https://www.benchmarkheaven.com/jev-models/how-to-choose)
also distinguishes Intelligence from the composite.

A paired-model transfer between benchmark releases can support a conditional
estimate once Ayaka is anchored to the old JevBench scale. The
[paired version analysis](JEVBENCH_VERSION_TRANSFER_2026-10-07.md) measures
that transfer and identifies the additional public-reference evaluation
needed to connect this separate development corpus to it.

The next calibration-only work should use more independent, representative
Noul cases and fresh validation cases, without retuning this candidate on the
now-observed development answers. Reasoned outputs need their own reserved
calibration reads at the deployed effort/budget; the direct temperatures have
not been independently validated for that route. This is the next useful
weight-free direction for the reasoning model's probability losses.

If weight training is considered later, the direct route needs balanced hard
Noul and the verified/CommonsenseQA areas that regressed against v1, rather
than a universal confidence increase. For reasoning, focus on the
ContractNLI Choice regression (approximately 4.41 local competence points)
and independently track NLL, Brier and ordinal Score RPS. On the existing
comparison, reasoned NLL worsens for all three types even though local
competence rises. A reasoning trace must improve the final distribution as
well as its selected answer. No such weight retraining was started here.
