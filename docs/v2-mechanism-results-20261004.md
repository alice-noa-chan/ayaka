# Frozen v1 reasoning mechanism results — 2026-10-04

Published v1 benefited from generated worked steps on this small English dev
cohort. The existing hybrid readout did not show a disadvantage against the
native LM readout in the measured reasoning interaction. Generated reasoning
still made Noul probabilities worse, and verified reference notes performed
substantially better than generated notes. These observations support improving
trace generation and uncertainty handling before another training run. They do
not establish a release improvement or identify the cause of the failed v2 SFT.

## Completed scope

Follow the [frozen protocol](v2-mechanism-protocol.md) and its execution amendment.
The completed run evaluated **v1 only**, without changing any weights or fitting
calibration. Twelve preselected underlying English dev cases produced 36 typed
questions: 12 Noul, 12 Choice, 12 Score. Each question had five contexts and three
readouts, giving 540 scored distributions. No independent test was opened.

Contexts were direct, empty matched reasoning context, greedy generated worked
steps (512-token cap), verified reference derivation, and another case's
derivation from the same rule. Reference notes were recomputed from visible
facts and checked against the prepared targets. They contain privileged answer
information; their scores are **not a deployable model capability**.

All three readouts used the same hidden states for a given context: native LM,
the inherited trained Set Mixer pointer, and the inherited hybrid gate. The
pointer was not independently trained as a standalone head and is not Jeeves's
plain pointer. Direct hybrid reproduced production cached probabilities with
maximum absolute error **0.0**. Question caches were isolated, and controls used
the generated trace's actual closing token, or remained unclosed when capped.

## Measured results

Scores below are local, equal-type **chance-corrected competence points**.
Zero means chance; negative values are possible. They are not accuracy
percentages, published v1 benchmark estimates, or official sealed-inclusive
JevBench scores. All probabilities use raw readouts, without temperature fitting.

| Context | Inherited hybrid | Native LM | Inherited pointer |
| --- | ---: | ---: | ---: |
| Direct | -11.74 | -15.71 | -14.32 |
| Empty matched context | -9.56 | -9.56 | 10.43 |
| Generated worked steps | 45.06 | 45.06 | 47.86 |
| Verified reference notes (privileged) | 81.59 | 81.64 | 80.72 |
| Other-case notes (qualified below) | 41.40 | 41.49 | 47.30 |

Hybrid generated-versus-direct gain was **56.80 points**, with exploratory 95%
case-paired bootstrap interval **[19.66, 91.68]**. Versus the empty matched
context, the gain was **54.62 [25.83, 83.83]**. Thus the measured improvement was
not explained solely by switching to the reasoning prompt/suffix.

The native-LM gain minus hybrid gain, both relative to the empty control, was
**0.0073 [-0.1772, 0.1541]** for generated notes and
**0.0469 [-0.0829, 0.1654]** for reference notes. This experiment did not detect a
hybrid handicap for these interventions. It does not establish universal head
equivalence or show what a separately trained Jeeves-style pointer would do.
Using direct-relative gains alone would confuse reasoning interaction with the
different direct baselines.

Intervals resample the 12 underlying cases, keeping their typed questions
together, with 2,000 replicates and seed 15. This is a small, previously used dev
domain with multiple comparisons; the intervals do not constitute a release gate.

### Typed outcomes and probability quality

| Hybrid metric | Direct | Generated | Reference notes |
| --- | ---: | ---: | ---: |
| Choice competence | 4.92 | 64.35 | 88.12 |
| Noul competence | -66.67 | 0.00 | 83.33 |
| Score competence | 26.51 | 70.83 | 73.32 |
| Choice NLL (lower is better) | 1.9831 | 1.3070 | 0.1444 |
| Noul NLL (lower is better) | 1.0207 | 2.8664 | 0.2126 |
| Noul Brier (lower is better) | 0.3519 | 0.4511 | 0.0278 |
| Score RPS (lower is better) | 0.19781 | 0.06061 | 0.00293 |
| Noul abstentions / 12 | 7 | 0 | 0 |

Relative to direct, generated hybrid fixed 12 classification decisions and broke
2. These counts cover Noul and Choice under the evaluator's correctness
threshold, not Score. Score nMAE improved on 7 questions and worsened on 3;
2 were unchanged within 1e-6. Mean NLL across all three types improved by 0.0747,
but **Noul NLL and Brier worsened**: better headline decisions do not imply better
uncertainty estimates. The evaluator's mixed `proper_loss` rose from 1.0672 to
1.4113; this combines Choice/Noul NLL with Score RPS and should not be interpreted
as a uniform-unit probability loss.

| Family (4 underlying cases each) | Direct hybrid | Generated hybrid | Reference hybrid |
| --- | ---: | ---: | ---: |
| Numeric | -23.24 | 81.66 | 99.82 |
| Dates / temporal numeric | 8.04 | 23.77 | 97.45 |
| Rules / exceptions / missing information | -16.40 | 32.02 | 47.58 |

Date generation retained a large gap to supplied correct notes. This is four
cases, and two business-day traces reached the cap after repetitive generation;
it is not a broad date-accuracy estimate or proof that a larger budget fixes it.

### Concrete generation failure

Case `independent-recovery-1/dev/142/en` requires all of: not cancelled, cost
within limit, and override or sufficient credential. Its visible facts include
`cancelled=False`, `cost=861`, `limit=669`, `override=True`, `credential=4`, and
`required=3`. Override does not waive the cost condition. The approval result is
therefore 0, and the Noul question "Is the result 1?" has target false.

The model generated **"The result is 1."**, then ended normally after 7 tokens.
All three generated-context readouts were wrong. Hybrid assigned true probability
**0.992302**. With the recomputed reference note
"Not cancelled = True; cost within limit = False; override or sufficient
credential = True. All three are required; approval value = 0.", all three
readouts were correct; hybrid assigned false probability **0.998620**.
This intervention demonstrates that the reader can use a correct derivation for
this example. It does not establish that the generator can produce it reliably.

### Other-case control limitation

The cyclic control was fixed before scoring. **8 of 12 other-case derivations
have the same final answer as the target case**; 2 have identical note text,
affecting 6 typed questions. The aggregate other-case score consequently cannot
be treated as a strong wrong-answer control or proof of robustness.

Only the four numeric cases have different final answers in this control. On
that descriptive subgroup, hybrid competence was -23.24 direct, -10.73 empty,
81.66 generated, 99.82 reference, and **-0.67 with the other-case note**. Incorrect
supplied notes hurt sharply relative to correct notes on these four cases. This
does not estimate general resistance to misleading reasoning. The control was
not replaced after seeing results.

## What remains unresolved

The basic idea of generating a trace and then making a typed decision was useful
on the completed frozen-v1 cohort. These data provide no basis to blame an
inherent incompatibility with the inherited hybrid head. The observed remaining
issues include wrong or repetitive traces, Noul overconfidence, and prompt
effects specific to the existing pointer. More evidence is needed to rank their
importance outside this cohort.

This is not a reproduction of the full
[Jeeves recipe at `3f948dec`](https://github.com/PostHog/jeeves/tree/3f948dec68187ed3ced9152ed3d84b73e498665c).
Its [head](https://github.com/PostHog/jeeves/blob/3f948dec68187ed3ced9152ed3d84b73e498665c/model/head.py)
and [trainer](https://github.com/PostHog/jeeves/blob/3f948dec68187ed3ced9152ed3d84b73e498665c/trainer.py)
use a different backbone, plain pointer, SFT and CISPO reinforcement learning.
Ayaka requests ordinary worked steps with `enable_thinking=False`; native Gemma
thinking, 1,024-token forced high, RL, and independently trained heads were not
tested. The previously failed clean pilot was not evaluated in the completed
mechanism cohort. Its reasoning-off regression cannot by itself falsify the
reasoning principle, and this v1 experiment cannot isolate its training cause.

An appropriate next comparison would freeze the same v1 reader and test
generation format / repetition control and reasoning-specific Noul calibration
on a fresh predeclared dev cohort, before using independent test for a release
decision. That is a recommendation, not another job launched or a predicted win.
No retraining, model publication, v1 replacement, or Top 5 claim follows here.

## Execution, costs, and closure

Three explicitly admitted tasks were used; automatic retries were disabled.
The first two saved incomplete receipts after one v1 question each. Their rows
are not pooled as additional observations, and neither evaluated the pilot.

| Task | Outcome | Actual credit used |
| --- | --- | ---: |
| Initial paired plan | Invalid fixed-overhead extrapolation rejected continuation | $0.285578 |
| Corrected paired plan | Full-cap forecast 4,275.668 s exceeded available 4,105.071 s | $0.292934 |
| v1-only plan | Complete 12-case / 36-question / 15-arm comparison | $0.692772 |
| Total | Includes both incomplete attempts | **$1.271284** |

The initial timing error was an implementation mistake: fixed prefill/readout
overhead was scaled by 512/7 along with decode. Commit `90d6520` separated fixed
overhead from synchronized per-token decode time. After the corrected paired
plan still failed admission, commit `5f4d70d` reduced checkpoint scope to v1
before a complete comparison. Cases, questions, contexts, readouts and the
512-token budget stayed fixed. There is no paired v1/pilot result.

The final Beam RTX5090 task was
`bebf70bf-ba60-4cd3-9697-f5f75c0e62b8`, started 2026-10-03 15:08:09.906955 UTC,
and ended 15:27:43.372585 UTC: **19 min 33.47 s** total task duration. Worker time
was 1,171.079 s. Its backend limit was 3,400 s and conservative admission cap
$2.35 compute plus $0.10 margin. This includes staging/model loading, not just
decode latency; it is not a production p95 benchmark.

Generation used **7,292 physical tokens** (counted once, not once per head),
with 34 normal EOS stops and 2 length stops. The capped questions were
`independent-recovery-1/dev/30/en/choice` and `/score`.

After downloading all three receipts and verifying every returned file, all
three tasks were stopped. The final container list was empty. Only the owned
temporary volume `ayaka-mechanism-20261003` was deleted; the three unrelated
volumes were preserved. Actual charges stayed within the existing credit, with
cash spend $0.00. No top-up or account-limit change was made. The exact wallet
observations and local billing screenshot are retained outside Git.

## Reproducibility and validation

The [aggregate JSON receipt](v2-mechanism-results-20261004.json) carries the exact
metrics, paired intervals, control audit and identities. Raw rows, traces,
failed receipts, delivery manifests and cleanup evidence remain locally under
`runs/ayaka-mechanism-20261003/`; they are not model release artifacts.

| Identity | Value |
| --- | --- |
| Execution code | `5f4d70d42e2ff5fae5ea2449e64261a07dc74166` |
| Parent checkpoint fingerprint | `26109de64d2e6417c64a7e0942ac2d18b43208f7d870ffabf7c8fb814ad606d2` |
| Backbone revision | `ee0ef6023621cff504d758262d4e04895a5af4a2` |
| Dev source SHA256 | `5722121aed02da74b5f5ddfae1cf0d92214f28b04cd6f51c75abd9d5ab08578c` |
| Cohort SHA256 | `af49fde0dd3f2202779400b5530a7034c4143bfc9b20a48be2e1853863e21b37` |
| Plan SHA256 | `79fa44ceccd452ef015a75898691a803f200defe893d05e296367e3cc60b2c5b` |
| Overlay SHA256 | `5493493dedf784f6c5299cf372607169571d925bfc0f256c0ba098c2e8370ca1` |
| Completed archive SHA256 | `5ed1c914d32fc38c2ad6dd280e234458d8782cc51b771348709c0ae7bf97d78d` |
| Raw `mechanism/parent.json` SHA256 | `9bb4b8440a1ca0a3ade5b23bcafb877ecd4fb76789b8b5ff5f7437148adf2820` |

The completed zstd-level-3 archive is 102,336 bytes; all 10 manifested files
verified. Parent weights retained their original fingerprint before and after
execution. Optimizer steps were zero, calibration was not fitted, weights were
not selected or promoted, and independent test was not evaluated.

Implementation commits separately covered diagnosis, bounded delivery, timing
correction, and completed v1 scope. Ruff lint and format checks passed for all
181 Python files. The last integrated full suite passed **521 tests, with 1
pre-existing skip** in 139.63 s; the related scope suite passed 24 tests. This
results-only addition also checks aggregate values against the delivered raw
receipt, generation counts, identities, and cost arithmetic on CPU. Work stays
on `ayaka-v2-experiments`; stable main and published v1 are unchanged.
