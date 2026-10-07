# CPU checks of the low-score and abstention causes, 2026-10-07

Each candidate cause from the October 7 review that saved data can settle
without a GPU, checked one by one. No model was run, nothing was fitted, and
no threshold changed. "Test 1" is the October 3 Beam pilot (ayaka-base
parent against its 200-step continuation, 720 Noul questions); "test 2" is
the October 7 matched comparison (376 questions, 64 Noul).

Noul tables come from the new `ayaka.eval.noul_discrimination` diagnostic:
[test 1](results/noul-discrimination-test1-20261007.json),
[test 2](results/noul-discrimination-test2-20261007.json).

## Main result

**Test 1's Noul regression is the loss of a lucky bias, not a loss of
skill.** On its two calculation families the parent and the pilot both have
no ranking skill at all, AUC 0.50 to 0.55, and the mean P(true) is the same
for gold-true and gold-false questions:

| Family (240 each) | Model | AUC | Correct | Abstain | Wrong | Mean P(true), gold false / true |
|---|---|---:|---:|---:|---:|---:|
| numeric | parent | 0.50 | 7 | 222 | 11 | 0.259 / 0.258 |
| numeric | pilot | 0.52 | 0 | 240 | 0 | 0.448 / 0.450 |
| temporal_numeric | parent | 0.54 | 70 | 116 | 54 | 0.322 / 0.314 |
| temporal_numeric | pilot | 0.55 | 17 | 191 | 32 | 0.519 / 0.506 |
| rule_revision | parent | 0.99 | 106 | 9 | 5 | 0.150 / 0.947 |
| rule_revision | pilot | 0.99 | 106 | 6 | 8 | 0.175 / 0.969 |

The parent leaned "false" on every calculation question. On
temporal_numeric, 135 of 240 gold answers are false, so that lean earned
70 correct answers without any discrimination. Training moved the mean
P(true) to about 0.5, which is better calibrated and correctly reflects the
model's lack of skill, and those questions became abstentions. Where the
model can rank (rule_revision, AUC 0.99), nothing regressed.

The pilot was trained on 513 questions of each of these families, so data
volume alone did not create the skill. A single forward pass does not solve
these items in either checkpoint.

## Check results

| # | Candidate cause | Result |
|---|---|---|
| B2 | Soft-target "cannot tell" Noul items are scored 0.5 when asserted and 0 when abstained | **Confirmed, local artifact.** All 74 public JevBench Noul items have hard yes/no answers. Test 1's dev has 120 explicit 50/50-prior items that its training data never contains; 40 of the 147 new abstentions are these, where abstaining is the calibrated answer. Test 2 has none. |
| B3 | The local formula differs from the current board | **Not for Noul.** In the v1.6.1 aggregate (`noul_method` O1S), tier Noul values equal 2 × (decisive and correct) − 1, e.g. 56.52 = 18/23 on 23 hard items. An abstention still scores like a wrong answer, matching the local CC. |
| B1 | Proper-scoring training does not see the abstention band | **Supported.** Test 1 shows the mechanism directly: centered, no-skill probabilities fall into the band. |
| D1 | The missing-evidence loss pushes confidence below 0.8 | **Ruled out for both tests.** Neither training set contains a flagged evidence state, so the loss never activated. |
| C1 | Too little calculation Noul training data | **Ruled out as the sole cause.** Test 2 has 13 questions per calculation family; test 1 has 513 and still no skill (AUC 0.50–0.55). |
| C5 / E2 / E3 | Bias toward "false" | **Confirmed as an asymmetry on the true side.** Gold-false questions are confident (mean P(true) 0.00–0.17); gold-true questions are not (test 2 v2 off: verified 0.69, Korean 0.62, Japanese 0.86; v1 Korean 0.48). Training labels are 50% true in both tests, and the Swift contract always shows `false` first. CPU data cannot separate a letter-position effect from a backbone prior. |
| E4 | A direct read cannot do the arithmetic | **Confirmed for test 1** (AUC about 0.5 on 480 questions). Test 2's v2 direct still ranks its 17 calculation items fairly (AUC 0.79) but credits only 5; forced reasoning reaches AUC 1.00 and credits all 17. |
| ko Noul | Korean intent abstentions | **A confidence problem, not a ranking problem.** v2 direct has AUC 1.00 on the 16 Korean items yet credits 10; its gold-true mean is 0.62. Forced reasoning makes it worse: gold-true mean 0.38, 5 wrong. |
| A6 | The same checkpoint scores differently by evaluation path | **Confirmed, cause open.** Training-time and comparison reads of the same 376 questions have identical token counts but probabilities that differ by up to 0.111 (median 0.003; 53 questions above 0.01). The adapter is stored in float32 and loaded without a dtype change. Kernel selection or batch shape remain candidates and need a GPU check. |
| H1 | v2 direct collection takes 2.13 s per question | **Cause found.** Each Swift-contract encode calls `input_serving_recipe`, which reserializes the full 262K-vocabulary tokenizer (about 0.7 s on this CPU). The comparison path encodes every question twice, adding about 1.4–2.8 s. `ayaka.training.tokenizer_identity.tokenizer_identity_scope` already provides reuse but the serving paths do not use it. This affects serving speed, not quality. |
| D2 / D6 | Under-training | **Inconclusive.** Training NLL fell across the run (0.69 in steps 26–40, 0.48 in 41–55, 0.42 in 56–74) with no held-out curve (`eval_every` 0). The gradient norm stayed between about 15 and 171 against a clip of 1.0, so every update was clipped. |
| G2 | Noul temperature fit is thin | **Confirmed.** The Noul temperature 0.683 comes from 64 calibration questions, all in the short bucket. |
| F5 | Reasoning traces truncated at 384 tokens | **Minor.** 15 of 376 hit the limit, 11 of them HelpSteer2 Score; on the 4 classification items credit is 0.75 against 0.88 for completed traces. |

## What this changes

1. The two tests do not show training destroying Noul skill. Test 1 lost
   credit that the parent earned by leaning "false" on questions it could
   not solve; test 2 lost credit on questions where v1 used worked steps.
2. The remaining direct-read problem is **under-confidence on "true"**:
   ranking is good on verified and MASSIVE items, but gold-true
   probabilities stay inside the band. Whether candidate order causes it is
   the cheapest GPU check left (read the same items with `true` first).
3. Calculation questions need the reasoning route; data volume did not fix
   them in a single pass.
4. Test 1's soft-target items should be reported separately from benchmark
   comparisons; they do not exist in JevBench's public Noul set.
5. The repeated tokenizer serialization is a serving latency bug worth
   fixing independently of model quality.

Still requiring a GPU: the candidate-order test (E2), the A6 kernel check, a
validated v2 router and reasoned-route temperatures (F1, F2: no reasoned
calibration reads exist yet), and training ablations.

## Reproduce

```bash
D=runs/beam-v2-clean-20261002/received/result/recovery
R=.dev/v1v2-compare-20261007/retrieved/results
python -m ayaka.eval.noul_discrimination --report parent=$D/parent-raw-dev.json \
  --report pilot=$D/pilot-raw-dev.json --group family --out test1.json
python -m ayaka.eval.noul_discrimination --report v1_on=$R/v1_on.jsonl \
  --report v2_off=$R/v2_off.jsonl --report v2_on=$R/v2_on.jsonl --group source --out test2.json
```
