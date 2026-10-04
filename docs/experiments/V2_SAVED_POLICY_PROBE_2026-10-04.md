# Existing-read policy probe — 2026-10-04

Noul commitment improves local thresholded competence but worsens emitted
probability quality on these existing Ayaka dev reads. With the same commitment
rule applied to both checkpoints, the continuation pilot's nominal advantage
over its parent is small and its case-level interval includes zero.

This is exploratory analysis of **ayaka-base and the completed 200-step v2
pilot**, not a measurement of the proposed frozen Gemma 4 12B letter reader,
published ayaka-large, an independent test or official JevBench standings.
No policy or checkpoint was promoted.

## Protocol

- Same complete dev cohort: 2,672 typed questions / 495 underlying cases.
- Existing reserved-calibration reports: 1,664 questions / 379 cases per model.
- Require exact checkpoint binding, matched raw/calibrated dev metadata and
  cohort fingerprint, and zero calibration/dev question or case overlap.
- Read the historical calibrated dev outputs; perform **no new fitting**.
- Four fixed arms: raw, raw+commit, calibrated, calibrated+commit.
- Commit only Noul probabilities strictly inside `(0.2, 0.8)`: move to `.199`
  below `.5` or `.801` at/above `.5`; recompute metrics on the emitted values.
- Paired uncertainty resamples complete underlying cases: 2,000 draws, seed 15.
- Model forwards, fitting steps and new GPU seconds: **0**. Test remained unopened.

## Measurements

Competence is local equal-type chance-corrected competence, not accuracy percent
or a four-axis official composite. Noul loss/calibration columns use 720 rows.

| Checkpoint | Arm | Competence | Noul NLL | Noul Brier | Noul ECE | Abstentions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Parent base | raw | 9.9262 | .90035 | .26115 | .27068 | 367 |
| Parent base | raw+commit | 24.7410 | .97946 | .29068 | .31750 | 0 |
| Parent base | calibrated | -6.2255 | .65679 | .19373 | .11407 | 660 |
| Parent base | calibrated+commit | 24.6078 | .84667 | .26945 | .26740 | 0 |
| v2 pilot | raw | 2.8080 | .83528 | .22268 | .16128 | 497 |
| v2 pilot | raw+commit | 26.5117 | .96383 | .27376 | .28342 | 0 |
| v2 pilot | calibrated | 1.7561 | .72953 | .21022 | .12168 | 522 |
| v2 pilot | calibrated+commit | 26.4783 | .86906 | .26567 | .27506 | 0 |

Relative to calibrated-only, calibrated+commit raises overall competence by
30.8333 points for the parent (95% interval `[27.2685, 34.7222]`) and
24.7222 points for the pilot (`[21.2037, 28.3796]`). It also worsens Noul
NLL, Brier and ECE for **both** models. Temperature fitting NLL alone therefore
does not validate the final commitment policy.

These classification improvement counts concern thresholded target credit;
soft targets can yield fractional credit. They are not hard-label accuracy counts.

## Matched checkpoint comparison

Applying the same raw+commit policy to both models yields:

| Change, pilot minus parent | Competence points |
| --- | ---: |
| Choice | -1.4664 |
| Noul | +4.4444 |
| Score | +2.3340 |
| Equal-type overall | +1.7707 |
| Overall 95% paired interval | `[-0.8448, 4.3534]` |

The interval includes zero; this does not establish a better checkpoint.
The original raw regression of -7.1182 points remains a real regression under
the original decision policy. Its magnitude is sensitive to how uncertain
Noul probabilities become decisions. The evidence does not establish inherent
Jeeves/hybrid incompatibility or identify the training cause of probability changes.

## Limits and next review

- The predeclared commitment rule was motivated by already-inspected public/dev
  results. This cohort is exploratory, not an untouched confirmation set.
- Existing calibrated report tags and disjoint rows do not cryptographically
  prove which historical inputs fitted the temperatures. Their original pipeline
  uses the reserved split; input identities for this probe are recorded separately.
- Saturated logits and omitted prompt/candidate strings cannot be reconstructed
  from these receipts. Matching metadata is weaker than full prompt equality.
- Stored timing columns describe prior inference, not new postprocessing latency.
- Proper-loss and competence tradeoffs need evaluation on the new system itself.
  No GPU budget or execution authorization is created by these findings.
- Claude review of the implementation and policy interpretation is pending.

## Reproduce

Inputs are under `runs/beam-v2-clean-20261002/received/result/recovery/`.

```powershell
.venv/Scripts/python.exe -m ayaka.eval.saved_policy_probe `
  --raw runs/beam-v2-clean-20261002/received/result/recovery/parent-raw-dev.json `
  --calibrated runs/beam-v2-clean-20261002/received/result/recovery/parent-calibrated-dev.json `
  --calibration runs/beam-v2-clean-20261002/received/result/recovery/parent-calibration.json `
  --out runs/saved-policy-probe-20261004/parent-reproduction.json
```

Replace `parent` with `pilot` for the other checkpoint and choose a fresh output
filename. The CLI refuses receipt replacement. Original local outputs are
`runs/saved-policy-probe-20261004/{parent,pilot}.json`; the matched post-policy
comparison is `checkpoints-raw-commit.json` in that directory.

The corresponding [JSON summary](V2_SAVED_POLICY_PROBE_2026-10-04.json) records
checkpoint identities, input/receipt SHA-256s, full typed summaries and paired
results. Related tests: **25 passed**; independent Swift HTTP tests: **13 passed**;
the integrated snapshot: **633 passed, 1 skipped** (98.63 seconds). Lint and
format checks passed for the Codex-owned implementation and helper scripts.
