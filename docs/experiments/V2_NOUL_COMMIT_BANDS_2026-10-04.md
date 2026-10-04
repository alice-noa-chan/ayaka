# Existing-read Noul commitment bands — 2026-10-04

The previously abstaining groups do not support the `.801` selected confidence
assigned by unconditional boundary commitment on these existing Ayaka dev reads.
Their mean target credit is `.3659–.5682`; their individual case-bootstrap
interval upper bounds remain below `.801`. Both the original predictions and
the commitment rule contribute to the observed confidence gap.

This measures the old **ayaka-base and completed v2 pilot**, not the proposed
frozen 12B reader or published ayaka-large. No policy was fitted or selected,
no model forward or GPU ran, and independent test remained unopened.

## Fixed diagnostic

Use the same complete, matched raw and historically calibrated dev reports as
the [policy probe](V2_SAVED_POLICY_PROBE_2026-10-04.md). Group only Noul rows with
`0.2 < P(yes) < 0.8`, by the emitted decision: No below `.5`, Yes at/above `.5`.
Move them to `.199/.801` without changing that rule or tuning on these observations.

Target credit is the target distribution's mass on the selected label. It can
be fractional for an information-insufficient soft target and must not be
called hard-label accuracy. Compute NLL on both the original and emitted full
distributions. Resample complete underlying cases 2,000 times with seed 15.
Empty groups and groups with fewer than two cases have no interval.

## Observations

The confidence on the selected decision is `.801` for every moved row.
All intervals below are conditional subgroup diagnostics; they are not
adjusted for the eight comparisons or prior development-set inspection.

| Checkpoint | Input arm | Decision | Rows / cases | Mean target credit | Credit 95% interval | Mean NLL increase | NLL increase 95% interval |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| Parent base | raw | No | 326 / 124 | .4448 | [.3548, .5336] | .1425 | [.0978, .1890] |
| Parent base | raw | Yes | 41 / 39 | .3659 | [.2674, .4744] | .2564 | [.1710, .3412] |
| Parent base | calibrated | No | 515 / 179 | .4951 | [.4249, .5636] | .2193 | [.1482, .2944] |
| Parent base | calibrated | Yes | 145 / 61 | .5379 | [.4362, .6323] | .1640 | [.0935, .2446] |
| v2 pilot | raw | No | 409 / 162 | .5037 | [.4275, .5833] | .2081 | [.1200, .2955] |
| v2 pilot | raw | Yes | 88 / 58 | .5682 | [.4302, .6882] | .0845 | [-.0552, .2442] |
| v2 pilot | calibrated | No | 419 / 166 | .5060 | [.4303, .5858] | .2133 | [.1154, .3031] |
| v2 pilot | calibrated | Yes | 103 / 65 | .5340 | [.4056, .6524] | .1076 | [-.0174, .2466] |

Every subgroup's point estimate of NLL change is adverse, but the **two pilot
Yes NLL intervals include zero**. Do not present those individual subgroup
loss increases as independently established effects. The companion policy
probe records aggregate NLL/Brier/ECE deterioration and competence gains.

This rejects the assumption that unconditional `.801` commitment is calibrated
for these old-model subgroups. It does not prove that a selective margin rule
or the new frozen reader will have the same behavior. Evaluate their emitted
probabilities on isolated calibration/dev reads before choosing a policy.

## Reproduce and provenance

The implementation is `2de3006`, `ayaka.eval.saved_policy_probe`. Its output
adds `commit_bands` to the four-arm probe without changing the fixed policy.

```powershell
.venv/Scripts/python.exe -m ayaka.eval.saved_policy_probe `
  --raw runs/beam-v2-clean-20261002/received/result/recovery/parent-raw-dev.json `
  --calibrated runs/beam-v2-clean-20261002/received/result/recovery/parent-calibrated-dev.json `
  --calibration runs/beam-v2-clean-20261002/received/result/recovery/parent-calibration.json `
  --out runs/saved-policy-probe-20261004/parent-bands-reproduction.json
```

Replace `parent` with `pilot` and choose a fresh output name. Existing receipts
were preserved; new receipts are `parent-bands.json` and `pilot-bands.json` in
the same output directory. The [JSON record](V2_NOUL_COMMIT_BANDS_2026-10-04.json)
contains exact checkpoint/source/receipt hashes, raw selected confidence,
hard/soft target counts, complete intervals and interpretation limits.

Related tests: **35 passed** (3.13 seconds). Lint and format rechecks passed.
The concurrent integrated run ended **663 passed, 3 failed, 1 skipped** (189.18
seconds), while Claude-owned Swift code/tests changed. Failures were two Swift
fitting contract expectations and Windows/WSL Bash path handling. Exact failures
and suggested fixes were sent in `.dev` as R10; Swift files were not modified
by this implementation. This is not a final stable integration pass.

Historical calibration tags do not cryptographically establish fitting input
provenance. The dev observations do not establish independent transfer or an
official sealed-inclusive benchmark score. Claude's independent review is pending.
