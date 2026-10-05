# Where the cygnet prompt's gain comes from (2026-10-05)

Diagnostic only. It uses saved reads (attempt 7 v2 dev, hard dev), makes no GPU calls and fits nothing for serving.
Script: `scripts/swift/noul_decomposition.py`.

## Question

Before spending on prompt-only KL distillation (long cygnet prompt → short `min` student), check what the teacher
actually adds. If its Noul gain were only a threshold shift, a cheap policy lever would capture it without training.

## Per-type CC change, cygnet − min (from `hard_dev_gate.json`)

| Dev slice | Choice | Noul | Score |
|---|---:|---:|---:|
| v2 dev | +0.3 | **+12.5** | +1.2 |
| hard dev | +1.8 | 0.0 | +1.3 |

Almost all of cygnet's v2 Intelligence gain (ΔI +4.8) is Noul.

## Noul: discrimination or threshold?

Accuracy is on raw P(true). "Cal-fit" uses a logit threshold fit on calibration. "Oracle" is the best threshold on
dev; it is an upper bound and never a policy.

| Slice | Prompt | AUC | acc @0.5 | cal-fit threshold | oracle | policy CC |
|---|---|---:|---:|---:|---:|---:|
| v2 dev, all (352) | min | 0.930 | 0.849 | 0.872 | 0.886 | 71.0 |
| | cygnet | 0.978 | 0.918 | 0.918 | 0.918 | 83.5 |
| v2 dev, repository-authored (320) | min | 0.934 | 0.853 | 0.878 | 0.894 | 71.9 |
| | cygnet | 0.984 | 0.934 | 0.934 | 0.934 | 86.9 |
| v2 dev, verified (32) | min | 0.898 | 0.812 | 0.844 | 0.875 | 62.5 |
| | cygnet | 0.898 | 0.750 | 0.750 | 0.844 | 50.0 |
| hard dev, HotpotQA (116) | min | 0.978 | 0.905 | 0.879 | 0.922 | 82.8 |
| | cygnet | 0.985 | 0.914 | 0.905 | 0.931 | 82.8 |

## Findings

1. **Mostly real discrimination, not a threshold.** On v2 dev, `min` leans "false": mean P(true) is .43 against 51%
   true gold. Even its dev-oracle threshold reaches only 0.886, while cygnet reaches 0.918 at 0.5. About a third of
   the gap is threshold; about two thirds is AUC (0.930 → 0.978).
2. **The gain is specific to the repository-authored Noul items.** It does not appear on natural multi-hop Noul:
   hard dev HotpotQA has equal policy CC (82.8 and 82.8) and AUC .978 vs .985. On the 32 verified procedural items,
   cygnet is worse.
3. **A Noul threshold lever for `min` does not transfer.** Fit on calibration, it gains 2.3 points on v2 dev but
   loses 2.6 points on hard dev (0.905 → 0.879). It is not proposed.

## Consequence for distillation

- The teacher signal worth distilling is concentrated in one in-house authored Noul family.
- On fresh natural hard data, cygnet's edge is small: Choice +1.8 and Score +1.3 CC, no Noul gain.
- So the expected benefit of prompt-only KL on JevBench-like hard items is modest. Its main measurable effect would
  be on the authored family, which is not evidence of benchmark gain.
- Recommendation: do not spend GPU on prompt-only KL on this evidence alone. If it is tried later:
  - the train pool must not be dominated by the authored Noul family;
  - success must be judged on a fresh natural hard cohort, separately from authored items.
