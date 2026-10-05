# Swift hard dev protocol (frozen 2026-10-05, before any hard read)

## Why

On v2 dev the `cygnet` prompt raises Intelligence (+4.8) but fails the composite gate: ΔA +0.80, 95% CI
[−0.24, 2.24]. On the public JevBench items it looks better on hard items (201 vs 193 correct). Public items are
diagnostic only, and the gate must not be changed after the fact. This protocol adds new non-public hard and judge
data, then reruns the unchanged gate once.

## Data

`scripts/swift/build_hard_dev.py`, seed 20261005, builds the data deterministically from validation splits only.

| Per split | Source | Tier | License |
|---|---|---|---|
| 120 choice | QuALITY validation (long articles, ≤17K chars) | hard | CC BY 4.0 |
| 40 choice + ≤120 noul | HotpotQA distractor validation (multi-hop) | hard | CC BY-SA 4.0 |
| 120 score | HelpSteer2 validation (helpfulness, correctness) | judge | CC BY 4.0 |

- Rows that share a source document (QuALITY article, HotpotQA paragraph set, HelpSteer2 prompt) stay together, in
  calibration or in dev, never both. The split is assigned by `sha256(seed:cluster)`.
- Any row sharing a 13-gram with the JevBench public items is dropped. None were dropped.
- None of this data is ever used for training.

| File | Decisions | Clusters | sha256 |
|---|---|---|---|
| calibration.jsonl | 390 | 262 | `01f17bc9a454e69f771f4afb0d3d73cf415b09609f6dded36517da36b261b58c` |
| dev.jsonl | 396 | 271 | `8a237769ee38d51adfa89bf6c6fedb5d49e966196497f165dbf5ca2072c3f7ee` |

A rebuild reproduced both hashes. The hashes are of LF-terminated files. The first freeze recorded the Windows
CRLF bytes of the same content, and the GPU job's hash check refused the Linux rebuild. The JSON content was
identical row for row. The builder now always writes LF, and the hashes above replace the CRLF ones. This was
amended before any hard read.

A second amendment, also before any model call: the collector refused the data because several HelpSteer2 responses
to one prompt shared a row id (19 calibration and 24 dev duplicates). Row ids now include the source example id, and
the builder refuses duplicate ids. This changes which rows the seeded selection picks, so the hashes above are new.
The sources, targets, clustering and seed are unchanged. HotpotQA noul fell short of its 120 target in both splits (calibration 110,
dev 116).

## Reads

- Frozen gemma-4-12B-it at revision `707f0a3b…`, through the same vLLM recipe as attempt 7.
- Implementation hashes are identical to attempt 7: readers `0234f36e…`, prompt `2fa3444e…`, grouping `7fde0eba…`.
  The new reads therefore join the attempt 7 v2 reads under one recipe.
- All four variants (`min`, `cygnet`, `rules`, `labeled`) on hard calibration and hard dev.
- No reasoned reads and no latency probes are collected.

## Decision (`scripts/swift/hard_dev_gate.py`)

1. **Calibration** = attempt 7 v2 calibration + hard calibration. Every variant, `min` included, is refit with
   `fit_policy`.
2. **Selection:** the highest calibration A wins. Ties follow the predeclared order.
3. **Gate:** if the selected variant is not `min`, it is compared with `min` on dev = v2 dev + hard dev.
   - The comparison uses `paired_comparison` and `gate_decision` from `adopt.py`, with unchanged constants:
     ΔA CI lower bound > 0, ΔI guards, per-type drop ≤ 2, case-cluster bootstrap B = 2000, seed 15.
   - Speed is the assumed 91 for every variant. Cost is measured from tokens at $0.0403 per million.
4. **Diagnostics only:** v2-only and hard-only dev comparisons are reported but decide nothing.

**Check before any hard read:** with no hard data, the script reproduced the attempt 7 cygnet gate exactly: ΔA
0.8026, CI [−0.2400, 2.2428].

## What the result means

- **If the gate passes,** the selected variant becomes the direct candidate. The reasoning route was fit on `min`
  reads, so it is not carried over. It needs reasoned reads for the new variant and its own gate (a separate run).
  Until then the current policy (`min` + route) stays.
- **If the gate fails,** `min` + route stays, and `cygnet` is not revisited on v2/hard data.
- **No retries:** the hard data, the gate and this rule are not changed after the reads.

## Result (2026-10-05, run once)

The reads ran on one RTX 6000 Ada on vast.ai, at commit `cce279c`:
- 3,144 single-pass reads (4 variants × 786 decisions), every one a canonical letter raw readout.
- The rebuilt data matched both frozen hashes.
- About 36 instance-minutes at $0.601/h, about $0.45 with bandwidth. The instance was destroyed after download.
- Archive `hard.tgz` sha256 `305c4044…11fa23`. Gate report sha256 `840a7275…e1b124`.

**Calibration selection:** cygnet won on calibration A (min 73.79, cygnet 74.44, rules 73.29, labeled 73.15).

**Gate (decides):** cygnet vs min on v2 + hard dev, 1,708 decisions: ΔA +0.46, 95% CI [−0.26, +1.22]. ΔI +2.25.
The CI lower bound is not above 0, so **cygnet is not adopted.**

**Diagnostics (decide nothing):**

| Comparison with min | Dev slice | ΔA | 95% CI | ΔI |
|---|---|---|---|---|
| cygnet | v2 only | +0.67 | [−0.51, +2.12] | +4.67 |
| cygnet | hard only | −1.06 | [−1.61, −0.11] | +1.04 |
| rules | union | −1.24 | [−2.13, −0.37] | −2.21 |
| labeled | union | −0.35 | [−0.60, +0.00] | −0.14 |

On hard dev, cygnet raises Intelligence (+1.0) and calibration (C +2.8) but lowers A. Two things drive this:
- Its longer prompt costs more tokens on long documents.
- With 4K-token states, the Cost axis falls below 50 for both prompts (min 45.9, cygnet 45.4), where the composite's
  (axis/50)² gate amplifies small cost differences.

Here the Cost axis comes from hard-only token counts. On the benchmark, Cost is computed over the whole run, so this
slice overstates the effect. That is why the slice is a diagnostic and does not decide.

**Consequence, as declared:**
- The current policy stays: `min`, the fitted policy and the adopted reasoning route.
- `cygnet` is closed on v2/hard data.
- The public-item advantage (201 vs 193) is not used to override this.
