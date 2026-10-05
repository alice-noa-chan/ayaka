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
| calibration.jsonl | 390 | 262 | `05c61cde6132ffa28ede9933dcdcb1cc7a0c2a3f34c6ced9fad679daf81875d8` |
| dev.jsonl | 396 | 271 | `8a0e5b8a1f2f03b2f25d5492d311d88c56d2d6d8dd4a8cc9c24c9b955508f4e3` |

A rebuild reproduced both hashes. HotpotQA noul fell short of its 120 target in both splits (calibration 110,
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
