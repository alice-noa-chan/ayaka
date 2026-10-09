# v2 held-out re-read with a merged LoRA (2026-10-09)

**Scope:** diagnostic development evidence on the held-out cohorts of `V2_HELDOUT_COMPARISON_2026-10-08.md`. Nothing is promoted and no serving default changes. The private test split was not opened, and no optimizer update was made.

## Why

`V2_GENERATION_SPEED_2026-10-08.md` found that reasoned decode is launch-bound and that merging the LoRA into the bf16 weights makes it about 1.65× faster. Merging also changes the greedy traces, so the temperatures, router and policy fitted on unmerged reads cannot be reused. This run re-reads all three held-out splits with the merged adapter and refits everything from scratch.

## Execution

- **Hardware:** one vast.ai A100 SXM4 40GB (instance 54929918), $0.69/h, with source commit `b274878`.
- **Reads:** v2 off and v2 on for calibration (334), router_train (334) and dev (668). Every row records `"adapter": "merged"`. v1 is not re-read: it toggles its adapter during evidence reads and cannot be merged, so the 10-08 `dev/v1_on` read is reused.
- **Inputs:** the cohorts and protocols are byte-identical to the 10-08 run (checked before upload and again after retrieval).
- **Elapsed and cost:** the rental ran 5.36 h, for an estimated cost of $3.69. The 10-08 unmerged run took 8.94 h.
- **Integrity:** all 6 runs are complete with zero errors. The result archive has SHA-256 `f0ab2f2b…`. All 25 indexed files were verified before the rental was destroyed.

Reproduce the analysis by placing the 10-08 `dev/v1_on.jsonl` next to the merged results, then run:

```
python -m ayaka.eval.heldout_tuning --results <merged>/results --out report.json --adapter merged
```

The aggregate report is `docs/experiments/results/heldout-tuning-merged-2026-10-09.json`. Rows are not committed because some sources are CC BY-SA.

## 1. Speed

| Read | Unmerged (10-08) | Merged |
|---|---|---|
| v2 on, dev mean latency | 19.9 s | 11.5 s (1.72×) |
| v2 on, dev p50 / p95 | 17.3 / 38.1 s | 10.0 / 22.0 s |
| direct (+T), dev p50 | 0.24 s | 0.16 s |

The speed-up on real held-out reads matches the benchmark.

## 2. Refitted path temperatures (calibration only)

| Path | Unmerged T | Merged T |
|---|---|---|
| choice / direct | 1.19 | 1.22 |
| choice / reasoned | 1.73 | 1.82 |
| noul / direct | 1.02 | 1.04 |
| noul / reasoned | 2.56 | 2.71 |
| score / direct | 1.17 | 1.20 |
| score / reasoned | 2.61 | 2.76 |

The router was promoted again with λ = 0.001. Its dev NLL gain is 0.0376 (unmerged: 0.0316).

## 3. Policies on dev

The policy is again chosen on router_train, where `noul_always+router` scores 80.21 (unmerged: 80.58).

| Policy | Adapter | Dev CC | Choice CC / NLL | Noul CC / NLL | Noul abstentions | p50 / p95 | Speed axis |
|---|---|---|---|---|---|---|---|
| direct (+T) | unmerged | 63.07 | 78.5 / 0.438 | 51.9 / 0.300 | 64 | 0.24 / 0.84 s | 79.4 |
| direct (+T) | merged | 63.10 | 78.5 / 0.440 | 51.9 / 0.298 | 63 | 0.16 / 0.65 s | 81.7 |
| router | unmerged | 71.01 | 85.4 / 0.419 | 68.8 / 0.246 | 37 | 0.25 / 22.8 s | 65.2 |
| router | merged | 71.46 | 85.4 / 0.401 | 70.0 / 0.243 | 33 | 0.17 / 12.8 s | 69.0 |
| noul_always | unmerged | 73.70 | 78.5 / 0.438 | 83.7 / 0.227 | 6 | 1.46 / 20.8 s | 58.9 |
| noul_always | merged | 73.31 | 78.5 / 0.440 | 82.5 / 0.232 | 7 | 1.17 / 12.1 s | 62.2 |
| **noul_always+router** | unmerged | 76.01 | 85.4 / 0.419 | 83.7 / 0.227 | 6 | 10.4 / 25.4 s | 49.7 |
| **noul_always+router** | **merged** | **75.62** | 85.4 / **0.401** | 82.5 / 0.232 | 7 | **6.1 / 14.7 s** | **54.4** |

v1 on (published) scores 67.47, with Choice NLL 0.407 and Noul NLL 0.223.

## 4. Dev gates (`quality_hierarchy.RULE`)

| Policy (merged) | Over calibrated v2 off | Over v1 on |
|---|---|---|
| router | +8.36, CI +5.25…+11.94, **passed** | +3.99, CI −1.04…+8.54, failed |
| noul_always | +10.21, CI +7.16…+13.25, **passed** | +5.84, CI +0.38…+10.36, failed |
| noul_always+router | +12.53, CI +8.98…+16.39, **passed** | +8.15, CI +3.73…+12.44, failed |

For merged `noul_always+router` over v1 on, three checks fail (four when unmerged):

- noul/nll (0.232 vs 0.223);
- commonsense_qa Choice CC;
- helpsteer2 Score CC.

choice/nll now passes (0.401 vs 0.407).

## Conclusions

1. **Merged reads are 1.7× faster at the same quality.** Dev CC changes by −0.39 for the chosen policy (76.01 → 75.62) and moves both ways across the other policies. The changes are within the noise of 668 questions. Note that this run computed no paired merged-vs-unmerged CI.
2. **The chosen policy's Speed axis moves above the 50 penalty threshold** (49.7 → 54.4), with p50 at 6.1 s.
3. **The v1 gate still fails, now on three checks.** The remaining blockers are Noul NLL (+0.009), commonsense_qa Choice and helpsteer2 Score. Speed does not fix them.
4. Merged reading is a sound default for later held-out experiments. Merged *serving* stays off until v2 as a whole passes the v1 gate, under the proven-gains rule.

## Next

- Address the three remaining v1-gate blockers, all on merged reads:
  - Noul NLL: a per-route temperature for the composed policy, or a reasoned-Noul budget check;
  - commonsense_qa Choice and helpsteer2 Score: source-level diagnosis.
- Fused RMSNorm/GeGLU or a CUDA-graph decoder would add speed, but with Speed now above 50 they are lower priority than the gate blockers.
