# Qwen3.5-4B pilot: result (2026-10-11)

This reports the pilot predeclared in `QWEN_PILOT_PREDECLARATION_2026-10-11.md`. It contains aggregates only. The machine-readable reports are in `docs/experiments/results/qwen-pilot-20261011/`.

**Decision under the predeclared rule: do not proceed.** On dev, calibrated single-pass Qwen3.5-4B v2 scores +2.4 equal-type CC over E4B v2 base. The paired 95% interval is [−1.4, +6.0], which includes zero. v2 stays on Gemma 4, and no full Qwen run is predeclared from this pilot.

## Rental

| Item | Value |
|---|---|
| Instance | vast.ai 55353734, RTX PRO 6000 S, $1.4413/h |
| Source commit | `4d4191f` on branch `qwen-pilot` |
| Rental time | 2.06 h |
| Compute cost | about $2.97 (an estimate; the invoice also includes storage and bandwidth) |
| Recovery | results verified locally; rental destroyed; no other instances |
| final_test | not in the payload, not read |

- **Kernel check:** flash-linear-attention 0.5.2 trained a tiny Qwen3.5 on CUDA before the paid training started (`kernel-check.log`). The short causal convolution still used transformers' PyTorch fallback.
- **Archive:** the result archive also packed the 0.8 GB optimizer I/O probe. The local run script now leaves it out.

## Training and checkpoint selection

Training forecast 2.7 s per step. Checkpoint selection read the bundle's router_train split every 250 steps:

| Step | Qwen3.5-4B loss | E4B v2 base loss |
|---|---|---|
| 250 | 0.574 | 0.635 |
| 500 | 0.563 | 0.627 |
| 750 | 0.535 | 0.636 |
| 1000 | **0.516** | **0.598** |
| 1250 | 0.551 | 0.620 |
| 1500 | 0.551 | 0.601 |
| 1750 | 0.549 | 0.603 |

- **Training run:** three reads without improvement stopped training at step 1750, and the exported adapter is the step-1000 state. The training phase, including profiling and the selection reads, took 1.8 h.
- **Loss caveat:** the two losses use different tokenizers and chat templates, so they are not directly comparable. The decision uses the dev reads below.

## Comparison on dev (668 questions, 342 cases)

Each model's single-pass dev read was calibrated with path temperatures fitted on its own calibration read. Both were read on the same GPU class with the LoRA merged.

| System | Equal-type CC | Choice CC / NLL / Brier | Noul CC / NLL / Brier | Score CC / NLL / Brier | Speed axis | mean latency |
|---|---|---|---|---|---|---|
| E4B v2 base | 66.11 | 77.8 / 0.503 / 0.260 | 63.8 / 0.212 / 0.063 | **56.8** / 1.114 / 0.596 | 87.0 | 0.113 s |
| Qwen3.5-4B v2 | **68.56** | **80.5 / 0.409 / 0.214** | **70.6 / 0.187 / 0.052** | 54.5 / 1.108 / 0.589 | 86.4 | 0.128 s |

| Check | Result |
|---|---|
| Paired CC gain, 95% CI (the decision) | +2.45, [−1.44, +5.96]: **includes zero** |
| Report only: per-type and per-source checks, zero tolerance | worse on Score CC/RPS/nMAE, HelpSteer2 and authored Score, CommonsenseQA Choice, Japanese MASSIVE Noul |

- **Strengths:** Qwen is better on Choice (+2.7 CC, NLL 0.503 → 0.409) and Noul (+6.8 CC). Its calibration is better on both.
- **Weakness:** Score is 2.3 CC lower. Score was already the weak type of every v2 model.
- **Speed:** single-pass speed is about the same: Speed axis 86.4 against 87.0, and mean latency 0.128 s against 0.113 s.

## What this means

- **Decision:** the pilot does not show a reliable gain over E4B, so the predeclared rule stops here. No final_test read is spent on a Qwen model.
- **Future backbone work:** the type pattern of a gain on Choice and Noul and a loss on Score suggests the backbone is not the bottleneck for Score.
  - A larger Qwen (Qwen3.5-9B, also pinned in the survey) or a Score-data change would need its own predeclaration.
  - The pilot scripts on branch `qwen-pilot` would carry over.
