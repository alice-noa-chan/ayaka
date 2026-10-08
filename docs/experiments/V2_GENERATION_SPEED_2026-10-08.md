# v2 reasoned-generation speed: profile and options (2026-10-08)

**Scope:** an operational diagnostic on one A100. No accuracy was measured, nothing is promoted, and no serving default changes. The questions are the first dev questions of the held-out cohort (`heldout-dev-20261007`). Only timings and token-agreement counts are recorded, never dataset rows.

## Why

In the held-out comparison, reasoned reads generated at about 10 tokens/s (about 100 ms per token, with no fixed cost). That puts the chosen `noul_always+router` policy at a p50 of 10.4 s and a Speed axis of 49.7, just under the leaderboard penalty threshold. See `V2_HELDOUT_COMPARISON_2026-10-08.md`.

## Setup

- **Hardware:** vast.ai A100 SXM4 40GB (instance 54837702), $0.68/h.
- **Elapsed and cost:** the rental ran 47.5 min, for an estimated cost of $0.54. It was destroyed after the measurements.
- **Source commit:** `ba0a5a4`.
- **Runtime:** torch 2.8.0+cu128, transformers 5.17.0, peft 0.21.0, SDPA attention. These match the held-out comparison runtime.
- **Model:** the v2 checkpoint on `google/gemma-4-12B-it` at revision `707f0a3b`. The backbone was downloaded on the host and checked against the pinned SHA-256 values.
- **Benchmark:** `python -m ayaka.eval.generation_bench`, with 16 questions, budget 384 and one warm-up question.

## 1. The per-token host sync is not the bottleneck

| Variant | Tokens | ms/token | tokens/s | Same tokens as baseline |
|---|---|---|---|---|
| `baseline` (`generate_trace`, unmerged LoRA) | 4546 | 96.5 | 10.4 | — |
| `nosync` (EOS checked every 16 tokens) | 4546 | 98.6 | 10.1 | 16/16 |

The baseline reproduces the held-out run's speed. Removing the `.item()` wait changes nothing, so the host never waits on the GPU in the first place.

## 2. Profile of one decode step

One question was prefilled. Timings use 30 decode steps, and `torch.profiler` was run over 10 more. "GPU busy" is the total CUDA kernel time per step.

| Adapter state | Wall per step | GPU busy per step | Kernel launches per step |
|---|---|---|---|
| unmerged LoRA (as served) | 92.8 ms | 47.6 ms | 7,300 |
| adapter disabled | 74.3 ms | 37.4 ms | 4,775 |
| merged (`merge_adapter`) | 69.9 ms | 37.4 ms | 4,775 |

Without the profiler attached, merged eager decode measured 55–61 ms per token on three questions.

- **The step is launch-bound.** The GPU is idle for about half of every step while Python and `cudaLaunchKernel` (about 5 µs each, 31% of host time) issue thousands of small kernels.
- **Matmuls are near the bandwidth bound.** Weight reads take about 24 ms per step.
- **The rest is small elementwise work.** RMSNorm `pow`/`mean`/`mul`, together with the dtype casts (`_to_copy`, about 1,080 per step), makes up most of the 4,775 launches.
- **The unmerged LoRA alone adds about 2,500 launches and 20 ms per step.** It runs two extra small matmuls, with fp32 casts, on every adapted projection.
- **No weights were offloaded.** Every parameter is on the GPU: 618 tensors in bf16 and 686 in fp32 (the LoRA factors).

## 3. Static cache and `torch.compile`

- **Static cache, eager:** 61–67 ms per token, no faster than the dynamic cache. Its tokens already diverge from dynamic-cache greedy decoding within 16–89 tokens: the padded-cache attention reduces in a different order in bf16.
- **Static cache with `torch.compile(mode="reduce-overhead")`:** 136–296 ms per token, slower. Dynamo recompiles on every step. Each sliding layer's `cumulative_length_int` is a Python int guard, so the compiler hits its recompile limit, falls back, and records CUDA graphs for 51 distinct shapes.

Off-the-shelf compilation does not work with this transformers/Gemma 4 stack. A real CUDA-graph decode would need a fixed-shape cache of our own.

## Options

| Option | Expected ms/token | Changes numerics? | Effort |
|---|---|---|---|
| Merge LoRA for serving and reading (`load_checkpoint(merge=True)`) | ~58 (1.65×) | yes (bf16 rounding of merged weights) | small |
| + fused RMSNorm/GeGLU (Liger, already used in training) | ~40–45 (est.) | yes | small–medium |
| Fixed-shape decode cache + manual CUDA graph | ~30–37 (est., the GPU-busy floor of eager kernels) | yes, slightly | medium |
| vLLM / dedicated engine | ~20 (est.) | yes | large; LoRA and Swift readout parity unproven |

Every option changes the generated traces. Merged bf16 weights are an approximation of the unmerged adapter, and generation is greedy, so one changed token alters the rest of the trace. The path temperatures and the router were fitted on traces from the unmerged model, so **none of these can be adopted on timing alone.**

The proven-gains rule requires a fresh held-out read: calibration, router_train and dev for v2 off and v2 on, collected with the faster model. Temperatures, the router and the policy are then refitted, and the same dev gate is applied. The faster generation itself shortens that run. At 58 ms/token, the reasoned reads that took about 9 h would take about 5 h.

## Recommendation

1. Make the adapter mode an explicit, recorded collection option (unmerged by default). Then read the three held-out splits with merged weights, and adopt merged serving only if the dev gate still passes with the refitted temperatures and router.
2. In the same run, add the fused RMSNorm/GeGLU kernels as a second variant. Both variants change the numerics anyway, so one re-read can validate the faster of the two.
3. Pursue a fixed-shape CUDA-graph decoder only if the Speed axis is still below 50 after steps 1 and 2. At 40 ms/token the chosen policy's p50 would be about 4.5 s.
