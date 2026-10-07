# Ayaka v2 native GPU comparison — 2026-10-02

RTX PRO 6000 Blackwell Server Edition is the preferred measured GPU for the
current BF16 native E4B/LoRA recipe. Its representative backward time is 14.3%
lower than H100, and the estimated complete-work Modal compute cost is 32.2%
lower. A100 80GB PCIe works, but its 68.5% longer representative step time
outweighs the hourly discount on Modal. These are training cost forecasts from
zero-update probes, not completed training or benchmark quality results.

## Matched measurement

The immutable bundle is `runs/v2-pretraining-20261002-ready`, manifest SHA-256
`7b11be347d7a319663102b26aa6ec91de5947f23ae9efc6d6ed3aaabad9cad7e`.
The model remains `google/gemma-4-E4B-it`, revision
`ee0ef6023621cff504d758262d4e04895a5af4a2`. All GPUs use Torch 2.8.0/CUDA 12.8,
four CPU threads, 4 physical CPU cores, 32GiB requested host RAM, BF16,
frozen vision, and the same fresh adapter/head. Initial trainable SHA-256 is
`f15d1f5e0c69a34d6a3a06e4bfa8397b4d872cd348eb7816495cffa6d98eebed`.

The exact 1,200 × 64 full-plan sampler supplies eight evenly spaced batches and
six cost-proxy maxima, with duplicates removed: steps
`0,31,171,343,351,514,547,685,856,1028,1100,1112,1199`.
Each receives one warmup and two timed production input-preparation/backward
passes, with image feature caches cleared every pass and equivalent lazy AdamW
moment bytes reserved. Model and backbone training modes are enabled.
All GPUs retain micro-token budgets 4,096/8,192 and checkpoint threshold 1,024.
Schedule hash, selected steps, row/chunk/image/trace/proposal counts, bundle,
initial weights and effective policy match. No live optimizer update occurs.

The representative mean uses the eight evenly spaced step medians. Maxima
select additional expensive batches but do not inflate the representative mean.
The maximum below covers all 26 timed passes. Gradients and losses are finite;
model weights stay unchanged. Across architectures, maximum absolute total-loss
differences versus H100 are 0.01541 (A100) and 0.02655 (RTX PRO). BF16/kernel
outputs are not bit-identical, and trained accuracy equivalence is unmeasured.

| Actual allocated GPU | Representative step | Maximum sampled pass | Peak allocated / reserved | Complete probe function |
|---|---:|---:|---:|---:|
| H100 80GB HBM3 | 4.915s | 7.233s | 42.38 / 42.74GiB | 256.9s |
| A100 80GB PCIe | 8.281s | 12.836s | 42.34 / 42.70GiB | 372.8s |
| RTX PRO 6000 Blackwell Server Edition 96GB | 4.212s | 7.233s | 42.34 / 42.70GiB | 203.6s |

Memory numbers are PyTorch allocator peaks including reserved optimizer moments;
they exclude CUDA context/driver memory and cover selected mixed batches.
H100 is therefore not mandatory, and 80GB is not a demonstrated minimum.
48GB L40S could fit some batches, but the small headroom and untested full stress
suite prevent recommending it as a verified substitute. No L40S GPU was rented.
A100 40GB and single 24/32GB GPUs do not fit this measured allocation unchanged.
Quantization/offload would constitute a different, unmeasured recipe.

## Complete-work cost forecasts

All forecasts retain 1,200 updates × 64 rows, one cold optimizer allocation plus
1,199 measured warm optimizer costs, 14 actual-sized checkpoint writes and a
25% timing margin. Setup uses the previous full H100 preflight of 708.15s;
A100/RTX setup estimates scale it by their representative timing ratio. Their
full preflights have not been measured. The conservative scenario repeats the
largest sampled pass at every update; it is not a mathematical upper bound.

| GPU | Modal compute/hour | Typical total time / cost | Conservative scenario time / cost |
|---|---:|---:|---:|
| H100 | $4.394 | 2h16m / $9.93 | 3h14m / $14.18 |
| A100 80GB | $2.943 | 3h48m / $11.16 | 5h42m / $16.75 |
| RTX PRO 6000 | $3.476 | 1h56m / $6.73 | 3h12m / $11.11 |

Compute rates use [Modal's current base prices](https://modal.com/pricing),
including requested CPU and RAM. Taxes, credits, persistent storage/network,
later evaluations and extra training are excluded; these are not invoices.
A100 would need to be less than 1.493× as slow as H100 to beat its total compute
cost with this host allocation; the measured ratio is 1.685×.

[Runpod's listed Pod rates](https://www.runpod.io/pricing) are $3.49/hour for
H100 SXM, $1.59 for A100 80GB PCIe and $2.09 for RTX PRO 6000 96GB.
Applying those rates to the Modal timing gives typical projections of
$7.89 / $6.03 / $4.05 respectively. Runpod hardware, host configuration, startup
and throughput were not measured. On that price schedule A100 can beat H100
in cost, while RTX PRO remains the preferred projection. This does not establish
an actual Runpod training runtime.

These replace the earlier 2h55m H100 typical forecast only for the matched
four-thread, per-batch-warmed protocol. The earlier report used a different
profiling protocol. A completed training run has not confirmed either forecast.
Production should use the same CPU thread configuration when applying these
estimates; do not present the comparison as a proven whole-run optimization.

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
# Keep these settings for the eventual native training invocation.
```

## Reproduction and audit

`scripts/modal/benchmark_gpu_v2.py` is the local/native zero-update probe.
`scripts/modal/compare_v2.py` prepares and audits cached weights on CPU before reserving
and executing H100, A10080 and RTX PRO sequentially. Each GPU has a 1,100s
function limit, 970s child deadline, no retry and a 2s scale-down window.
The current exploration budget cannot accommodate another comparison unchanged.
Do not rerun the command against the existing ledger as though this were a
fresh eight-hour allowance.

Recorded invocation:

```powershell
modal run scripts/modal/compare_v2.py --out runs/v2-gpu-comparison-20261002
```

The [completed Modal app](https://modal.com/apps/gaon12/main/ap-SuCqeWun7UA65bt5gQnKLj)
returned all three reports with exit code zero. Native optimizer steps remain
zero. The conservative cumulative exploration reservation is 28,549.85s
(7h55m50s), below the original eight GPU-hour ceiling. This is a reservation
bound, not an invoice or observed allocation duration.

The [machine-readable comparison](results/v2-gpu-comparison-20261002.json)
stores per-step timings, exact parity checks, raw report hashes, actual
overheads, rates and forecast assumptions. Raw reports remain under
`runs/v2-gpu-comparison-20261002/`. Benchmark SHA-256:
`4efedefcb1d85575788c58c9d40d1b2f44dd534f9a1732ad39a8ae0343feca38`.
CPU validation passed 440 tests with one existing Beam skip; Ruff lint/format
and bytecode compilation pass. Main training, router fitting, calibration and
held-out trained quality evaluation remain pending.
