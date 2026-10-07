# Device-local ragged RPS reduction

The training RPS loss now sorts candidates by question and ordinal, computes
segmented CDF differences, and reduces eligible Score questions on device.
It no longer copies question boundaries or selected question IDs to the CPU,
and it does not allocate a padded candidate matrix. A double-precision scan
protects CPU/CUDA batch boundaries from cancellation; MPS uses float32 because
it does not support float64. The returned loss retains the probability dtype.

Single-candidate and empty segments contribute no RPS denominator. An empty
eligible set returns a differentiable zero. Loss and gradient comparisons cover
hard and soft distributions, shuffled ordinals, mixed primitives, and 512
questions. Existing optimizer-batch and microbatch reduction tests also pass.

Ragged softmax, max, sum, and padding now pass the known flat candidate count
to `repeat_interleave`. This avoids discovering the output size through device
synchronization. Operations without a known count retain the existing behavior.

## Local measurement

Measured with PyTorch 2.14.1 CPU, one thread, and
`torch.utils.benchmark.Timer.blocked_autorange(min_run_time=0.5)`. Each batch has
2–7 candidates per question, half its questions selected for RPS, uniform
predictions, and one-hot targets. Both implementations compute the same loss.
These are forward-only function timings, not end-to-end training measurements.

| Questions | Previous median (ms) | Vectorized median (ms) | Ratio |
| ---: | ---: | ---: | ---: |
| 8 | 0.120 | 0.101 | 1.19× |
| 32 | 0.442 | 0.103 | 4.30× |
| 128 | 1.765 | 0.119 | 14.90× |
| 1024 | 14.419 | 0.303 | 47.63× |

GPU throughput and complete training-step latency remain unmeasured. No model
quality improvement follows from a faster implementation of the same loss.

Validation: Ruff lint/format checks and 76 loss, reduction, distillation, and
model tests passed. The historical matched-comparison source pins are unchanged.
