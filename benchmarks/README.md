# Candidate pooling measurement

Run from the repository root:

```powershell
.venv/Scripts/python.exe -m benchmarks.head_pooling
```

The script generates random tensors with seed 0, checks parity against the
previous normalization/cumulative-sum implementation, then uses blocked
autorange medians. It downloads no model weights. The default is 4 rows,
Gemma E2B's hidden width of 1,536, 4 candidates per row and 32 tokens per
candidate (512 selected tokens total). Change `--threads`, `--hidden`,
`--device` or `--min-run-time` to reproduce on another machine.

## Local result, 2026-09-26

Windows, PyTorch `2.14.0+cpu`, 2 threads, default parameters:

| Dtype | Tokens per row | Old normalization/pooling (ms) | Sparse (ms) | Ratio |
|---|---:|---:|---:|---:|
| fp32 | 256 | 7.03 | 3.82 | 1.84x |
| fp32 | 1,024 | 31.98 | 3.44 | 9.28x |
| fp32 | 4,096 | 303.27 | 2.56 | 118.34x |
| bf16 | 256 | 5.97 | 2.53 | 2.36x |
| bf16 | 1,024 | 33.90 | 1.91 | 17.74x |
| bf16 | 4,096 | 295.96 | 2.74 | 108.00x |

Maximum absolute difference across these measurements: `4.89e-7`.
These are forward-only normalization/pooling timings using `torch.nn.RMSNorm`;
they exclude backbone execution, the Set Mixer, tokenization and backpropagation.
They are **not whole-model speedups**. Full training rows can contain the entire
state; cached inference and shared-prefix training pool only the question suffix,
so the short-row measurement is more relevant there. GPU timing and memory
usage have not been measured in this CPU environment.

## Quality and compatibility checks

- Full suite after the model change: 168 passed, 1 optional Beam test skipped.
  One earlier run had a Windows HTTP connection reset; the targeted HTTP/model
  tests and subsequent full suite passed. No server code was changed.
- Local mean and gradient parity for fp32/fp64/bf16, including overlapping and
  empty spans; a large-prefix cancellation regression.
- Legacy three-value gates and temperatures preserve readout behavior in both
  length buckets, compared with the previous cumulative-sum pooling.
- Full normalization versus gathered normalization preserves forward outputs
  and gradients. Existing cache, question isolation, candidate permutation,
  shared-prefix training and export round-trip checks also pass.
- Training and gradient-enabled evaluation keep the pointer at zero gates;
  pointer-only sets also retain it. No-grad evaluation skips it only when all
  active gates are zero and every question has label readout.

These checks establish implementation correctness and compatibility. They do
not establish a retrained JevBench accuracy gain. The reported Small hard score
of 45.9% corresponds to 51/111; a 75% target requires at least 84/111. A trained
checkpoint must be evaluated on the unchanged public benchmark, with separate
family results and end-to-end latency, before reporting that target as achieved.

## Generated data length check

Using the cached Gemma E2B tokenizer, seed 0/7/19 and 200 states per seed:

| Source | Questions | Minimum | Median | Maximum | Over 4,096 |
|---|---:|---:|---:|---:|---:|
| long rules | 1,800 | 2,258 | 2,920 | 3,528 | 0 |
| calendar | 1,800 | 276 | 321 | 331 | 0 |
| probability | 600 | 178 | 180 | 210 | 0 |

Lengths include the system prompt, state, question, options and answer cue.
This is a sample across three seeds, not a length guarantee for every tokenizer,
configuration or generated example. No public benchmark text is used to
generate these sources. Independent rendered-evidence oracles verify the labels.
