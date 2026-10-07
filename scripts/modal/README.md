# Modal entry points

Every file resolves its local paths (`ayaka/`, `docs/`, `runs/`) from the
repository root, so `modal run` works from any working directory.

| File | Purpose |
|---|---|
| `train_v1.py` | Run any `ayaka.pipeline` command (v1 training, teacher, export) on A100-80GB or H100. |
| `explore_v2.py` | CPU preparation followed by one bounded H100 v2 exploration worker. |
| `preflight_v2.py` | One zero-update H100 profile after CPU bundle and weight checks. |
| `compare_v2.py` | Matched zero-update GPU probes on H100, A100 and RTX PRO 6000. |
| `benchmark_gpu_v2.py` | The probe that `compare_v2.py` runs inside the container; also runs locally. |

```bash
modal run scripts/modal/train_v1.py --cmd "train --model electra-small --run small-v1"
modal run scripts/modal/explore_v2.py
modal run scripts/modal/compare_v2.py --out runs/v2-gpu-comparison-20261002
```

`beam_train.py` stays at the repository root because the Beam SDK derives the
handler name from the module path. The most recent GPU runs (2026-10-05 and
2026-10-07) used vast.ai; see
[the trained checkpoint comparison](../../docs/experiments/TRAINED_CHECKPOINT_COMPARISON_2026-10-07.md).
