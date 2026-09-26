"""CPU/GPU microbenchmark of the old and sparse normalization/pooling path.

Run from the repository root: python -m benchmarks.head_pooling
No model weights or datasets are downloaded. Timings cover normalization and
candidate pooling only; they are not end-to-end model latency or accuracy.
"""

from __future__ import annotations

import argparse
import json
import platform

import torch
from torch.utils.benchmark import Timer

from ayaka.model.electra import at_least_fp32, span_means


@torch.no_grad()
def legacy_pool(hidden, rows, spans, norm):
    cumulative = torch.nn.functional.pad(at_least_fp32(norm(hidden)).cumsum(1), (0, 0, 1, 0))
    starts, ends = spans.unbind(1)
    return (cumulative[rows, ends] - cumulative[rows, starts]) / (ends - starts).clamp(
        min=1
    ).unsqueeze(1)


@torch.no_grad()
def sparse_pool(hidden, rows, spans, norm):
    return span_means(hidden, rows, spans, norm=norm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=1536, help="Gemma E2B hidden width")
    parser.add_argument("--min-run-time", type=float, default=0.3)
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    results = []
    for dtype in (torch.float32, torch.bfloat16):
        for length in (256, 1024, 4096):
            hidden = torch.randn(4, length, args.hidden, dtype=dtype, device=args.device)
            norm = torch.nn.RMSNorm(args.hidden).to(device=args.device, dtype=dtype)
            rows = torch.arange(4, device=args.device).repeat_interleave(4)
            starts = torch.arange(length - 128, length, 32, device=args.device).repeat(4)
            spans = torch.stack((starts, starts + 32), dim=1)
            reference = legacy_pool(hidden, rows, spans, norm)
            actual = sparse_pool(hidden, rows, spans, norm)
            error = float((actual - reference).abs().max())
            if not torch.allclose(actual, reference, atol=1e-4, rtol=1e-4):
                raise RuntimeError(f"pooling parity failed: max absolute error {error}")
            timings = {}
            for name, operation in (("legacy", legacy_pool), ("sparse", sparse_pool)):
                timer = Timer(
                    "operation(hidden, rows, spans, norm)",
                    globals={
                        "operation": operation,
                        "hidden": hidden,
                        "rows": rows,
                        "spans": spans,
                        "norm": norm,
                    },
                    num_threads=args.threads,
                )
                timings[name] = (
                    timer.blocked_autorange(min_run_time=args.min_run_time).median * 1000
                )
            results.append(
                {
                    "dtype": str(dtype),
                    "rows": 4,
                    "sequence_tokens": length,
                    "hidden": args.hidden,
                    "candidate_tokens": 512,
                    "full_row_tokens": 4 * length,
                    "legacy_ms": timings["legacy"],
                    "sparse_ms": timings["sparse"],
                    "speedup": timings["legacy"] / timings["sparse"],
                    "max_abs_error": error,
                }
            )
    print(
        json.dumps(
            {
                "scope": "normalization and candidate pooling only; no backbone or set mixer",
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": args.device,
                "threads": args.threads,
                "seed": 0,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
