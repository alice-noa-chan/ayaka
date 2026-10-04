"""Measure serial Swift HTTP latency, independent of bulk collection timing."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.swift.collect import iter_dataset  # noqa: E402
from ayaka.swift.prompt import PROMPT_VARIANTS, validate_prompt_variant  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def quantile(values: list[float], q: float) -> float | None:
    # Same order-statistic convention as ayaka.eval.jevbench._quantile.
    if not values:
        return None
    return sorted(values)[min(len(values) - 1, int(q * len(values)))]


def probe(
    paths: list[str | Path],
    url: str,
    output: Path,
    *,
    model: str,
    revision: str,
    reads: int = 200,
    timeout: float = 120,
    seed: int = 20261004,
    prompt_variant: str = "min",
) -> dict:
    validate_prompt_variant(prompt_variant)
    items = list(iter_dataset(paths))
    if reads < 1 or reads > len(items):
        raise ValueError("probe requires enough distinct items for the requested reads")
    random.Random(seed).shuffle(items)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    result = {
        "model": model,
        "revision": revision,
        "prompt_variant": prompt_variant,
        "endpoint": url.rstrip("/") + "/v1/systemone",
        "measurement": "serial Swift HTTP requests; no bulk timing",
        "concurrency": 1,
        "requested_reads": reads,
        "completed_reads": 0,
        "complete": False,
        "seed": seed,
        "datasets": [str(path) for path in paths],
        "units": "seconds",
        "p50_s": None,
        "p95_s": None,
        "samples": samples,
        "self_hosted_adjustment": {"scale": 2, "offset_s": 0.15, "applied": False},
    }
    write_json(output, result)
    for item in items[:reads]:
        question = item.question
        body = {
            "model": model,
            "state": item.state,
            "questions": {
                "probe": {
                    "type": question.type,
                    "instructions": question.instruction,
                    "criteria": dict(zip(question.labels, question.descriptions, strict=True)),
                }
            },
        }
        request = urllib.request.Request(
            result["endpoint"],
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            start = time.perf_counter()
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            seconds = time.perf_counter() - start
            decoded = json.loads(payload)
            if "probe" not in decoded.get("answers", {}):
                raise ValueError("Swift response has no probe answer")
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            write_json(output, result)
            raise
        samples.append({"id": item.id, "latency_s": seconds})
        values = [sample["latency_s"] for sample in samples]
        result.update(
            completed_reads=len(samples),
            complete=len(samples) == reads,
            p50_s=quantile(values, 0.50),
            p95_s=quantile(values, 0.95),
        )
        # Persist each sample so deadline termination still leaves a valid file.
        write_json(output, result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="+")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--reads", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="min")
    args = parser.parse_args(argv)
    probe(
        args.dataset,
        args.url,
        args.output,
        model=args.model,
        revision=args.revision,
        reads=args.reads,
        timeout=args.timeout,
        prompt_variant=args.prompt_variant,
    )


if __name__ == "__main__":
    main()
