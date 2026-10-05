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

from ayaka.eval.read_artifact import fingerprint  # noqa: E402
from ayaka.swift.adopt import policy_fingerprint  # noqa: E402
from ayaka.swift.collect import iter_dataset  # noqa: E402
from ayaka.swift.policy import Policy  # noqa: E402
from ayaka.swift.prompt import PROMPT_VARIANTS, validate_prompt_variant  # noqa: E402

WARMUP_REQUESTS = 3


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
    non_public: bool = False,
    policy: Policy | None = None,
) -> dict:
    validate_prompt_variant(prompt_variant)
    items = list(iter_dataset(paths))
    if non_public:
        if any(item.public or item.split != "dev" for item in items):
            raise ValueError("system probes require exclusively non-public dev")
        items = [
            item
            for item in items
            if item.tier in ("standard", "judge") and len(item.question.labels) <= 26
        ]
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
        "public": not non_public,
        "split": "dev" if non_public else "public",
        "tier_filter": ["standard", "judge"] if non_public else None,
        "system": "reasoning_route" if policy and policy.reasoning_route else "direct",
        "router_sha256": fingerprint(policy.reasoning_route)
        if policy and policy.reasoning_route
        else None,
        "policy_sha256": policy_fingerprint(policy) if policy else None,
    }
    result["warmup_requests"] = WARMUP_REQUESTS
    write_json(output, result)

    def body_for(item):
        question = item.question
        # Jev API wire shapes: Score criteria are an ordered array of level descriptions;
        # Choice/Noul criteria map option/label to description.
        criteria = (
            list(question.descriptions)
            if question.type == "score"
            else dict(zip(question.labels, question.descriptions, strict=True))
        )
        return {
            "model": model,
            "state": item.state,
            "questions": {
                "probe": {
                    "type": question.type,
                    "instructions": question.instruction,
                    "criteria": criteria,
                }
            },
        }

    # Unmeasured warm-up requests, as JevBench warms a self-hosted endpoint before timing.
    # A fixed synthetic Noul keeps measured prompts out of the prefix cache.
    warmup = {
        "model": model,
        "state": "warm-up",
        "questions": {"probe": {"type": "noul", "instructions": "Is this a warm-up?"}},
    }
    for _ in range(WARMUP_REQUESTS):
        request = urllib.request.Request(
            result["endpoint"],
            data=json.dumps(warmup).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
    for item in items[:reads]:
        body = body_for(item)
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
    parser.add_argument("--non-public", action="store_true")
    parser.add_argument("--policy", type=Path, help="policy served by the probed system")
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
        non_public=args.non_public,
        policy=Policy.load(args.policy) if args.policy else None,
    )


if __name__ == "__main__":
    main()
