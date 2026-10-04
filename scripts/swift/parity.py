"""Compare canonical HF/vLLM probabilities before collecting experiment reads."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.swift.collect import iter_dataset  # noqa: E402
from ayaka.swift.prompt import PROMPT_VARIANTS, render_question  # noqa: E402
from ayaka.swift.readers import HFReader, VLLMChatReader, template_kwargs  # noqa: E402


def compare(
    items,
    hf,
    vllm,
    *,
    n=50,
    max_abs=0.02,
    min_agreement=0.98,
    prompt_variant="min",
    state_format="pretty",
    samples=None,
):
    if n < 1 or not math.isfinite(max_abs) or max_abs < 0 or not 0 <= min_agreement <= 1:
        raise ValueError("invalid parity sample count or thresholds")
    samples = [] if samples is None else samples
    for item in items:
        if len(item.question.labels) > 26:
            continue
        messages, mapping = render_question(
            item.state, item.question, prompt_variant=prompt_variant, state_format=state_format
        )
        letters = list(mapping)
        left = hf.read(messages, letters).letter_probs
        right = vllm.read(messages, letters).letter_probs
        differences = [abs(left[letter] - right[letter]) for letter in letters]
        samples.append(
            {
                "id": item.id,
                "max_abs": max(differences),
                "sum_abs": math.fsum(differences),
                "letters": len(letters),
                "argmax_agrees": max(letters, key=left.__getitem__)
                == max(letters, key=right.__getitem__),
            }
        )
        if len(samples) == n:
            break
    if len(samples) != n:
        raise ValueError(f"parity requires {n} single-pass items, found {len(samples)}")
    maximum = max(sample["max_abs"] for sample in samples)
    average = math.fsum(sample["sum_abs"] for sample in samples) / sum(
        sample["letters"] for sample in samples
    )
    agreement = sum(sample["argmax_agrees"] for sample in samples) / n
    return {
        "n": n,
        "max_abs": maximum,
        "mean_abs": average,
        "argmax_agreement": agreement,
        "max_abs_threshold": max_abs,
        "argmax_agreement_threshold": min_agreement,
        "passed": maximum <= max_abs and agreement >= min_agreement,
        "readout": "canonical_letter",
        "prompt_variant": prompt_variant,
        "samples": samples,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="+")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--hf-device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--chat-template-kwargs", type=template_kwargs)
    parser.add_argument("--prompt-variants", nargs="+", choices=PROMPT_VARIANTS, default=["min"])
    parser.add_argument("--state-format", choices=["pretty", "compact"], default="pretty")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--max-abs", type=float, default=0.02)
    parser.add_argument("--min-argmax-agreement", type=float, default=0.98)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {
        "model": args.model,
        "revision": args.revision,
        "complete": False,
        "passed": False,
        "variants": {},
    }
    try:
        hf = HFReader(
            args.model,
            args.hf_device,
            args.dtype,
            revision=args.revision,
            chat_template_kwargs=args.chat_template_kwargs,
        )
        vllm = VLLMChatReader(
            args.vllm_url,
            args.model,
            revision=args.revision,
            chat_template_kwargs=args.chat_template_kwargs,
        )
        for variant in args.prompt_variants:
            samples = []
            report["variants"][variant] = {"samples": samples}
            report["variants"][variant] = compare(
                iter_dataset(args.dataset),
                hf,
                vllm,
                n=args.n,
                max_abs=args.max_abs,
                min_agreement=args.min_argmax_agreement,
                prompt_variant=variant,
                state_format=args.state_format,
                samples=samples,
            )
        report["complete"] = True
        report["passed"] = all(result["passed"] for result in report["variants"].values())
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
