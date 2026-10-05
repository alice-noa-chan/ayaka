"""Fixed-cohort bf16 parity, with HF reference reads before vLLM starts.

Gates: identical prompt/canonical token ids, complete finite gathers, probability
max-abs <= 0.02, argmax agreement >= 0.98, and — for every candidate with HF
probability >= 1e-3 — the log-odds against the HF top letter may differ by at most
4 bf16 ulps of the compared logits' magnitude.

Revision (2026-10-05, disclosed): the first GPU run used a 0.05-nat centered
log-mass gate over ALL letters. Raw logits there sit at |x| ~ 16-32, where one bf16
ulp is 0.125 nats, so that gate was below the arithmetic's resolution and could not
pass for any two kernels; observed tail differences (p < 1e-5) reached 0.97 nats
while relevant candidates differed by 0-4 ulps and probabilities by <= 0.0125. The
ulp-based gate is derived from bf16 arithmetic, not fitted to those numbers, and was
approved by the user before rerunning. The all-letter centered log-mass maximum is
still recorded as a diagnostic. Failed comparisons are diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.eval.read_artifact import fingerprint  # noqa: E402
from ayaka.swift.collect import iter_dataset  # noqa: E402
from ayaka.swift.prompt import PROMPT_VARIANTS, render_question  # noqa: E402
from ayaka.swift.readers import (  # noqa: E402
    READOUT,
    HFReader,
    ReadResult,
    VLLMChatReader,
    logmass_probs,
    template_kwargs,
)

# Predeclared bf16 gates. Every item must have identical prompt/candidate IDs
# and complete finite raw gathers; aggregate probability gates are additional.
PROBABILITY_MAX_ABS = 0.02
# Diagnostic only since the 2026-10-05 revision (see module docstring).
CENTERED_LOG_MASS_MAX_ABS_NATS = 0.05
RELEVANT_PROBABILITY_FLOOR = 1e-3
RELEVANT_LOG_ODDS_MAX_BF16_ULPS = 4
MIN_ARGMAX_AGREEMENT = 0.98


def bf16_ulp(value: float) -> float:
    """Spacing of bf16 numbers at |value| (8 significand bits)."""
    magnitude = abs(value)
    if magnitude < 2.0**-126:
        return 2.0**-133
    return 2.0 ** (math.floor(math.log2(magnitude)) - 7)


COHORT = Path(__file__).with_name("parity_cohort.jsonl")


def validate_read(result, letters):
    masses, ids, logits = result.letter_log_masses, result.canonical_token_ids, result.token_logits
    if (
        not isinstance(result.input_token_ids, list)
        or not result.input_token_ids
        or any(type(i) is not int or i < 0 for i in result.input_token_ids)
    ):
        raise ValueError("parity needs actual prompt token ids")
    if (
        not isinstance(ids, dict)
        or set(ids) != set(letters)
        or any(
            not isinstance(v, list) or len(v) != 1 or type(v[0]) is not int or v[0] < 0
            for v in ids.values()
        )
    ):
        raise ValueError("parity needs complete canonical token ids")
    required = {v[0] for v in ids.values()}
    if not isinstance(logits, dict) or len(required) != len(letters) or set(logits) != required:
        raise ValueError("parity gather is incomplete or has duplicate canonical ids")
    if any(type(k) is not int for k in logits) or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in logits.values()
    ):
        raise ValueError("parity gather must be finite")
    if not isinstance(masses, dict) or set(masses) != set(letters):
        raise ValueError("parity needs complete raw log masses")
    if any(masses[k] != logits[ids[k][0]] for k in letters):
        raise ValueError("parity log masses differ from gathered canonical logits")
    expected = logmass_probs(masses)
    if (
        not isinstance(result.letter_probs, dict)
        or set(result.letter_probs) != set(letters)
        or any(
            type(result.letter_probs[k]) not in (int, float)
            or not math.isfinite(result.letter_probs[k])
            or not math.isclose(result.letter_probs[k], expected[k], rel_tol=1e-9, abs_tol=1e-12)
            for k in letters
        )
    ):
        raise ValueError("parity probabilities differ from gathered logits")


def compare(
    items,
    hf,
    vllm,
    *,
    n=None,
    max_abs=PROBABILITY_MAX_ABS,
    min_agreement=MIN_ARGMAX_AGREEMENT,
    log_mass_max_abs=CENTERED_LOG_MASS_MAX_ABS_NATS,
    prompt_variant="min",
    state_format="pretty",
    samples=None,
    on_progress=None,
):
    if (
        (n is not None and n < 1)
        or not math.isfinite(max_abs)
        or max_abs < 0
        or not 0 <= min_agreement <= 1
        or not math.isfinite(log_mass_max_abs)
        or log_mass_max_abs < 0
    ):
        raise ValueError("invalid parity sample count or thresholds")
    samples = [] if samples is None else samples
    for item in items:
        if item.public or len(item.question.labels) > 26:
            raise ValueError("parity cohort must contain non-public single-pass items")
        messages, mapping = render_question(
            item.state, item.question, prompt_variant=prompt_variant, state_format=state_format
        )
        letters = list(mapping)
        sample = {
            "id": item.id,
            "letters": len(letters),
            "complete_finite": False,
        }
        samples.append(sample)
        try:
            left = hf.read(messages, letters)
            validate_read(left, letters)
            sample.update(
                hf_prompt_token_ids_sha256=fingerprint(left.input_token_ids),
                hf_canonical_token_ids=left.canonical_token_ids,
                hf_log_masses=left.letter_log_masses,
            )
            right = vllm.read(messages, letters)
            validate_read(right, letters)
        except Exception as exc:
            sample["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if on_progress:
                on_progress()
        sample.update(
            {
                "input_token_ids_equal": left.input_token_ids == right.input_token_ids,
                "canonical_token_ids_equal": left.canonical_token_ids == right.canonical_token_ids,
                "hf_prompt_token_ids_sha256": fingerprint(left.input_token_ids),
                "vllm_prompt_token_ids_sha256": fingerprint(right.input_token_ids),
                "hf_canonical_token_ids": left.canonical_token_ids,
                "vllm_canonical_token_ids": right.canonical_token_ids,
            }
        )
        sample["complete_finite"] = True
        differences = [abs(left.letter_probs[k] - right.letter_probs[k]) for k in letters]

        # Center AFTER shifting by max to avoid overflow from irrelevant offsets.
        def centered(masses):
            peak = max(masses.values())
            shifted = {k: v - peak for k, v in masses.items()}
            if any(not math.isfinite(v) for v in shifted.values()):
                raise ValueError("parity logit span exceeds finite arithmetic")
            average = math.fsum(v / len(masses) for v in shifted.values())
            return {k: v - average for k, v in shifted.items()}

        lm, rm = centered(left.letter_log_masses), centered(right.letter_log_masses)
        log_difference = max(abs(lm[k] - rm[k]) for k in letters)
        if not math.isfinite(log_difference):
            raise ValueError("centered log-mass difference exceeds finite arithmetic")
        hl, vl = left.letter_log_masses, right.letter_log_masses
        top = max(letters, key=left.letter_probs.__getitem__)
        relevant_ulps = 0.0
        for k in letters:
            if k == top or left.letter_probs[k] < RELEVANT_PROBABILITY_FLOOR:
                continue
            ulp = bf16_ulp(max(abs(hl[top]), abs(hl[k]), abs(vl[top]), abs(vl[k])))
            relevant_ulps = max(relevant_ulps, abs((hl[k] - hl[top]) - (vl[k] - vl[top])) / ulp)
        sample.update(
            max_abs=max(differences),
            sum_abs=math.fsum(differences),
            centered_log_mass_max_abs=log_difference,
            relevant_log_odds_max_bf16_ulps=relevant_ulps,
            hf_log_masses=left.letter_log_masses,
            vllm_log_masses=right.letter_log_masses,
            argmax_agrees=max(letters, key=left.letter_probs.__getitem__)
            == max(letters, key=right.letter_probs.__getitem__),
        )
        if on_progress:
            on_progress()
        if n is not None and len(samples) == n:
            break
    if not samples or (n is not None and len(samples) != n):
        raise ValueError(f"parity requires {n} single-pass items, found {len(samples)}")
    maximum = max(sample["max_abs"] for sample in samples)
    log_maximum = max(sample["centered_log_mass_max_abs"] for sample in samples)
    relevant_maximum = max(sample["relevant_log_odds_max_bf16_ulps"] for sample in samples)
    agreement = sum(sample["argmax_agrees"] for sample in samples) / len(samples)
    identities = all(
        sample["input_token_ids_equal"]
        and sample["canonical_token_ids_equal"]
        and sample["complete_finite"]
        for sample in samples
    )
    passed = (
        identities
        and maximum <= max_abs
        and agreement >= min_agreement
        and relevant_maximum <= RELEVANT_LOG_ODDS_MAX_BF16_ULPS
    )
    return {
        "n": len(samples),
        "max_abs": maximum,
        "mean_abs": math.fsum(s["sum_abs"] for s in samples) / sum(s["letters"] for s in samples),
        "argmax_agreement": agreement,
        "relevant_log_odds_max_bf16_ulps": relevant_maximum,
        "relevant_log_odds_threshold_bf16_ulps": RELEVANT_LOG_ODDS_MAX_BF16_ULPS,
        "relevant_probability_floor": RELEVANT_PROBABILITY_FLOOR,
        "centered_log_mass_max_abs": log_maximum,
        "max_abs_threshold": max_abs,
        "argmax_agreement_threshold": min_agreement,
        "centered_log_mass_diagnostic_reference_nats": log_mass_max_abs,
        "gate_revision": "2026-10-05 bf16-ulp relevant log-odds (disclosed in module docstring)",
        "passed": passed,
        "comparison_valid": passed,
        "readout": READOUT,
        "prompt_variant": prompt_variant,
        "samples": samples,
    }


class ReferenceReader:
    def __init__(self, rows, variant):
        self.rows = iter(row for row in rows if row["prompt_variant"] == variant)

    def read(self, messages, letters):
        row = next(self.rows)
        if row["messages_sha256"] != fingerprint(messages) or row["letters"] != letters:
            raise ValueError("HF reference prompt/order differs from comparison")
        result = dict(row["result"])
        result["token_logits"] = {int(k): v for k, v in result["token_logits"].items()}
        return ReadResult(**result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="*", default=[str(COHORT)])
    parser.add_argument("--phase", choices=["reference", "compare"], default="compare")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--served-model")
    parser.add_argument("--lora-path")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--hf-device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--chat-template-kwargs", type=template_kwargs)
    parser.add_argument(
        "--prompt-variants", nargs="+", choices=PROMPT_VARIANTS, default=list(PROMPT_VARIANTS)
    )
    parser.add_argument("--state-format", choices=["pretty", "compact"], default="pretty")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {
        "model": args.model,
        "revision": args.revision,
        "complete": False,
        "passed": False,
        "comparison_valid": False,
        "variants": {},
        "rows": [],
        "lora_path": args.lora_path,
        "phase": args.phase,
        "dtype": args.dtype,
        "hf_load_s": None,
        "cohort_sha256": None,
        "chat_template_kwargs": args.chat_template_kwargs,
        "state_format": args.state_format,
    }

    def checkpoint():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(args.output)

    checkpoint()
    try:
        report["cohort_sha256"] = fingerprint(
            [Path(p).read_text(encoding="utf-8") for p in args.dataset]
        )
        items = list(iter_dataset(args.dataset))
        if any(item.public or item.split != "calibration" for item in items):
            raise ValueError("fixed cohort must derive from non-public calibration data")
        if args.phase == "reference":
            if args.dtype != "bfloat16" or args.hf_device != "cuda":
                raise ValueError("reference requires the predeclared cuda/bfloat16 recipe")
            hf = HFReader(
                args.model,
                args.hf_device,
                args.dtype,
                revision=args.revision,
                chat_template_kwargs=args.chat_template_kwargs,
            )
            loaded = time.perf_counter()
            if args.lora_path:
                hf.lora_path = args.lora_path
            try:
                hf._load()
            finally:
                report["hf_load_s"] = time.perf_counter() - loaded
                checkpoint()
            for variant in args.prompt_variants:
                for item in items:
                    messages, mapping = render_question(
                        item.state,
                        item.question,
                        prompt_variant=variant,
                        state_format=args.state_format,
                    )
                    result = hf.read(messages, list(mapping))
                    validate_read(result, list(mapping))
                    report["rows"].append(
                        {
                            "id": item.id,
                            "prompt_variant": variant,
                            "messages_sha256": fingerprint(messages),
                            "letters": list(mapping),
                            "result": asdict(result),
                        }
                    )
                    checkpoint()
            # The reference child exits before vLLM starts: allocator/context and
            # model memory are released by process teardown on the same GPU.
            report.update(complete=True, passed=True)
        else:
            if args.reference is None:
                raise ValueError("comparison requires --reference produced before vLLM startup")
            reference = json.loads(args.reference.read_text(encoding="utf-8"))
            for key in (
                "model",
                "revision",
                "dtype",
                "cohort_sha256",
                "chat_template_kwargs",
                "state_format",
                "lora_path",
            ):
                if reference.get(key) != report[key]:
                    raise ValueError(f"reference binding mismatch: {key}")
            if not reference.get("complete") or reference.get("phase") != "reference":
                raise ValueError("incomplete HF reference")
            report["hf_load_s"] = reference["hf_load_s"]
            vllm = VLLMChatReader(
                args.vllm_url,
                args.served_model or args.model,
                tokenizer_model=args.model,
                revision=args.revision,
                chat_template_kwargs=args.chat_template_kwargs,
            )
            for variant in args.prompt_variants:
                samples = []
                report["variants"][variant] = {"samples": samples}
                report["variants"][variant] = compare(
                    items,
                    ReferenceReader(reference["rows"], variant),
                    vllm,
                    prompt_variant=variant,
                    state_format=args.state_format,
                    samples=samples,
                    on_progress=checkpoint,
                )
            report["complete"] = True
            report["passed"] = all(r["passed"] for r in report["variants"].values())
            report["comparison_valid"] = report["passed"]
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        checkpoint()
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
