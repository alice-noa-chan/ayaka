"""Measure reasoned-trace generation speed on held-out questions, per decode variant.

The held-out comparison generated reasoning at about 10 tokens/s on an A100,
roughly 100 ms per token with no fixed cost. This benchmark loads a checkpoint
exactly as that run did (unmerged LoRA) and times the same Swift trace
generation under several decode variants:

- ``baseline``: ``TraceGenerator.generate_trace`` unchanged.
- ``nosync``: the same greedy loop, but the chosen token stays on the device
  and EOS is checked every ``--sync-every`` tokens, so the host no longer
  waits on the GPU once per token. Up to ``sync_every - 1`` tokens past EOS
  are computed and discarded; the kept tokens are identical by construction.
- ``merged`` / ``merged_nosync``: the same two loops after
  ``merge_adapter()`` folds the LoRA into the bf16 weights. Rounding makes
  this an approximation, so agreement with ``baseline`` is reported.

Every variant reads the same questions with the same budget. The report gives
ms/token, tokens/s and per-question token agreement with ``baseline`` (exact
match and the length of the common prefix). Nothing is scored and no
probabilities are written, so the report holds no dataset rows.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from ..data.schema import Sample
from .checkpoint_comparison import questions

VERSION = "ayaka-generation-bench-1"
VARIANTS = ("baseline", "nosync", "merged", "merged_nosync")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate_nosync(generator, messages, budget, sync_every=16):
    """Greedy generation that checks for EOS only every ``sync_every`` tokens.

    Returns the generated token IDs, ending at (and including) the first EOS,
    exactly as ``generate_trace`` would. The cache is not returned: Swift
    readouts re-read the whole conversation, so it is not reused.
    """
    from ..model.fastpath import prefill_last

    input_ids, _ = generator.prepare(messages)
    model = generator.model
    text = model.text_model()
    device = model.embed_weight().device
    eos = torch.tensor(sorted(generator.eos), dtype=torch.long, device=device)
    hidden, cache = prefill_last(text, torch.tensor([input_ids], device=device))
    pending, kept = [], []
    for step in range(budget):
        token = model.lm_logits(hidden).argmax(-1).view(1, 1)
        pending.append(token)
        last = step == budget - 1
        if len(pending) == sync_every or last:
            chunk = torch.cat(pending, dim=1)[0]
            pending = []
            hits = torch.isin(chunk, eos).nonzero()
            if len(hits):
                kept.extend(chunk[: int(hits[0]) + 1].tolist())
                return kept
            kept.extend(chunk.tolist())
            if last:
                return kept
        out = text(input_ids=token, past_key_values=cache, use_cache=True)
        hidden, cache = out.last_hidden_state[:, -1], out.past_key_values
    return kept


def generate_baseline(generator, messages, budget, reserve):
    return list(generator.generate_trace(messages, budget, reserve=reserve).token_ids)


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def run_variant(generator, items, budget, *, nosync, sync_every, device):
    rows = []
    for messages, reserve in items:
        _sync(device)
        start = time.perf_counter()
        if nosync:
            tokens = generate_nosync(generator, messages, budget, sync_every)
        else:
            tokens = generate_baseline(generator, messages, budget, reserve)
        _sync(device)
        rows.append({"seconds": time.perf_counter() - start, "tokens": tokens})
    return rows


def summary(rows, reference=None):
    tokens = sum(len(r["tokens"]) for r in rows)
    seconds = sum(r["seconds"] for r in rows)
    out = {
        "questions": len(rows),
        "generated_tokens": tokens,
        "seconds": seconds,
        "ms_per_token": 1000 * seconds / max(tokens, 1),
        "tokens_per_s": tokens / seconds if seconds else 0.0,
    }
    if reference is not None:
        pairs = list(zip(reference, rows, strict=True))
        out["exact_token_match"] = sum(a["tokens"] == b["tokens"] for a, b in pairs)
        out["mean_common_prefix_fraction"] = sum(
            common_prefix(a["tokens"], b["tokens"]) / max(len(a["tokens"]), 1) for a, b in pairs
        ) / len(pairs)
    return out


def bench_items(generator, cohort, limit):
    samples = [
        Sample.from_json(json.loads(line))
        for line in Path(cohort).read_bytes().splitlines()
        if line.strip()
    ]
    items = []
    for state, spec, _ in questions(samples, "dev"):
        messages = generator.messages_for(state, spec)
        items.append((messages, generator.reserve_tokens(messages, spec)))
        if len(items) == limit:
            break
    return items


def bench(model, tok, cohort, *, limit, budget, sync_every, variants=VARIANTS, warmup=1):
    from ..reasoning_pipeline import controlled_decision

    generator = controlled_decision(model, tok, 8192).generator
    device = model.embed_weight().device
    items = bench_items(generator, cohort, limit + warmup)
    warm, items = items[:warmup], items[warmup:]
    results, raw = {}, {}
    for name in variants:
        merged = name.startswith("merged")
        if merged and not raw.get("_merged"):
            if not hasattr(model.backbone, "merge_adapter"):
                raise ValueError("merged variants need an unmerged PEFT checkpoint")
            model.backbone.merge_adapter()
            raw["_merged"] = True
        nosync = name.endswith("nosync")
        run_variant(generator, warm, budget, nosync=nosync, sync_every=sync_every, device=device)
        raw[name] = run_variant(
            generator, items, budget, nosync=nosync, sync_every=sync_every, device=device
        )
        results[name] = summary(raw[name], raw.get("baseline") if name != "baseline" else None)
        print(json.dumps({name: results[name]}), flush=True)
    if raw.get("_merged"):
        model.backbone.unmerge_adapter()
    return {
        "version": VERSION,
        "budget": budget,
        "sync_every": sync_every,
        "warmup_questions": warmup,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "variants": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--cohort", required=True, help="dev cohort JSONL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--sync-every", type=int, default=16)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    args = parser.parse_args(argv)
    variants = tuple(args.variants.split(","))
    if unknown := set(variants) - set(VARIANTS):
        parser.error(f"unknown variants: {sorted(unknown)}")
    from transformers import AutoTokenizer

    from ..checkpoint import load_checkpoint
    from ..tokenization import HFTokenizer

    model = load_checkpoint(
        args.checkpoint,
        device=args.device,
        dtype=torch.bfloat16,
        merge=False,
        backbone_path=args.backbone,
        local_files_only=True,
        strict_loading=True,
    ).eval()
    tok = HFTokenizer(
        AutoTokenizer.from_pretrained(args.backbone, local_files_only=True), model.cfg.backbone
    )
    report = bench(
        model,
        tok,
        args.cohort,
        limit=args.limit,
        budget=args.budget,
        sync_every=args.sync_every,
        variants=variants,
    )
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
