"""Local random-CPU comparison of equal v2 work; no pretrained or optimizer updates."""

import argparse
import copy
import itertools
import json
from dataclasses import replace
from pathlib import Path

import torch

from ..backbone import tiny_text_config
from ..checkpoint import apply_lora
from ..config import tiny_config
from ..data.candidate_v2 import candidate_curriculum
from ..data.multimodal_v2 import image_curriculum
from ..data.reasoning_v2 import curriculum
from ..model.electra import ElectraDecisionModel
from ..multimodal import ImageBackend
from ..tokenization import ToyTokenizer
from .prepare_v2 import canonical, prepared_items
from .throughput import profile_backward
from .trainer import TrainConfig, Trainer


class ImageTokenizer(ToyTokenizer):
    def decode(self, ids):
        reserved = {value: key for key, value in self._reserved.items()}
        return "".join(reserved.get(i, chr(max(0, i - self._base))) for i in ids if i != 2)


class Processor:
    image_token = "<image>"

    def __init__(self, tok):
        from transformers import Gemma4ImageProcessor

        self.images = Gemma4ImageProcessor(patch_size=2, pooling_kernel_size=3, max_soft_tokens=70)
        self.tok = tok

    def __call__(self, text, images, **kwargs):
        patches = self.images(images=images, return_tensors="pt")
        counts = patches.pop("num_soft_tokens_per_image")
        parts, ids = text[0].split(self.image_token), []
        if len(parts) - 1 != len(counts):
            raise ValueError("tiny benchmark image marker mismatch")
        for i, part in enumerate(parts):
            ids += self.tok.encode(part)
            if i < len(counts):
                ids += [510] * counts[i]
        ids = torch.tensor([ids])
        return {
            **patches,
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": (ids == 510).long(),
        }


def run_benchmark(*, repeats=5):
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4VisionConfig

    torch.set_num_threads(1)
    torch.manual_seed(20261002)
    text = tiny_text_config()
    vision = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        patch_size=2,
        pooling_kernel_size=3,
        position_embedding_size=100,
    )
    native = Gemma4ForConditionalGeneration(
        Gemma4Config(text_config=text, vision_config=vision, image_token_id=510)
    ).model
    cfg = tiny_config(version=2, max_seq_len=4096, lora_dropout=0)
    model = ElectraDecisionModel(cfg, native.language_model, text)
    model.backbone.requires_grad_(False)
    apply_lora(model)
    tok = ImageTokenizer()
    backend = ImageBackend(native, Processor(tok), model, tok, processing_device="cpu")
    reference_model, reference_backend = copy.deepcopy((model, backend))
    reference_backend.model = reference_model
    samples = [s for s in image_curriculum("train", 1) if s.metadata["language"] == "en"]
    for sample, traces in curriculum("train", 1):
        sample.metadata["verified_traces"] = traces
        samples.append(sample)
    samples += candidate_curriculum("train", 2)
    items = [item for sample in samples for item in prepared_items(sample, tok, cfg, backend)]
    common = TrainConfig(bf16=False, micro_batch_tokens=8192)
    reference = Trainer(
        reference_model,
        tok,
        replace(
            common,
            ce_chunk_tokens=32,
            image_batch_rows=1,
            image_feature_cache_bytes=0,
            prune_supervised_positions=False,
        ),
        "cpu",
        image_backend=reference_backend,
    )
    optimized = Trainer(model, tok, common, "cpu", image_backend=backend)
    first = profile_backward(reference, itertools.repeat(items), warmup=1, repeats=repeats)
    second = profile_backward(optimized, itertools.repeat(items), warmup=1, repeats=repeats)
    deltas = {
        key: abs(first["last_losses"][key] - second["last_losses"][key])
        for key in first["last_losses"]
    }
    if any(delta > 2e-5 * max(1, abs(first["last_losses"][key])) for key, delta in deltas.items()):
        raise ValueError("equal-work benchmark changed joint losses")
    return {
        "scope": "random tiny Gemma4 CPU, equal direct/trace/image/proposal work; not H100 or pretrained speed evidence",
        "reference": first,
        "optimized": second,
        "speedup": first["median_seconds"] / second["median_seconds"],
        "loss_absolute_deltas": deltas,
        "supervised_rows_per_batch": len(items),
        "image_feature_hits": optimized.image_features.hits,
        "image_feature_misses": optimized.image_features.misses,
        "optimizer_steps": 0,
        "gpu_seconds": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if Path(args.out).exists() or not 1 <= args.repeats <= 20:
        raise ValueError("benchmark needs a new output and 1–20 repeats")
    report = run_benchmark(repeats=args.repeats)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(canonical(report) + b"\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "speedup",
                    "supervised_rows_per_batch",
                    "optimizer_steps",
                    "gpu_seconds",
                )
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    main()
