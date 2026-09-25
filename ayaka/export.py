"""Export trained Electra models as self-contained model folders.

    <out>/electra-small/         bf16, LoRA merged into the weights
    <out>/electra-small-int8/    int8 rows (see ayaka.quant)

Each folder::

    electra_config.json   ElectraConfig; backbone = "backbone"
    head.pt               pointer head + gate + per-primitive temperatures
    backbone/             Gemma 4 text-only config, tokenizer, safetensors
    export_meta.json      source checkpoint, quantization, parity report
    README.md             how to load and serve it

Usage::

    python -m ayaka.export --ckpt artifacts/<run>/checkpoint --out exports --name electra-small
    python -m ayaka.serve --model exports/electra-small-int8 --device cpu
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace

import torch
import torch.nn as nn

from .checkpoint import load_config
from .model.electra import ElectraDecisionModel
from .quant import (
    Int8Embedding,
    _Float32IO,
    cpu_dynamic_int8_supported,
    dequantize_rows,
    dynamic_int8_linear,
    quantized_state,
)

SHARD_BYTES = 4 * 1024**3
# linear_mode for int8 exports on CPU. Measured on Gemma 4 E2B zero-shot,
# JevBench public tiers, 8-core x86 CPU:
#   "auto"/"dequant": int8 weights, bf16 GEMM  easy 96-100% original 88.9% p50 3.0s
#   "mixed": dynamic int8 on RMSNorm-fed linears  easy 100% original 81.9% p50 2.1s
#   "dynamic": dynamic int8 everywhere             easy 54%  original 33%   p50 1.5s
# Per-tensor activation scales cannot absorb Gemma's activation outliers,
# so the accuracy-preserving weight-only mode is the default.
DYNAMIC_SAFE = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
LINEAR_MODES = ("auto", "dequant", "mixed", "dynamic")
QUANT_FILE = "quantization.json"


# ------------------------------------------------------------------ write


def _save_sharded(state: dict[str, torch.Tensor], out_dir: str) -> None:
    from safetensors.torch import save_file

    shards: list[dict[str, torch.Tensor]] = [{}]
    size = 0
    for k, t in state.items():
        n = t.numel() * t.element_size()
        if shards[-1] and size + n > SHARD_BYTES:
            shards.append({})
            size = 0
        shards[-1][k] = t.detach().cpu().contiguous()
        size += n
    weight_map, total = {}, 0
    for i, shard in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(out_dir, name), metadata={"format": "pt"})
        for k, t in shard.items():
            weight_map[k] = name
            total += t.numel() * t.element_size()
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=1)


def _save_tokenizer(source: str, out_dir: str) -> None:
    if source == "tiny":
        return
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(source).save_pretrained(out_dir)


def export_model(
    model: ElectraDecisionModel,
    out_dir: str,
    tokenizer_source: str,
    quantize: bool = False,
    meta: dict | None = None,
) -> str:
    """Write one export folder from an in-memory (merged) model."""
    bb_dir = os.path.join(out_dir, "backbone")
    os.makedirs(bb_dir, exist_ok=True)
    text = model.text_model()
    tcfg = model.text_config
    tcfg.save_pretrained(bb_dir)
    state = quantized_state(text) if quantize else text.state_dict()
    _save_sharded({f"model.{k}": v for k, v in state.items()}, bb_dir)
    if quantize:
        with open(os.path.join(bb_dir, QUANT_FILE), "w") as f:
            json.dump({"method": "int8-rowwise-symmetric", "version": 1}, f)
    _save_tokenizer(tokenizer_source, bb_dir)
    cfg = replace(model.cfg, backbone="backbone")
    with open(os.path.join(out_dir, "electra_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    torch.save(model.head_state_dict(), os.path.join(out_dir, "head.pt"))
    with open(os.path.join(out_dir, "export_meta.json"), "w") as f:
        json.dump({"quantized": quantize, **(meta or {})}, f, indent=2, default=str)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(_readme(os.path.basename(os.path.normpath(out_dir)), quantize))
    return out_dir


def _readme(name: str, quantized: bool) -> str:
    q = (
        "Weights are stored int8 (per-row symmetric), half the size of the bf16 export. "
        "Embeddings stay int8 in memory; Linear layers are dequantized to bf16 at load, which "
        "keeps accuracy identical to bf16. On CPU, `--linear-mode mixed` trades ~7 points on "
        "JevBench 'original' for ~30% lower latency (measured on Gemma 4 E2B).\n"
        if quantized
        else "Weights are bf16 with the LoRA adapter merged in.\n"
    )
    return f"""# {name}

Electra decision model (Gemma 4 backbone). {q}
## Serve (TypeSafe-compatible `/v1/systemone`)

    pip install "ayaka @ <path-or-git-url-of-this-repo>"
    python -m ayaka.serve --model . --device cuda   # or --device cpu

## Python

    from ayaka.export import load_exported
    from ayaka.primitives import Decision, QuestionSpec
    model, tok = load_exported(".", device="cuda")
    Decision(model, tok).choice(state, "What does the user want?", ["refund", "track order"])
"""


# ------------------------------------------------------------------- read


def _load_state(bb_dir: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    with open(os.path.join(bb_dir, "model.safetensors.index.json")) as f:
        files = sorted(set(json.load(f)["weight_map"].values()))
    state: dict[str, torch.Tensor] = {}
    for name in files:
        state.update(load_file(os.path.join(bb_dir, name)))
    return {k.removeprefix("model."): v for k, v in state.items()}


def _set_module(root: nn.Module, name: str, new: nn.Module) -> None:
    parent, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, child, new)


def load_int8_backbone(bb_dir: str, device="cpu", dtype=torch.bfloat16, linear_mode: str = "auto"):
    """Build the text model on meta, swap in int8 modules, then fill."""
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    dev = torch.device(device)
    cfg = Gemma4TextConfig.from_pretrained(bb_dir)
    with torch.device("meta"):
        text = Gemma4ForCausalLM(cfg).model
    state = _load_state(bb_dir)
    if linear_mode not in LINEAR_MODES:
        raise ValueError(f"linear_mode must be one of {LINEAR_MODES}")
    cpu_dyn = (
        linear_mode in ("mixed", "dynamic") and dev.type == "cpu" and cpu_dynamic_int8_supported()
    )

    for name, m in list(text.named_modules()):
        key = f"{name}.weight"
        if isinstance(m, nn.Embedding) and f"{key}.q" in state:
            emb = Int8Embedding(
                m.num_embeddings, m.embedding_dim, float(m.scalar_embed_scale), dtype
            )
            emb.qweight = state.pop(f"{key}.q").to(dev)
            emb.scale = state.pop(f"{key}.scale").float().to(dev)
            _set_module(text, name, emb)
        elif isinstance(m, nn.Linear) and f"{key}.q" in state:
            q, s = state.pop(f"{key}.q"), state.pop(f"{key}.scale")
            bias = state.pop(f"{name}.bias", None)
            new: nn.Module | None = None
            if cpu_dyn and (linear_mode == "dynamic" or name.endswith(DYNAMIC_SAFE)):
                try:
                    new = _Float32IO(dynamic_int8_linear(q, s, bias))
                except Exception:  # quantized-tensor API missing in this torch build
                    cpu_dyn = False
            if new is None:
                new = nn.Linear(
                    m.in_features, m.out_features, bias=bias is not None, device=dev, dtype=dtype
                )
                with torch.no_grad():
                    new.weight.copy_(dequantize_rows(q, s, dtype))
                    if bias is not None:
                        new.bias.copy_(bias)
            _set_module(text, name, new)

    # remaining tensors (norms, layer scalars, ...) are assigned directly:
    # load_state_dict would also walk the int8 modules, which own their keys
    for key, t in state.items():
        mod_name, _, attr = key.rpartition(".")
        mod = text.get_submodule(mod_name) if mod_name else text
        if attr in mod._parameters:
            mod._parameters[attr] = nn.Parameter(t.to(dev), requires_grad=False)
        elif attr in mod._buffers:
            mod._buffers[attr] = t.to(dev)
        else:
            raise RuntimeError(f"unexpected key in int8 export: {key}")
    # non-persistent buffers never live in the file: rebuild them for real
    text.rotary_emb = type(text.rotary_emb)(config=cfg).to(dev)
    leftovers = [
        n for n, t in list(text.named_parameters()) + list(text.named_buffers()) if t.is_meta
    ]
    if leftovers:
        raise RuntimeError(f"int8 export left tensors unmaterialized: {leftovers[:5]}")
    text.int8_linear_backend = "cpu-dynamic-int8" if cpu_dyn else f"dequantized-{dtype}"
    return text.eval(), cfg


def load_exported(
    path: str, device="cpu", dtype: torch.dtype | None = None, linear_mode: str = "auto"
):
    """Load an export folder -> (ElectraDecisionModel, tokenizer)."""
    from .tokenization import HFTokenizer, ToyTokenizer

    dev = torch.device(device)
    dtype = dtype or torch.bfloat16
    cfg = load_config(path)
    bb_dir = os.path.join(path, cfg.backbone)
    if os.path.exists(os.path.join(bb_dir, QUANT_FILE)):
        text, tcfg = load_int8_backbone(bb_dir, dev, dtype, linear_mode)
        model = ElectraDecisionModel(replace(cfg, backbone=bb_dir), text, tcfg)
    else:
        model = ElectraDecisionModel.from_config(
            replace(cfg, backbone=bb_dir), dtype=dtype, device=dev
        )
    model.head.to(dev)
    model.load_head_state_dict(torch.load(os.path.join(path, "head.pt"), map_location=dev))
    model.requires_grad_(False).eval()
    has_tok = os.path.exists(os.path.join(bb_dir, "tokenizer_config.json"))
    tok = HFTokenizer.from_pretrained(bb_dir) if has_tok else ToyTokenizer()
    return model.to(dev), tok


# ----------------------------------------------------------------- parity


@torch.no_grad()
def parity(model_a, model_b, tok, items: list, device) -> dict:
    """Agreement of two models on the same items (quantization check)."""
    import math

    from .training.trainer import TrainConfig, Trainer

    cfg = TrainConfig(steps=1, grad_checkpointing=False, micro_batch_tokens=8192)
    pa = Trainer(model_a, tok, cfg, device).predict(items)
    pb = Trainer(model_b, tok, cfg, device).predict(items)
    agree = sum(
        max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
        for a, b in zip(pa, pb, strict=True)
    )
    kl = sum(
        sum(
            x * (math.log(max(x, 1e-12)) - math.log(max(y, 1e-12)))
            for x, y in zip(a, b, strict=True)
        )
        for a, b in zip(pa, pb, strict=True)
    )
    max_abs = max(
        abs(x - y) for a, b in zip(pa, pb, strict=True) for x, y in zip(a, b, strict=True)
    )
    n = max(len(items), 1)
    return {
        "n": len(items),
        "argmax_agreement": agree / n,
        "mean_kl": kl / n,
        "max_abs_prob_diff": max_abs,
    }


def main(argv: list[str] | None = None) -> dict:
    import argparse

    from .checkpoint import load_checkpoint

    ap = argparse.ArgumentParser(description="Export an Electra checkpoint (bf16 + int8)")
    ap.add_argument("--ckpt", required=True, help="training checkpoint directory")
    ap.add_argument("--out", required=True, help="export root directory")
    ap.add_argument("--name", required=True, help="folder name, e.g. electra-small")
    ap.add_argument("--no-int8", action="store_true", help="skip the int8 export")
    ap.add_argument(
        "--parity", type=int, default=128, help="jev test items for the int8 parity check (0=skip)"
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--code-url", default="<code-url>", help="git URL for pip install in the model card"
    )
    args = ap.parse_args(argv)

    dev = torch.device(args.device)
    model = load_checkpoint(args.ckpt, device=dev, dtype=torch.bfloat16).requires_grad_(False)
    source_backbone = load_config(args.ckpt).backbone
    full_dir = export_model(
        model, os.path.join(args.out, args.name), source_backbone, meta={"source": args.ckpt}
    )
    report = {"full": full_dir}
    if not args.no_int8:
        q_dir = export_model(
            model,
            os.path.join(args.out, f"{args.name}-int8"),
            source_backbone,
            quantize=True,
            meta={"source": args.ckpt},
        )
        report["int8"] = q_dir
        if args.parity:
            from .data.decontam import Decontaminator
            from .tokenization import HFTokenizer
            from .training.run import items_from_spec

            tok = HFTokenizer.from_pretrained(source_backbone)
            items = items_from_spec(
                "jev_open_test",
                args.parity,
                tok,
                model.cfg,
                0,
                Decontaminator.from_jevbench(),
            )
            qmodel, _ = load_exported(q_dir, device=dev)
            report["int8_parity"] = parity(model, qmodel, tok, items, dev)
            meta_path = os.path.join(q_dir, "export_meta.json")
            with open(meta_path) as f:
                meta = json.load(f)
            meta["parity_vs_bf16"] = report["int8_parity"]
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
    from .modelcard import write_card

    for d in (report["full"], report.get("int8")):
        if d:
            write_card(d, args.code_url)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
