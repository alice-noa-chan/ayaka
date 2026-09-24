"""Int8 weight quantization for exported Electra models.

Storage: every Linear weight and both token embeddings are stored as
symmetric per-row int8 + a float scale (row = output channel / token),
halving the file size.

Runtime:

- Embeddings stay int8 on every device (``Int8Embedding``): lookups are
  gathers, so dequantizing only the gathered rows costs nothing. On
  Gemma 4 E2B/E4B the per-layer embeddings are the majority of all
  parameters, so this alone roughly halves memory.
- Linear: dequantized to bf16 at load by default — no accuracy loss vs
  bf16 (measured). On CPU, ``linear_mode="mixed"`` runs PyTorch dynamic
  int8 on RMSNorm-fed projections: ~30% faster but it cost 7 points on
  JevBench "original" for E2B, because dynamic int8 uses one activation
  scale per tensor and Gemma's activation outliers dominate it (see
  ayaka.export.LINEAR_MODES for the measurements).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row int8: w ≈ q * scale[:, None]."""
    w32 = w.detach().float()
    scale = w32.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = torch.round(w32 / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize_rows(q: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    return (q.float() * scale.float()[:, None]).to(dtype)


class Int8Embedding(nn.Module):
    """Drop-in for Gemma4TextScaledWordEmbedding with int8 rows."""

    def __init__(self, num_embeddings: int, dim: int, embed_scale: float, out_dtype=torch.bfloat16):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = dim
        self.scalar_embed_scale = embed_scale
        self.out_dtype = out_dtype
        self.register_buffer("qweight", torch.empty(num_embeddings, dim, dtype=torch.int8))
        self.register_buffer("scale", torch.empty(num_embeddings, dtype=torch.float32))

    def rows(self, ids: torch.Tensor) -> torch.Tensor:
        """Dequantized (unscaled) embedding rows — also the tied LM-head rows."""
        return self.qweight[ids].to(self.out_dtype) * self.scale[ids].to(self.out_dtype).unsqueeze(
            -1
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.rows(input_ids) * torch.tensor(self.scalar_embed_scale, dtype=self.out_dtype)


def dynamic_int8_linear(
    q: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None
) -> nn.Module:
    """CPU dynamic-quantized Linear built directly from int8 rows."""
    import warnings

    from torch.ao.nn.quantized.dynamic import Linear as DynLinear

    out_f, in_f = q.shape
    with warnings.catch_warnings():
        # quantized tensors are deprecated upstream; callers fall back to
        # dequantized bf16 Linear when this path disappears
        warnings.simplefilter("ignore")
        mod = DynLinear(in_f, out_f, bias_=bias is not None, dtype=torch.qint8)
        qw = torch._make_per_channel_quantized_tensor(
            q, scale.double(), torch.zeros(out_f, dtype=torch.long), axis=0
        )
        mod.set_weight_bias(qw, bias.float() if bias is not None else None)
    return mod


class _Float32IO(nn.Module):
    """Dynamic int8 Linear wants fp32 activations; keep the model's dtype
    outside so norms/embeddings can stay in bf16."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x.float()).to(x.dtype)


def cpu_dynamic_int8_supported() -> bool:
    try:
        from torch.ao.nn.quantized.dynamic import Linear  # noqa: F401

        return bool(torch.backends.quantized.supported_engines)
    except Exception:
        return False


def quantized_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """State dict of a text model with Linear/Embedding weights int8-packed.

    Keys: ``<param>.q`` (int8) and ``<param>.scale`` (fp32) replace
    ``<param>`` for quantized tensors; everything else is kept as-is.
    """
    quant_targets = set()
    for name, m in module.named_modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            quant_targets.add(f"{name}.weight")
    out: dict[str, torch.Tensor] = {}
    for key, t in module.state_dict().items():
        if key in quant_targets and t.ndim == 2:
            q, s = quantize_rows(t)
            out[f"{key}.q"] = q.contiguous()
            out[f"{key}.scale"] = s.contiguous()
        else:
            out[key] = t.contiguous()
    return out
