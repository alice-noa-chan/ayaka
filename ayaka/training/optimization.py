"""Instance-local native training kernels with explicit application and parity.

FA2 handles supported Gemma unified attention layers. Wider heads retain
SDPA with the same causal/padding mask; no approximate head splitting occurs.
Liger replaces only scaled RMSNorm and tanh GeGLU arithmetic, never typed
decision losses, output heads, parameter objects or global transformer classes.
"""

from __future__ import annotations

import importlib.metadata
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from types import MethodType

import torch

ATTENTION_NAME = "ayaka_fa2_with_wide_head_sdpa"
GEMMA_TYPES = {"gemma4_text", "gemma4_unified_text"}


@dataclass(frozen=True)
class OptimizationConfig:
    attention: str = "native"  # native | sdpa | flash_attention_2 (explicit per-layer hybrid)
    liger: bool = False
    probability_atol: float = 0.003
    probability_rtol: float = 0.02
    gradient_scaled_tolerance: float = 0.05

    def __post_init__(self):
        if self.attention not in {"native", "sdpa", "flash_attention_2"}:
            raise ValueError("unknown training attention implementation")
        if type(self.liger) is not bool:
            raise ValueError("liger must be boolean")
        if any(
            type(x) not in (int, float) or not math.isfinite(x) or x <= 0
            for x in (
                self.probability_atol,
                self.probability_rtol,
                self.gradient_scaled_tolerance,
            )
        ):
            raise ValueError("kernel parity tolerances must be finite and positive")


def dependency_versions():
    result = {}
    for name in ("torch", "transformers", "peft", "liger-kernel", "flash-attn"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


class KernelApplication:
    """Keep reversible instance attributes; no weights are replaced or copied."""

    def __init__(self, report):
        self.report = report
        self.saved = []

    def set(self, obj, key, value):
        self.saved.append((obj, key, key in obj.__dict__, obj.__dict__.get(key)))
        setattr(obj, key, value)

    def rollback(self):
        for obj, key, existed, value in reversed(self.saved):
            if existed:
                setattr(obj, key, value)
            else:
                obj.__dict__.pop(key, None)
        self.saved.clear()
        self.report["applied"] = False


def causal_padding_mask(query, key, padding=None, *, sliding_window=None):
    """Bottom-right causal alignment for full prefill and cached suffixes."""
    q_len, k_len = query.shape[2], key.shape[2]
    if q_len > k_len:
        raise ValueError("causal native attention requires queries to fit the key sequence")
    keys = torch.arange(k_len, device=query.device)
    queries = torch.arange(q_len, device=query.device) + k_len - q_len
    allowed = keys[None, :] <= queries[:, None]
    if sliding_window is not None:
        allowed &= keys[None, :] > queries[:, None] - sliding_window
    allowed = allowed[None, None, :, :]
    if padding is not None:
        if padding.ndim != 2 or padding.shape != (query.shape[0], k_len):
            raise ValueError("hybrid FA2 requires the full 2D key padding mask")
        allowed = allowed & padding[:, None, None, :].bool()
    return allowed


def hybrid_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    sliding_window=None,
    **kwargs,
):
    if not module.is_causal or kwargs.get("softcap") is not None:
        raise ValueError("hybrid FA2 supports causal text attention without attention softcap")
    if attention_mask is not None and attention_mask.ndim != 2:
        raise ValueError("hybrid FA2 does not accept arbitrary 4D or packed attention masks")
    if query.shape[-1] > 256 or value.shape[-1] != query.shape[-1]:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        module._ayaka_attention_calls["sdpa_wide_head"] += 1
        mask = causal_padding_mask(query, key, attention_mask, sliding_window=sliding_window)
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=False,
        )
    module._ayaka_attention_calls["flash_attention_2"] += 1
    if torch.are_deterministic_algorithms_enabled():
        kwargs["deterministic"] = True
    out = module._ayaka_flash_function(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attention_mask,
        query_length=query.shape[2],
        is_causal=True,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        use_top_left_mask=False,
        target_dtype=None,
        attn_implementation="flash_attention_2",
        **kwargs,
    )
    return out, None


def _liger_functions():
    try:
        from liger_kernel.transformers.functional import liger_geglu, liger_rms_norm
    except ImportError as exc:
        raise RuntimeError("requested Liger kernels are not installed; no silent skip") from exc
    return liger_rms_norm, liger_geglu


def _flash_function():
    from packaging.version import Version

    version = dependency_versions()["flash-attn"]
    if version is None or Version(version) < Version("2.5.5"):
        raise RuntimeError("requested FA2 requires flash-attn >=2.5.5; no silent fallback")
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

    return _flash_attention_forward


def _liger_topology(text):
    norms, mlps, unscaled = [], [], []
    for name, module in text.named_modules():
        kind = type(module).__name__
        if kind in {"Gemma4RMSNorm", "Gemma4UnifiedRMSNorm"}:
            if not getattr(module, "with_scale", True):
                unscaled.append(name)
                continue
            eps = getattr(module, "eps", None)
            if eps is None:
                eps = getattr(module, "variance_epsilon", None)
            if eps is None or not math.isfinite(eps) or eps <= 0 or not hasattr(module, "weight"):
                raise ValueError("unsupported native RMSNorm semantics")
            norms.append((name, module, eps))
        elif kind in {"Gemma4TextMLP", "Gemma4UnifiedTextMLP"}:
            if module.config.hidden_activation != "gelu_pytorch_tanh":
                raise ValueError("Liger GeGLU requires the native tanh GELU activation")
            mlps.append((name, module))
    if not norms or not mlps:
        raise ValueError("no supported native RMSNorm/GeGLU modules found")
    return norms, mlps, unscaled


def optimization_plan(model, options):
    """Inspect actual module topology without importing or executing CUDA kernels."""
    text = model.text_model()
    external = options.liger or options.attention == "flash_attention_2"
    if external and text.config.model_type not in GEMMA_TYPES:
        raise ValueError("external kernel adapter supports verified Gemma text architectures only")
    if external and getattr(text.config, "enable_moe_block", False):
        raise ValueError("external dense kernel adapter does not support MoE blocks")
    if options.attention == "flash_attention_2":
        if text.config.model_type != "gemma4_unified_text":
            raise ValueError(
                "hybrid FA2 requires unified causal text; E4B pruned masks need native SDPA"
            )
        if getattr(text.config, "use_bidirectional_attention", None) == "all":
            raise ValueError("hybrid FA2 cannot replace bidirectional attention")
    layers = {}
    for name, module in text.named_modules():
        if hasattr(module, "head_dim") and hasattr(module, "is_causal"):
            if options.attention == "flash_attention_2":
                if not module.is_causal:
                    raise ValueError("hybrid FA2 requires every text layer to be causal")
                layers[name] = "flash_attention_2" if module.head_dim <= 256 else "sdpa_wide_head"
            else:
                layers[name] = options.attention
    if options.attention == "flash_attention_2" and "flash_attention_2" not in layers.values():
        raise ValueError("no FA2-compatible native attention layers found")
    topology = {}
    if options.liger:
        norms, mlps, unscaled = _liger_topology(text)
        topology = {
            "scaled_rms_norm": [name for name, _, _ in norms],
            "geglu": [name for name, _ in mlps],
            "unscaled_norm_kept_native": unscaled,
        }
    return {
        "requested": asdict(options),
        "attention_layers": layers,
        "liger_modules": topology,
        "execution_verified": False,
    }


def _patch_liger(text, application, functions):
    rms, geglu = functions
    norms, mlps, unscaled = _liger_topology(text)
    # Validate the entire topology before mutating any module. Reuse existing
    # projections so PEFT wrapping and optimizer/checkpoint names stay intact.
    for _, norm, eps in norms:

        def forward(module, x, epsilon=eps):
            return rms(x, module.weight, epsilon, offset=0.0, casting_mode="gemma", in_place=False)

        application.set(norm, "forward", MethodType(forward, norm))
    for _, mlp in mlps:

        def forward(module, x):
            activated = geglu(module.gate_proj(x), module.up_proj(x))
            return module.down_proj(activated)

        application.set(mlp, "forward", MethodType(forward, mlp))
    application.report["liger_modules"] = {
        "scaled_rms_norm": [name for name, _, _ in norms],
        "geglu": [name for name, _ in mlps],
        "unscaled_norm_kept_native": unscaled,
    }


def apply_optimizations(model, options, *, liger_functions=None, flash_function=None):
    """Apply requested kernels or fail explicitly; injected functions are CPU tests only."""
    text = model.text_model()
    external = options.liger or options.attention == "flash_attention_2"
    injected = liger_functions is not None or flash_function is not None
    device = next(model.parameters()).device
    if external and not injected and device.type != "cuda":
        raise ValueError("requested external training kernels require CUDA")
    plan = optimization_plan(model, options)
    if external and text.config.model_type not in GEMMA_TYPES:
        raise ValueError("external kernel adapter supports verified Gemma text architectures only")
    if options.attention == "flash_attention_2" and text.config.model_type != "gemma4_unified_text":
        raise ValueError(
            "hybrid FA2 requires unified causal text; E4B pruned masks need native SDPA"
        )
    if (
        options.attention == "flash_attention_2"
        and getattr(text.config, "use_bidirectional_attention", None) == "all"
    ):
        raise ValueError("hybrid FA2 cannot replace bidirectional attention")
    functions = (liger_functions or _liger_functions()) if options.liger else None
    flash = (
        (flash_function or _flash_function()) if options.attention == "flash_attention_2" else None
    )
    application = KernelApplication(
        {
            "requested": asdict(options),
            "applied": False,
            "dependencies": dependency_versions(),
            "scope": "injected reference functions" if injected else "installed native kernels",
            "device": str(device),
            "architecture": text.config.model_type,
            "cross_entropy_replaced": False,
            "rope_replaced": False,
            "parity_verified": False,
            "plan": plan,
        }
    )
    try:
        if flash is not None:
            from transformers import AttentionInterface
            from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask

            AttentionInterface.register(ATTENTION_NAME, hybrid_attention)
            AttentionMaskInterface.register(ATTENTION_NAME, flash_attention_mask)
            counts = {"flash_attention_2": 0, "sdpa_wide_head": 0}
            application.report["observed_attention_calls"] = counts
            layers = {}
            for name, module in text.named_modules():
                if hasattr(module, "head_dim") and hasattr(module, "is_causal"):
                    layers[name] = (
                        "flash_attention_2" if module.head_dim <= 256 else "sdpa_wide_head"
                    )
                    application.set(module, "_ayaka_flash_function", flash)
                    application.set(module, "_ayaka_attention_calls", counts)
            if not layers or "flash_attention_2" not in layers.values():
                raise ValueError("no FA2-compatible native attention layers found")
            application.report["attention_layers"] = layers
        if options.attention != "native":
            attention = ATTENTION_NAME if flash else "sdpa"
            for cfg in {
                id(text.config): text.config,
                id(model.text_config): model.text_config,
            }.values():
                application.set(cfg, "_attn_implementation_internal", attention)
        if functions:
            _patch_liger(text, application, functions)
        application.report["applied"] = True
        return application
    except BaseException:
        application.rollback()
        raise


@contextmanager
def deterministic_kernel_probe(trainer):
    """Probe deterministic arithmetic without changing training dropout or RNG."""
    model = trainer.model
    mode = model.training
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if trainer.device.type == "cuda" else None
    dropout = [(m, m.p) for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    attn_dropout = [
        (m, m.attention_dropout) for m in model.modules() if hasattr(m, "attention_dropout")
    ]
    checkpoint = trainer._ckpt_active
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        for module, _ in dropout:
            module.p = 0.0
        for module, _ in attn_dropout:
            module.attention_dropout = 0.0
        yield
    finally:
        trainer.opt.zero_grad(set_to_none=True)
        for module, value in dropout:
            module.p = value
        for module, value in attn_dropout:
            module.attention_dropout = value
        trainer._set_checkpointing(checkpoint)
        model.train(mode)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


def optimize_and_verify(trainer, items, options, **injected):
    """Compare actual typed probabilities, losses and trainable gradients before steps."""
    from .run_v2 import trainable_digest

    if trainer.step_i != 0 or trainer.opt.state:
        raise ValueError("kernel parity must precede optimizer training")
    application = None
    before = trainable_digest(trainer.model)
    with deterministic_kernel_probe(trainer):

        def observe():
            trainer.opt.zero_grad(set_to_none=True)
            parts = trainer.backward_step(items)
            gradients = {
                name: p.grad.detach().float().cpu().clone()
                for name, p in trainer.model.named_parameters()
                if p.requires_grad and p.grad is not None
            }
            probs = trainer.predict(items)
            return {name: float(value) for name, value in parts.items()}, gradients, probs

        reference = observe()
        try:
            application = apply_optimizations(trainer.model, options, **injected)
            actual = observe()
            if reference[0].keys() != actual[0].keys() or reference[1].keys() != actual[1].keys():
                raise ValueError("kernel parity changed loss or gradient coverage")
            for key, value in actual[0].items():
                if not math.isfinite(value) or not math.isclose(
                    value,
                    reference[0][key],
                    abs_tol=options.probability_atol,
                    rel_tol=options.probability_rtol,
                ):
                    raise ValueError(f"kernel loss parity failed: {key}")
            maximum = 0.0
            for key, value in actual[1].items():
                ref = reference[1][key]
                error = float((value - ref).abs().max() / ref.abs().max().clamp_min(1e-6))
                if not torch.isfinite(value).all() or error > options.gradient_scaled_tolerance:
                    raise ValueError(f"kernel gradient parity failed: {key}")
                maximum = max(maximum, error)
            for ref, value in zip(reference[2], actual[2], strict=True):
                torch.testing.assert_close(
                    torch.tensor(value),
                    torch.tensor(ref),
                    atol=options.probability_atol,
                    rtol=options.probability_rtol,
                )
            if not actual[1] or trainable_digest(trainer.model) != before or trainer.opt.state:
                raise ValueError("kernel probe updated weights or omitted gradients")
            application.report.update(
                parity_verified=True,
                max_gradient_scaled_error=maximum,
                questions=len(items),
                optimizer_steps=0,
                weights_unchanged=True,
                dropout_disabled_for_probe=True,
                deterministic_algorithms_for_probe=True,
            )
            return application
        except BaseException:
            if application is not None:
                application.rollback()
            raise
