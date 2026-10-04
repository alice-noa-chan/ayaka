import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from ayaka.backbone import detach_text_backbone
from ayaka.checkpoint import apply_lora
from ayaka.config import tiny_config
from ayaka.data.schema import Question, Sample
from ayaka.model.electra import ElectraDecisionModel
from ayaka.tokenization import ToyTokenizer
from ayaka.training.batching import sample_to_items
from ayaka.training.optimization import (
    ATTENTION_NAME,
    OptimizationConfig,
    apply_optimizations,
    causal_padding_mask,
    hybrid_attention,
    optimization_plan,
    optimize_and_verify,
)
from ayaka.training.trainer import TrainConfig, Trainer


def reference_rms(x, weight, eps, *, offset, casting_mode, in_place):
    assert offset == 0 and casting_mode == "gemma" and in_place is False
    out = x.float() * (x.float().square().mean(-1, keepdim=True) + eps).pow(-0.5)
    return (out * weight.float()).to(x.dtype)


def reference_geglu(gate, up):
    return F.gelu(gate, approximate="tanh") * up


def reference_flash(q, k, v, padding, *, softmax_scale, sliding_window, **kwargs):
    assert kwargs["attn_implementation"] == "flash_attention_2"
    assert kwargs["is_causal"] and kwargs["use_top_left_mask"] is False
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    mask = causal_padding_mask(q, k, padding, sliding_window=sliding_window)
    out = F.scaled_dot_product_attention(q, k, v, mask, scale=softmax_scale, enable_gqa=True)
    return out.transpose(1, 2).contiguous()


def fixture(unified=False):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    cfg, tok = tiny_config(readout="lm", lora_dropout=0.05), ToyTokenizer()
    if unified:
        from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig

        text_cfg = Gemma4UnifiedTextConfig(
            vocab_size=512,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            global_head_dim=32,
            num_global_key_value_heads=1,
            layer_types=["sliding_attention", "sliding_attention", "full_attention"],
            num_kv_shared_layers=0,
            sliding_window=16,
            max_position_embeddings=4096,
            pad_token_id=0,
        )
        lm = Gemma4UnifiedForCausalLM(text_cfg)
        model = ElectraDecisionModel(cfg, detach_text_backbone(lm), text_cfg)
    else:
        model = ElectraDecisionModel.from_config(cfg, dtype=torch.float32)
    model.backbone.requires_grad_(False)
    apply_lora(model)
    samples = [
        Sample({"ready": True}, [Question.noul("ready", "Is ready true?", 1)]),
        Sample(
            {"ready": False, "note": "Extra input padding test."},
            [Question.noul("ready", "Is ready true?", 0)],
        ),
    ]
    items = [item for sample in samples for item in sample_to_items(sample, tok, cfg)]
    trainer = Trainer(model, tok, TrainConfig(steps=2, bf16=False, log_every=0), "cpu")
    return trainer, items


@pytest.mark.parametrize("unified", [False, True])
def test_instance_liger_patch_preserves_parameters_names_predictions_losses_and_gradients(unified):
    trainer, items = fixture(unified)
    model = trainer.model
    parameter_ids = {name: id(p) for name, p in model.named_parameters()}
    state_keys = list(model.state_dict())
    native = copy.deepcopy(model)
    dropout = [(m, m.p) for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    rng = torch.get_rng_state().clone()
    application = optimize_and_verify(
        trainer,
        items,
        OptimizationConfig(liger=True),
        liger_functions=(reference_rms, reference_geglu),
    )
    assert application.report["parity_verified"]
    assert application.report["scope"] == "injected reference functions"
    assert application.report["max_gradient_scaled_error"] < 1e-5
    assert application.report["liger_modules"]["unscaled_norm_kept_native"]
    assert application.report["liger_modules"]["geglu"]
    assert parameter_ids == {name: id(p) for name, p in model.named_parameters()}
    assert state_keys == list(model.state_dict())
    assert torch.equal(torch.get_rng_state(), rng)
    assert trainer.step_i == 0 and not trainer.opt.state
    assert all(p.grad is None for p in model.parameters())
    assert all(m.p == value for m, value in dropout)
    native_layers = native.text_model().layers
    assert "forward" not in native_layers[0].mlp.__dict__
    assert "forward" in model.text_model().layers[0].mlp.__dict__
    application.rollback()
    assert "forward" not in model.text_model().layers[0].mlp.__dict__


def test_native_to_sdpa_parity_preserves_checkpoint_mode_and_has_no_optimizer_update():
    trainer, items = fixture()
    trainer._set_checkpointing(True)
    trainer.model.eval()
    application = optimize_and_verify(trainer, items, OptimizationConfig(attention="sdpa"))
    assert application.report["parity_verified"]
    assert not trainer.model.training and trainer._ckpt_active
    assert trainer.step_i == 0


def test_unified_flash_hybrid_through_real_model_collation_and_backward():
    trainer, items = fixture(unified=True)
    application = optimize_and_verify(
        trainer,
        items,
        OptimizationConfig(attention="flash_attention_2", liger=True),
        liger_functions=(reference_rms, reference_geglu),
        flash_function=reference_flash,
    )
    assert application.report["parity_verified"]
    assert application.report["observed_attention_calls"]["flash_attention_2"] > 0
    assert trainer.model.text_config._attn_implementation == ATTENTION_NAME
    application.rollback()
    assert trainer.model.text_config._attn_implementation != ATTENTION_NAME


def test_kernel_plan_uses_actual_module_topology_without_loading_external_kernels(monkeypatch):
    from ayaka.training import optimization

    trainer, _ = fixture(unified=True)
    monkeypatch.setattr(optimization, "_liger_functions", lambda: pytest.fail("kernel import"))
    monkeypatch.setattr(optimization, "_flash_function", lambda: pytest.fail("kernel import"))
    before = list(trainer.model.state_dict())
    plan = optimization_plan(
        trainer.model, OptimizationConfig(attention="flash_attention_2", liger=True)
    )
    assert plan["execution_verified"] is False
    assert len(plan["attention_layers"]) == 3
    assert len(plan["liger_modules"]["geglu"]) == 3
    assert before == list(trainer.model.state_dict())


def _hybrid_reference_case(head_dim, q_len, window, dtype):
    torch.manual_seed(5)
    tensors = [
        torch.randn(2, 4, q_len, head_dim, dtype=dtype, requires_grad=True),
        torch.randn(2, 2, 8, head_dim, dtype=dtype, requires_grad=True),
        torch.randn(2, 2, 8, head_dim, dtype=dtype, requires_grad=True),
    ]
    reference = [x.detach().double().clone().requires_grad_(True) for x in tensors]
    padding = torch.tensor([[1] * 8, [1] * 6 + [0] * 2])
    calls = {"flash_attention_2": 0, "sdpa_wide_head": 0}
    module = SimpleNamespace(
        is_causal=True,
        num_key_value_groups=2,
        _ayaka_attention_calls=calls,
        _ayaka_flash_function=reference_flash,
    )
    actual, _ = hybrid_attention(module, *tensors, padding, scaling=0.125, sliding_window=window)
    # Independent explicit FP64 reference, rather than another SDPA dispatch
    # or the mask helper under test. Grouped keys/values repeat by head group.
    mask = torch.tensor(
        [
            [
                [
                    bool(padding[b, k])
                    and k <= 8 - q_len + i
                    and (window is None or 8 - q_len + i - k < window)
                    for k in range(8)
                ]
                for i in range(q_len)
            ]
            for b in range(2)
        ]
    )[:, None]
    query, key, value = reference
    key, value = key.repeat_interleave(2, 1), value.repeat_interleave(2, 1)
    scores = (query @ key.transpose(-1, -2)) * 0.125
    expected = (scores.masked_fill(~mask, -torch.inf).softmax(-1) @ value).transpose(1, 2)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert calls["flash_attention_2" if head_dim == 16 else "sdpa_wide_head"] == 1
    return actual, expected, tensors, reference


@pytest.mark.parametrize("head_dim", [16, 512])
@pytest.mark.parametrize("q_len", [1, 3, 8])
@pytest.mark.parametrize("window", [None, 4])
def test_hybrid_bottom_right_padding_gqa_and_gradients_match_explicit_sdpa(head_dim, q_len, window):
    actual, expected, tensors, reference = _hybrid_reference_case(
        head_dim, q_len, window, torch.float64
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    for a, b in zip(tensors, reference, strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-9, rtol=1e-8)


@pytest.mark.parametrize("head_dim", [16, 512])
@pytest.mark.parametrize("q_len", [1, 3, 8])
@pytest.mark.parametrize("window", [None, 4])
def test_default_fp32_cpu_attention_roundoff_is_bounded_against_fp64(head_dim, q_len, window):
    actual, expected, tensors, reference = _hybrid_reference_case(
        head_dim, q_len, window, torch.float32
    )
    # CPU fused SDPA and GQA/math take different FP32 reduction paths between
    # Torch releases. Keep a separate explicit precision bound; CUDA typed
    # probability/loss/gradient admission tolerances remain unchanged.
    limit = 64 * torch.finfo(torch.float32).eps

    def check(value, ref):
        assert torch.isfinite(value).all()
        scaled = (value.double() - ref).abs().max() / ref.abs().max().clamp_min(1.0)
        assert scaled < limit, f"FP32/FP64 scaled residual {float(scaled)} exceeds {limit}"

    check(actual, expected)
    for a, b in zip(tensors, reference, strict=True):
        check(a.grad, b.grad)


def test_bad_liger_arithmetic_rolls_back_and_restores_rng_dropout_and_gradients():
    trainer, items = fixture()
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="kernel .* parity"):
        optimize_and_verify(
            trainer,
            items,
            OptimizationConfig(liger=True),
            liger_functions=(reference_rms, lambda a, b: reference_geglu(a, b) * 3),
        )
    assert "forward" not in trainer.model.text_model().layers[0].mlp.__dict__
    assert torch.equal(torch.get_rng_state(), rng)
    assert trainer.step_i == 0 and not trainer.opt.state
    assert all(p.grad is None for p in trainer.model.parameters())


def test_unavailable_kernel_requests_are_explicit_errors_on_cpu():
    trainer, _ = fixture()
    for options in (
        OptimizationConfig(liger=True),
        OptimizationConfig(attention="flash_attention_2"),
    ):
        with pytest.raises(ValueError, match="require CUDA"):
            apply_optimizations(trainer.model, options)


def test_unsupported_liger_activation_does_not_leave_partial_instance_patches():
    trainer, _ = fixture()
    trainer.model.text_model().config.hidden_activation = "relu"
    with pytest.raises(ValueError, match="tanh GELU"):
        apply_optimizations(
            trainer.model,
            OptimizationConfig(liger=True),
            liger_functions=(reference_rms, reference_geglu),
        )
    assert "forward" not in trainer.model.text_model().norm.__dict__


@pytest.mark.parametrize(
    "changes",
    [{"liger": "true"}, {"attention": "fast"}, {"gradient_scaled_tolerance": float("nan")}],
)
def test_malformed_kernel_config_is_rejected(changes):
    with pytest.raises(ValueError):
        OptimizationConfig(**changes)
