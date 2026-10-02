import pytest
import torch

from ayaka.backbone import detach_text_backbone, native_logits
from ayaka.config import tiny_config
from ayaka.model.electra import ElectraDecisionModel
from ayaka.primitives import Decision, QuestionSpec
from ayaka.tokenization import ToyTokenizer


@pytest.mark.parametrize(
    "family", ["granite", "qwen3_5", "qwen3_5_hybrid", "gemma4_text", "gemma4_unified_text"]
)
def test_native_output_logits_and_cache_match(family, monkeypatch):
    from transformers import (
        Gemma4ForCausalLM,
        Gemma4UnifiedForCausalLM,
        Gemma4UnifiedTextConfig,
        GraniteConfig,
        GraniteForCausalLM,
        Qwen3_5ForCausalLM,
        Qwen3_5TextConfig,
    )

    from ayaka.backbone import tiny_text_config

    torch.set_num_threads(1)
    if family == "granite":
        cfg = GraniteConfig(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            logits_scaling=2.5,
            tie_word_embeddings=False,
        )
        lm = GraniteForCausalLM(cfg).eval()
    elif family.startswith("qwen3_5"):
        # This is a CPU reference test. Installed FLA may otherwise dispatch to
        # GPU-only Triton kernels even for CPU tensors (Transformers 5.17).
        import inspect

        from transformers.models.qwen3_5 import modeling_qwen3_5

        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
            function = getattr(modeling_qwen3_5, name)
            monkeypatch.setattr(modeling_qwen3_5, name, inspect.unwrap(function))
        cfg = Qwen3_5TextConfig(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            layer_types=["linear_attention", "full_attention"]
            if family.endswith("hybrid")
            else ["full_attention"] * 2,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            tie_word_embeddings=False,
        )
        lm = Qwen3_5ForCausalLM(cfg).eval()
    elif family == "gemma4_unified_text":
        cfg = Gemma4UnifiedTextConfig(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            layer_types=["full_attention"] * 2,
            final_logit_softcapping=30.0,
        )
        lm = Gemma4UnifiedForCausalLM(cfg).eval()
    else:
        cfg = tiny_text_config()
        lm = Gemma4ForCausalLM(cfg).eval()
    ids = torch.tensor([[7, 12, 18, 30]])
    with torch.no_grad():
        expected = lm(ids).logits
        text = detach_text_backbone(lm)
        hidden = text(ids).last_hidden_state
        assert torch.allclose(native_logits(text, hidden), expected, atol=1e-5)
        selected = torch.tensor([10, 11, 12, 13])
        assert torch.allclose(
            native_logits(text, hidden[0], selected),
            expected[0, torch.arange(4), selected],
            atol=1e-5,
        )
        first = text(ids[:, :2], use_cache=True)
        tail = text(ids[:, 2:], past_key_values=first.past_key_values, use_cache=True)
        assert torch.allclose(
            native_logits(text, tail.last_hidden_state), expected[:, 2:], atol=1e-5
        )
        model = ElectraDecisionModel(tiny_config(), text, cfg).eval()
        decision = Decision(model, ToyTokenizer())
        questions = [
            QuestionSpec("noul", "Which?", ["no", "yes"]),
            QuestionSpec("choice", "Intent?", ["ask", "reply", "refund"]),
        ]
        together = decision.decide("A polite request.", questions)
        for q, result in zip(questions, together, strict=True):
            alone = decision.decide("A polite request.", [q])[0]
            assert result.probs == pytest.approx(alone.probs, abs=1e-5)


def test_native_template_does_not_add_a_missing_bos_or_gemma_markers():
    from ayaka.prompt import prefix_head, render_question
    from ayaka.tokenization import HFTokenizer

    class FakeHF:
        bos_token_id = None
        pad_token_id = 0
        chat_template = "native"

        def get_vocab(self):
            return {}

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "USER:" + messages[0]["content"] + ":ASSISTANT:"

        def encode(self, text, **kwargs):
            return [ord(c) for c in text]

    tok = HFTokenizer(FakeHF(), "native")
    head = "".join(map(chr, prefix_head(tok)))
    suffix = render_question(QuestionSpec("choice", "Pick?", ["a", "b"]).view(), tok)
    tail = "".join(map(chr, suffix.suffix_ids))
    assert head.startswith("USER:")
    assert ":ASSISTANT:The answer is (" in tail
    assert "<|turn>" not in head + tail


def test_score_falls_back_to_letters_when_native_digits_are_not_single_tokens():
    from ayaka.prompt import render_question

    class SplitDigits(ToyTokenizer):
        def single_token_id(self, text):
            if text.isdigit():
                raise ValueError("native tokenizer splits bare digits")
            return super().single_token_id(text)

    tok = SplitDigits()
    spec = QuestionSpec("score", "Rating?", ["low", "high"], [0, 1])
    rendered = render_question(spec.view(), tok)
    assert rendered.label_ids == [tok.single_token_id("A"), tok.single_token_id("B")]
