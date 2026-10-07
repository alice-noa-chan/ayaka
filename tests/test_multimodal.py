import base64
import copy
import io

import pytest
import torch
from PIL import Image

from ayaka.backbone import native_logits, tiny_text_config
from ayaka.config import tiny_config
from ayaka.model.decision import AyakaDecisionModel
from ayaka.multimodal import (
    ImageBackend,
    ImageDecision,
    ImageState,
    ImageTraceGenerator,
    NativeImageDecision,
    decode_media,
)
from ayaka.primitives import QuestionSpec
from ayaka.reasoning_pipeline import TraceFailure, controlled_decision
from ayaka.serve import BadRequest, DecisionService
from ayaka.tokenization import ToyTokenizer


class ImageTok(ToyTokenizer):
    def decode(self, ids):
        reserved = {v: k for k, v in self._reserved.items()}
        return "".join(reserved.get(i, chr(max(0, i - self._base))) for i in ids if i != 2)


class Processor:
    """Test text tokenizer around the real native image processor (no Hub downloads)."""

    image_token = "<image>"

    def __init__(self, family, tok):
        from transformers import Gemma4ImageProcessor, Gemma4UnifiedImageProcessor

        cls = Gemma4ImageProcessor if family == "gemma4" else Gemma4UnifiedImageProcessor
        self.images = cls(
            patch_size=2,
            pooling_kernel_size=3 if family == "gemma4" else 1,
            max_soft_tokens=70,
        )
        self.tok = tok

    def __call__(self, text, images, **kwargs):
        assert kwargs == {"add_special_tokens": False, "return_tensors": "pt"}
        patches = self.images(images=images, return_tensors="pt")
        counts = patches.pop("num_soft_tokens_per_image")
        parts = text[0].split(self.image_token)
        assert len(parts) - 1 == len(counts)
        ids = []
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


def build(family):
    from transformers import (
        Gemma4Config,
        Gemma4ForConditionalGeneration,
        Gemma4UnifiedConfig,
        Gemma4UnifiedForConditionalGeneration,
        Gemma4UnifiedTextConfig,
        Gemma4UnifiedVisionConfig,
        Gemma4VisionConfig,
    )

    torch.set_num_threads(1)
    torch.manual_seed(7)
    if family == "gemma4":
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
        config = Gemma4Config(text_config=text, vision_config=vision, image_token_id=510)
        lm = Gemma4ForConditionalGeneration(config).eval()
    else:
        text = Gemma4UnifiedTextConfig(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=16,
            layer_types=["full_attention"] * 2,
            max_position_embeddings=4096,
            use_bidirectional_attention="vision",
            final_logit_softcapping=30.0,
        )
        vision = Gemma4UnifiedVisionConfig(
            patch_size=2,
            pooling_kernel_size=1,
            mm_embed_dim=64,
            output_proj_dims=64,
            mm_posemb_size=100,
        )
        config = Gemma4UnifiedConfig(text_config=text, vision_config=vision, image_token_id=510)
        lm = Gemma4UnifiedForConditionalGeneration(config).eval()
    model = AyakaDecisionModel(
        tiny_config(version=2, serve_max_seq_len=4096), lm.model.language_model, text
    ).eval()
    tok = ImageTok()
    backend = ImageBackend(lm.model, Processor(family, tok), model, tok)
    return lm, model, tok, backend


def media():
    out = io.BytesIO()
    Image.new("RGB", (16, 12), color=(100, 50, 20)).save(out, format="PNG")
    return [
        {
            "type": "image",
            "mime_type": "image/png",
            "data": base64.b64encode(out.getvalue()).decode(),
        }
    ]


@pytest.mark.parametrize("family", ["gemma4", "gemma4_unified"])
def test_native_image_logits_cache_and_question_isolation(family):
    lm, model, tok, backend = build(family)
    state = decode_media("An image document", media())
    decision = NativeImageDecision(backend)
    qs = [
        QuestionSpec("choice", "Type?", ["receipt", "letter", "chart"]),
        QuestionSpec("noul", "A receipt?", ["no", "yes"]),
    ]
    inputs, prefix, items = decision.encode(state, [q.view() for q in qs])
    with torch.no_grad():
        expected_prefix = lm(**inputs).logits
        actual = backend.prefill(inputs)
        assert torch.allclose(
            native_logits(model.text_model(), actual.last_hidden_state), expected_prefix, atol=1e-5
        )
        suffix = torch.tensor([items[0].rendered.suffix_ids])
        continued = model.text_model()(
            suffix, past_key_values=copy.deepcopy(actual.past_key_values)
        )
        full = {**inputs, "input_ids": torch.cat([inputs["input_ids"], suffix], dim=1)}
        full["attention_mask"] = torch.ones_like(full["input_ids"])
        full["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros_like(suffix)], dim=1
        )
        assert torch.allclose(
            native_logits(model.text_model(), continued.last_hidden_state),
            lm(**full).logits[:, len(prefix) :],
            atol=1e-5,
        )
    together = decision.decide(state, qs)
    for q, result in zip(qs, together, strict=True):
        assert result.probs == pytest.approx(decision.decide(state, [q])[0].probs, abs=1e-5)
    perm = QuestionSpec("choice", qs[0].instruction, list(reversed(qs[0].candidates)))
    assert decision.decide(state, [perm])[0].probs == pytest.approx(
        list(reversed(together[0].probs))
    )
    changed = ImageState(state.state, [Image.new("RGB", (16, 12), color=(0, 240, 0))])
    assert decision.decide(changed, qs)[0].probs != together[0].probs


@pytest.mark.parametrize("family", ["gemma4", "gemma4_unified"])
def test_image_reasoning_readout_matches_full_native_continuation(family):
    lm, _, tok, backend = build(family)
    state = decode_media("Image", media())
    spec = QuestionSpec("choice", "Type?", ["letter", "receipt"])
    generator = ImageTraceGenerator(backend)
    generator.eos = set()
    messages = generator.messages_for(state, spec)
    ids, inputs = generator.prepare(messages)
    trace = generator.generate_trace(messages, 2, reserve=100)
    assert trace.input_ids == ids and trace.generated_tokens == 2
    from ayaka.reasoning_pipeline import readout_suffix

    rendered = readout_suffix(tok, spec)
    tail = torch.tensor([trace.token_ids + rendered.suffix_ids])
    full = {**inputs, "input_ids": torch.cat([inputs["input_ids"], tail], dim=1)}
    full["attention_mask"] = torch.ones_like(full["input_ids"])
    full["mm_token_type_ids"] = torch.cat(
        [inputs["mm_token_type_ids"], torch.zeros_like(tail)], dim=1
    )
    with torch.no_grad():
        logits = lm(**full).logits[0, -1, rendered.label_ids]
    assert generator.readout(trace, spec) == pytest.approx(
        logits.float().softmax(0).tolist(), abs=1e-5
    )


def test_image_settings_usage_text_calibration_isolation_and_no_leak(monkeypatch):
    _, model, tok, backend = build("gemma4_unified")

    class Forbidden:
        def should_reason(self, *args, **kwargs):
            pytest.fail("text router used on images")

        def apply(self, *args, **kwargs):
            pytest.fail("text calibration used on images")

    text = controlled_decision(model, tok, router=Forbidden(), calibration=Forbidden())
    wrapper = ImageDecision(text, backend)
    server = DecisionService(wrapper, "test")
    qs = {"a": {"type": "noul", "instructions": "Image?"}}
    off = server.handle(
        {
            "state": "Simple",
            "media": media(),
            "questions": qs,
            "options": {"reasoning": {"mode": "off"}},
        }
    )
    assert off["usage"]["output_tokens"] == 0 and off["usage"]["input_tokens"] > 70
    assert off["reasoning"]["a"]["calibration"] == "unvalidated"
    assert (
        server.handle({"state": "Simple", "media": media(), "questions": qs})["reasoning"]["a"][
            "route"
        ]
        == "direct"
    )
    # Force high, but terminate normally at the first native EOS: no speculative baseline.
    spec = QuestionSpec("noul", "Image?", ["false", "true"])
    gen = wrapper.images.generator
    probe = gen.generate_trace(gen.messages_for(decode_media("Simple", media()), spec), 1)
    gen.eos = {probe.token_ids[0]}
    seen = []
    native_generate = gen.generate_trace

    def generate(messages, budget, reserve=0):
        seen.append(budget)
        result = native_generate(messages, budget, reserve)
        result.text = "Complete"
        return result

    monkeypatch.setattr(gen, "generate_trace", generate)
    monkeypatch.setattr(
        wrapper.images.original, "decide", lambda *a, **k: pytest.fail("speculative direct")
    )
    forced = server.handle(
        {
            "state": "Simple",
            "media": media(),
            "questions": qs,
            "options": {"reasoning": {"mode": "on", "effort": "high"}},
        }
    )
    assert seen == [1024]
    assert forced["reasoning"]["a"]["route"] == "reasoned"
    assert forced["usage"]["reasoning_tokens"] == 1


def test_image_rejects_invalid_media_and_context_without_silent_truncation():
    with pytest.raises(BadRequest, match="image backend"):
        DecisionService(object(), "text").handle(
            {"media": media(), "questions": {"a": {"type": "noul"}}}
        )
    for value in [
        [],
        [{"type": "audio"}],
        [{**media()[0], "data": "@@@"}],
        [{**media()[0], "mime_type": "image/jpeg"}],
    ]:
        with pytest.raises(ValueError):
            decode_media("", value)
    _, _, _, backend = build("gemma4_unified")
    decision = NativeImageDecision(backend, max_seq_len=8)
    with pytest.raises(ValueError, match="no truncation"):
        decision.decide(decode_media("", media()), [QuestionSpec("noul", "Hi?", ["no", "yes"])])
    gen = ImageTraceGenerator(backend)
    gen.max_context = 8
    with pytest.raises(TraceFailure) as exc:
        gen.generate_trace(
            gen.messages_for(decode_media("", media()), QuestionSpec("noul", "?", ["no", "yes"])),
            1024,
        )
    assert exc.value.trace.generated_tokens == 0
    assert exc.value.trace.finish_reason == "context_limit"


def test_large_image_choices_encode_media_once_and_count_rerank(monkeypatch):
    _, model, _, backend = build("gemma4_unified")
    decision = NativeImageDecision(backend)
    calls = []
    readouts = []
    native = backend.prefill
    forward = model.forward

    def prefill(inputs):
        calls.append(inputs["input_ids"].shape[1])
        return native(inputs)

    monkeypatch.setattr(backend, "prefill", prefill)

    def count_forward(batch, **kwargs):
        readouts.append(batch.input_ids.numel())
        return forward(batch, **kwargs)

    monkeypatch.setattr(model, "forward", count_forward)
    state = decode_media("Image", media())
    spec = QuestionSpec("choice", "Which?", [f"option {i}" for i in range(28)])
    result = decision.decide(state, [spec])[0]
    assert len(calls) == 1 and len(result.probs) == 28
    assert sum(result.probs) == pytest.approx(1)
    assert len(readouts) == 2
    count = calls[0] + sum(readouts)
    assert decision.input_counts(state, [spec]) == [count]
    assert decision._context is None
    # Text head temperatures must not change this image readout.
    with torch.no_grad():
        model.temperature.fill_(7)
    assert decision.decide(state, [spec])[0].probs == pytest.approx(result.probs)


@pytest.mark.parametrize("family", ["gemma4", "gemma4_unified"])
def test_saved_native_processor_and_adapter_load_end_to_end(tmp_path, family):
    from dataclasses import asdict, replace

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        Gemma4AudioFeatureExtractor,
        Gemma4Processor,
        Gemma4UnifiedAudioFeatureExtractor,
        Gemma4UnifiedProcessor,
        Gemma4UnifiedVideoProcessor,
        Gemma4VideoProcessor,
        PreTrainedTokenizerFast,
    )

    from ayaka.checkpoint import apply_lora, save_checkpoint
    from ayaka.multimodal import load_image_decision

    lm, model, _, backend = build(family)
    source, checkpoint = tmp_path / "source", tmp_path / "checkpoint"
    vocabulary = [f"token_{i}" for i in range(512)]
    for i, value in {
        0: "[PAD]",
        1: "[EOS]",
        2: "[BOS]",
        3: "[UNK]",
        4: "<|turn>",
        5: "<turn|>",
        6: "Yes",
        7: "No",
        508: "<boi>",
        509: "<eoi>",
        510: "<image>",
        500: "<audio>",
        501: "<boa>",
        502: "<eoa>",
    }.items():
        vocabulary[i] = value
    for i, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", 20):
        vocabulary[i] = char
    raw = Tokenizer(models.WordLevel({v: i for i, v in enumerate(vocabulary)}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        extra_special_tokens={
            "image_token": "<image>",
            "boi_token": "<boi>",
            "eoi_token": "<eoi>",
            "audio_token": "<audio>",
            "boa_token": "<boa>",
            "eoa_token": "<eoa>",
        },
    )
    cls = Gemma4Processor if family == "gemma4" else Gemma4UnifiedProcessor
    processor = cls(
        feature_extractor=Gemma4AudioFeatureExtractor()
        if family == "gemma4"
        else Gemma4UnifiedAudioFeatureExtractor(),
        image_processor=backend.processor.images,
        tokenizer=tok,
        video_processor=Gemma4VideoProcessor()
        if family == "gemma4"
        else Gemma4UnifiedVideoProcessor(),
    )
    lm.save_pretrained(source)
    processor.save_pretrained(source)
    model.cfg = replace(model.cfg, backbone=str(source))
    apply_lora(model)
    save_checkpoint(model, str(checkpoint), {"test_config": asdict(model.cfg)})
    loaded = load_image_decision(str(checkpoint), dtype=torch.float32)
    assert loaded.images.original.backend.native.language_model is not None
    result = DecisionService(loaded, "loaded").handle(
        {
            "state": "image",
            "media": media(),
            "questions": {"a": {"type": "noul"}},
            "options": {"reasoning": {"mode": "off"}},
        }
    )
    assert result["usage"]["input_tokens"] > 70
    assert 0 <= result["answers"]["a"]["noul"] <= 1
    assert loaded.images.original.backend.native.language_model is loaded.model.backbone
