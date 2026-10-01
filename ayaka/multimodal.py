"""Native Gemma 4 image prefixes with isolated text decisions and reasoning.

Images are explicit base64 payloads; no URL fetching or server filesystem access.
Text calibration/router artifacts are deliberately ineligible for image inputs.
"""

from __future__ import annotations

import base64
import binascii
import copy
import io
import warnings
from dataclasses import dataclass

import torch

from .collate import EncodedQuestion, suffix_rows
from .model.electra import PRIMITIVE_INDEX, ElectraDecisionModel
from .model.ragged import ragged_softmax
from .primitives import Decision
from .prompt import render_prefix, render_question, render_state
from .reasoning_pipeline import ControlledDecision, TraceGenerator, trace_messages

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_HTTP_BYTES = 24 * 1024 * 1024


@dataclass
class ImageState:
    state: object
    images: list
    is_multimodal = True


def decode_media(state, media):
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not isinstance(media, list) or not 1 <= len(media) <= 4:
        raise ValueError("media must contain 1–4 images")
    images, total = [], 0
    formats = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
    for item in media:
        if not isinstance(item, dict) or set(item) != {"type", "mime_type", "data"}:
            raise ValueError("media needs exactly type, mime_type, and base64 data")
        if (
            item["type"] != "image"
            or not isinstance(item["mime_type"], str)
            or item["mime_type"] not in formats
        ):
            raise ValueError("only PNG, JPEG and WebP images are supported")
        data = item["data"]
        if not isinstance(data, str) or len(data) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
            raise ValueError("image exceeds the 8 MiB limit")
        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("invalid image base64") from exc
        total += len(raw)
        if len(raw) > MAX_IMAGE_BYTES or total > MAX_TOTAL_BYTES:
            raise ValueError("image payload exceeds byte limits")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format != formats[item["mime_type"]]:
                        raise ValueError("image MIME type does not match its contents")
                    if (
                        image.width * image.height > MAX_PIXELS
                        or getattr(image, "n_frames", 1) != 1
                    ):
                        raise ValueError(
                            "image must be a single frame of at most 16 million pixels"
                        )
                    image.load()
                    images.append(ImageOps.exif_transpose(image).convert("RGB"))
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise ValueError("invalid or oversized image") from exc
    return ImageState(state, images)


class ImageBackend:
    def __init__(self, native, processor, model, tok, *, processing_device=None):
        if native.config.model_type not in {"gemma4", "gemma4_unified"}:
            raise ValueError("native image backend supports Gemma 4 and Gemma 4 Unified")
        if getattr(native.config, "vision_config", None) is None:
            raise ValueError("checkpoint has no native image components")
        self.native, self.processor, self.model, self.tok = native, processor, model, tok
        self.processing_device = processing_device

    def prepare(self, text, state):
        marker = self.processor.image_token
        # Reject literal media control tokens in supplied data, rather than mismatch images.
        if marker in text:
            raise ValueError("state contains a reserved image token")
        text += "\n" + "\n".join([marker] * len(state.images)) + "\n"
        return self.process(text, state)

    def process(self, text, state):
        inputs = self.processor(
            text=[text], images=[state.images], add_special_tokens=False, return_tensors="pt"
        )
        if inputs["input_ids"].shape[0] != 1 or not bool(inputs["attention_mask"].all()):
            raise ValueError("image prefix must be one unpadded sequence")
        dev = self.processing_device or self.model.embed_weight().device
        dtype = self.model.embed_weight().dtype
        return {
            k: v.to(device=dev, dtype=dtype if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()
        }

    @torch.inference_mode()
    def prefill(self, inputs):
        # Loading a text LoRA may replace the module; keep exactly that active adapter.
        self.native.language_model = self.model.backbone
        self.native.eval()
        return self.native(**inputs, use_cache=True, return_dict=True)


class NativeImageDecision(Decision):
    def __init__(self, backend, max_seq_len=None):
        super().__init__(backend.model, backend.tok, max_seq_len)
        self.backend = backend
        self.max_seq_len = min(self.max_seq_len, backend.model.text_config.max_position_embeddings)
        self._context = None
        self._last_inputs = None

    def encode(self, state, views):
        text = self.tok.decode(render_prefix(state.state, self.tok))
        inputs = self.backend.prepare(text, state)
        prefix = inputs["input_ids"][0].tolist()
        items = [
            EncodedQuestion(
                prefix, render_question(v, self.tok, self.max_labels), PRIMITIVE_INDEX[v.type]
            )
            for v in views
        ]
        if any(len(prefix) + len(it.rendered.suffix_ids) > self.max_seq_len for it in items):
            raise ValueError(
                "image prefix and question exceed the context budget; no truncation applied"
            )
        return inputs, prefix, items

    def input_counts(self, state, questions):
        key = (id(state), tuple(id(q) for q in questions))
        if self._last_inputs is not None and self._last_inputs[0] == key:
            return self._last_inputs[1]
        _, prefix, items = self.encode(state, [q.view() for q in questions])
        return [
            len(it.rendered.suffix_ids) + (len(prefix) if j == 0 else 0)
            for j, it in enumerate(items)
        ]

    @torch.no_grad()
    def _run(self, state, views, device):
        context = self._context if self._context is not None else {"cache": None, "calls": []}
        first = context["cache"] is None
        if first:
            inputs, prefix, items = self.encode(state, views)
            context["prefix"] = prefix
            context["cache"] = self.backend.prefill(inputs).past_key_values
        else:
            prefix = context["prefix"]
            items = [
                EncodedQuestion(
                    prefix, render_question(v, self.tok, self.max_labels), PRIMITIVE_INDEX[v.type]
                )
                for v in views
            ]
            if any(len(prefix) + len(it.rendered.suffix_ids) > self.max_seq_len for it in items):
                raise ValueError("image readout exceeds context budget")
        context["calls"].append(
            [
                len(it.rendered.suffix_ids) + (len(prefix) if first and j == 0 else 0)
                for j, it in enumerate(items)
            ]
        )
        cache = context["cache"]
        results = []
        for item in items:
            batch = suffix_rows([item], len(prefix), self.tok.pad_id).to(self._device(device))
            out = self.model(batch, past_key_values=copy.deepcopy(cache), apply_temperature=False)
            results.append(ragged_softmax(out.logits, out.cand_cu).tolist())
        return results

    def decide(self, state, questions, device=None):
        self._context = {"cache": None, "calls": []}
        try:
            results = super().decide(state, questions, device)
            calls = self._context["calls"]
            counts = list(calls[0])
            large = [
                i
                for i, q in enumerate(questions)
                if q.type == "choice" and len(q.candidates) > self.max_labels
            ]
            for i, rerank in zip(large, calls[1:], strict=True):
                counts[i] += rerank[0]
            self._last_inputs = ((id(state), tuple(id(q) for q in questions)), counts)
            return results
        finally:
            self._context = None


class ImageTraceGenerator(TraceGenerator):
    def __init__(self, backend):
        super().__init__(backend.model, backend.tok, apply_temperature=False)
        self.backend = backend

    def messages_for(self, state, spec):
        return state, trace_messages(state.state, spec)

    def prepare(self, messages):
        from .evidence_generation import chat_ids

        state, text_messages = messages
        # Place image evidence inside the user turn, before opening the assistant.
        message = copy.deepcopy(text_messages)
        marker = self.backend.processor.image_token
        if marker in render_state(state.state):
            raise ValueError("state contains a reserved image token")
        message[0]["content"] += "\n" + "\n".join([marker] * len(state.images))
        text = self.tok.decode(chat_ids(self.tok, message))
        # Markers are already in the chat; processing is otherwise the same native path.
        inputs = self.backend.process(text, state)
        return inputs["input_ids"][0].tolist(), inputs

    def prefill(self, ids, payload):
        out = self.backend.prefill(payload)
        return out.last_hidden_state[:, -1], out.past_key_values


class ImageDecision:
    supports_reasoning = True
    supports_images = True

    def __init__(self, text_decision, backend):
        self.text = text_decision
        self.model, self.tok = text_decision.model, text_decision.tok
        self.max_seq_len = text_decision.max_seq_len
        self.generator = text_decision.generator  # text-only proposals use the normal generator
        self.images = ControlledDecision(
            NativeImageDecision(backend, self.max_seq_len), ImageTraceGenerator(backend)
        )

    def prepare_media(self, state, media):
        return decode_media(state, media)

    def decide(self, state, questions, device=None, reasoning=None):
        if not isinstance(state, ImageState):
            return self.text.decide(state, questions, device=device, reasoning=reasoning)
        results = self.images.decide(state, questions, device=device, reasoning=reasoning)
        for result in results:
            result.extras["reasoning"].update(modality="image", calibration="unvalidated")
            if result.extras["reasoning"]["finish_reason"] == "no_validated_router":
                result.extras["reasoning"]["finish_reason"] = "no_validated_image_router"
        return results


def load_image_decision(
    path,
    device="cpu",
    dtype=torch.bfloat16,
    max_seq_len=None,
    router=None,
    calibration=None,
    *,
    trainable=False,
):
    """Load native image components and the existing text adapter/head once."""
    import os

    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    from .checkpoint import load_config, load_head
    from .reasoning_pipeline import controlled_decision
    from .tokenization import HFTokenizer

    cfg = load_config(path)
    config = AutoConfig.from_pretrained(cfg.backbone, revision=cfg.backbone_revision)
    if config.model_type not in {"gemma4", "gemma4_unified"} or config.vision_config is None:
        raise ValueError("--images requires a native Gemma 4 image checkpoint")
    lm = AutoModelForImageTextToText.from_pretrained(
        cfg.backbone,
        revision=cfg.backbone_revision,
        dtype=dtype,
        attn_implementation="sdpa",
        device_map={"": str(device)},
    ).eval()
    native = lm.model
    text = native.language_model
    if lm.get_output_embeddings().weight is not text.get_input_embeddings().weight:
        text.add_module("_ayaka_lm_head", lm.get_output_embeddings())
    model = ElectraDecisionModel(cfg, text, config.text_config).to(device).eval()
    adapter = os.path.join(path, "adapter")
    if os.path.isdir(adapter):
        from peft import PeftModel

        model.backbone = PeftModel.from_pretrained(
            model.backbone, adapter, is_trainable=trainable
        ).eval()
    model.load_head_state_dict(load_head(path, device))
    processor = AutoProcessor.from_pretrained(cfg.backbone, revision=cfg.backbone_revision)
    tok = HFTokenizer(processor.tokenizer, cfg.backbone)
    backend = ImageBackend(native, processor, model, tok)
    total = sum(
        p.numel() for p in {id(p): p for p in [*native.parameters(), *model.parameters()]}.values()
    )
    if total > 14_000_000_000:
        raise ValueError("native image model and decision adapter exceed the 14B parameter limit")
    return ImageDecision(controlled_decision(model, tok, max_seq_len, router, calibration), backend)
