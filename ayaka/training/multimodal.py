"""Differentiable native image SFT; frozen vision and the same text LoRA/readout."""

from dataclasses import replace

import torch

from ..collate import EncodedQuestion
from ..model.decision import PRIMITIVE_INDEX
from ..multimodal import ImageTraceGenerator, decode_media
from ..primitives import QuestionSpec
from ..prompt import render_prefix
from ..reasoning_pipeline import readout_suffix
from .batching import _noul_canonical, sample_to_items


def freeze_image_components(backend):
    """The saved text-adapter checkpoint cannot represent trained vision weights."""
    backend.native.language_model = backend.model.backbone
    text_ids = {id(p) for p in backend.model.backbone.parameters()}
    for parameter in backend.native.parameters():
        if id(parameter) not in text_ids:
            parameter.requires_grad_(False)


def image_items(sample, backend, traces=None, *, include_direct=True):
    state = decode_media(sample.state, sample.metadata.get("media"))
    cfg, tok = backend.model.cfg, backend.tok
    direct = sample_to_items(sample, tok, cfg)
    prefix_inputs = backend.prepare(tok.decode(render_prefix(sample.state, tok)), state)
    prefix = prefix_inputs["input_ids"][0].tolist()
    result = []

    def append(original, enc, payload, positions=None, labels=None):
        limit = min(cfg.max_seq_len, backend.model.text_config.max_position_embeddings)
        if len(enc.prefix_ids) + len(enc.rendered.suffix_ids) > limit:
            raise ValueError("complete image training row does not fit; do not truncate it")
        cpu = {k: v.detach().cpu().clone() for k, v in payload.items()}
        result.append(
            replace(
                original,
                enc=enc,
                native_inputs=cpu,
                reasoning_positions=positions,
                reasoning_labels=labels,
            )
        )

    generator = ImageTraceGenerator(backend)
    if traces and set(traces) - {q.id for q in sample.questions}:
        raise ValueError("trace references an unknown image question")
    for question, original in zip(sample.questions, direct, strict=True):
        q = _noul_canonical(question)
        if include_direct:
            append(original, replace(original.enc, prefix_ids=prefix), prefix_inputs)
        if not traces or q.id not in traces:
            continue
        notes = traces[q.id]
        if not isinstance(notes, str) or not notes.strip():
            raise ValueError("verified image reasoning must be non-empty")
        spec = QuestionSpec(
            q.type,
            q.instruction,
            [c.description for c in q.candidates],
            [c.ordinal for c in q.candidates] if q.type == "score" else None,
        )
        prompt, payload = generator.prepare(generator.messages_for(state, spec))
        labels = tok.encode(notes)
        eos = getattr(getattr(tok, "hf", None), "eos_token_id", getattr(tok, "eos_id", None))
        if eos is not None:
            labels.append(eos)
        if len(labels) > 1024:
            raise ValueError("image reasoning exceeds 1024 tokens; do not truncate it")
        enc = EncodedQuestion(
            prompt + labels,
            readout_suffix(tok, spec, cfg.max_label_candidates),
            PRIMITIVE_INDEX[q.type],
        )
        append(
            original,
            enc,
            payload,
            list(range(len(prompt) - 1, len(prompt) + len(labels) - 1)),
            labels,
        )
    return result


def image_forward(backend, items, batch, feature_cache=None):
    """Independent native rows in one padded batch with native vision masks."""
    if not isinstance(items, list):
        items = [items]
    model = backend.model
    backend.native.language_model = model.backbone
    backend.native.eval()  # frozen vision stays deterministic
    model.backbone.train(model.training)
    device, dtype = model.embed_weight().device, model.embed_weight().dtype
    payloads = [item.native_inputs for item in items]
    keys = set(payloads[0])
    if any(set(p) != keys for p in payloads) or keys - {
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values",
        "image_position_ids",
    }:
        raise ValueError("unsupported native image batch layout")
    inputs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
    mm = torch.zeros_like(batch.input_ids)
    for row, (item, payload) in enumerate(zip(items, payloads, strict=True)):
        prefix = payload["input_ids"][0]
        if prefix.tolist() != item.enc.prefix_ids[: len(prefix)]:
            raise ValueError("native media prefix does not match the training row")
        if "mm_token_type_ids" in payload:
            mm[row, : len(prefix)] = payload["mm_token_type_ids"][0].to(device)
    if "mm_token_type_ids" in keys:
        inputs["mm_token_type_ids"] = mm
    for key in ("pixel_values", "image_position_ids"):
        if feature_cache is not None:
            # The scoped feature provider consumes every original CPU payload.
            # Native forward only dispatches these arguments to that provider;
            # retain a nonempty payload without copying patches to GPU again.
            inputs[key] = payloads[0][key]
            continue
        width = max(p[key].shape[1] for p in payloads)
        values = []
        for payload in payloads:
            tensor = payload[key]
            pad = tensor.new_full(
                (tensor.shape[0], width - tensor.shape[1], *tensor.shape[2:]),
                -1 if key == "image_position_ids" else 0,
            )
            values.append(torch.cat([tensor, pad], dim=1))
        tensor = torch.cat(values)
        inputs[key] = tensor.to(
            device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype
        )
    from contextlib import nullcontext

    context = feature_cache.use(payloads) if feature_cache is not None else nullcontext()
    with context:
        outputs = backend.native(
            **inputs,
            use_cache=False,
            return_dict=True,
            output_hidden_states=model.span_layer is not None,
        )
    hidden = outputs.last_hidden_state
    spans = outputs.hidden_states[model.span_layer] if model.span_layer is not None else hidden
    norm = model.text_model().norm if model.span_layer is not None else None
    return hidden, spans, norm
