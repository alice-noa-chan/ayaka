"""Differentiable native image SFT; frozen vision and the same text LoRA/readout."""

from dataclasses import replace

import torch

from ..collate import EncodedQuestion
from ..model.electra import PRIMITIVE_INDEX
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


def image_forward(backend, item, batch):
    """Full native row with grad enabled, including image-bidirectional attention."""
    model = backend.model
    backend.native.language_model = model.backbone
    backend.native.eval()  # frozen vision stays deterministic
    model.backbone.train(model.training)
    device, dtype = model.embed_weight().device, model.embed_weight().dtype
    inputs = {
        k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
        for k, v in item.native_inputs.items()
    }
    prefix = inputs["input_ids"]
    if not torch.equal(batch.input_ids[:, : prefix.shape[1]], prefix):
        raise ValueError("native media prefix does not match the training row")
    inputs["input_ids"], inputs["attention_mask"] = batch.input_ids, batch.attention_mask
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros_like(batch.input_ids[:, prefix.shape[1] :])],
            dim=1,
        )
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
