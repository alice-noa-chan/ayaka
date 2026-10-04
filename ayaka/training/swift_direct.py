"""Direct student rows on Swift's actual serving prompt and canonical tokens.

The default Ayaka encoder is untouched. This opt-in encoder preserves original
candidate/gold order while mapping Swift's displayed letters (including sorted
integer Score levels) back to those candidates. No model or trace is loaded.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, replace

from ..collate import EncodedQuestion
from ..eval.read_artifact import fingerprint
from ..prompt import RenderedQuestion, render_prefix, render_question
from ..swift.prompt import parse_question, validate_prompt_variant
from ..swift.readers import READOUT
from .batching import FLAGGED_EVIDENCE, TrainItem, _noul_canonical, question_view, sample_to_items
from .swift_evidence import prepare_swift_evidence_inputs, validate_swift_evidence_inputs

VERSION = "ayaka-swift-direct-inputs-1"


def normalize_input_encoding(value=None):
    """Explicit per-run selection; serving defaults and v1 are unchanged."""
    value = {"encoder": "ayaka_segmented"} if value is None else copy.deepcopy(value)
    if not isinstance(value, dict):
        raise ValueError("direct input encoding must be an object")
    if value.get("encoder") == "ayaka_segmented" and set(value) == {"encoder"}:
        return value
    allowed = {"encoder", "prompt_variant", "state_format", "chat_template_kwargs"}
    if value.get("encoder") != "swift_canonical" or set(value) - allowed:
        raise ValueError("unsupported direct input encoder or encoding fields")
    value.setdefault("prompt_variant", "min")
    value.setdefault("state_format", "pretty")
    value.setdefault("chat_template_kwargs", {"enable_thinking": False})
    validate_prompt_variant(value["prompt_variant"])
    if not isinstance(value["state_format"], str) or value["state_format"] not in {
        "pretty",
        "compact",
    }:
        raise ValueError("Swift direct state format must be pretty or compact")
    kwargs = value["chat_template_kwargs"]
    if (
        not isinstance(kwargs, dict)
        or kwargs.get("enable_thinking", False)
        or any(
            key in kwargs
            for key in ("return_dict", "return_tensors", "tokenize", "add_generation_prompt")
        )
    ):
        raise ValueError("Swift direct chat kwargs must retain non-thinking canonical transport")
    fingerprint(value)  # require a serializable immutable preparation recipe
    return value


def input_serving_recipe(tok, input_encoding=None):
    encoding = normalize_input_encoding(input_encoding)
    if encoding["encoder"] == "ayaka_segmented":
        return {"version": "ayaka-segmented-direct-inputs-1", **encoding}
    native = getattr(tok, "hf", tok)
    backend = getattr(native, "backend_tokenizer", None)
    if not getattr(native, "is_fast", False) or backend is None:
        raise ValueError("Swift direct input requires a fast offset tokenizer")
    return {
        "version": VERSION,
        "readout": READOUT,
        **{key: value for key, value in encoding.items() if key != "encoder"},
        "chat_template_sha256": fingerprint(native.chat_template),
        "tokenizer_sha256": fingerprint(backend.to_str()),
    }


def encode_direct_sample(sample, tok, cfg, *, input_encoding=None, context_limit=None):
    """Full original inputs on the declared encoder, never silent truncation."""
    encoding = normalize_input_encoding(input_encoding)
    limit = cfg.max_seq_len if context_limit is None else context_limit
    if encoding["encoder"] == "swift_canonical":
        return swift_sample_to_items(
            sample,
            tok,
            cfg,
            context_limit=limit,
            **{k: v for k, v in encoding.items() if k != "encoder"},
        )
    prefix = render_prefix(sample.state, tok)
    for q in sample.questions:
        rendered = render_question(question_view(_noul_canonical(q)), tok, cfg.max_label_candidates)
        if len(prefix) + len(rendered.suffix_ids) > limit:
            raise ValueError("complete original direct input does not fit; refuse truncation")
    return sample_to_items(sample, tok, replace(cfg, max_seq_len=limit))


def direct_readout_binding(item):
    """Compact exact direct-probability observation contract for saved teachers."""
    if item.direct_input_binding is None:
        return None
    validate_direct_input_items([item])
    return copy.deepcopy(
        {
            key: item.direct_input_binding[key]
            for key in (
                "version",
                "recipe_sha256",
                "candidate_ids",
                "displayed_candidate_ids",
                "messages_sha256",
                "input_token_ids_sha256",
                "canonical_token_ids_sha256",
            )
        }
    )


def swift_question(question):
    """Return API-equivalent criteria and wire-label -> original-id mapping.

    Swift's Score API uses integer level keys. Arbitrary dataset candidate IDs
    remain valid, but fractional levels cannot silently change meaning.
    """
    q = _noul_canonical(question)
    labels = [c.id for c in q.candidates]
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Swift direct rows require nonempty string candidate IDs")
    if q.type == "score":
        if any(c.ordinal != int(c.ordinal) for c in q.candidates):
            raise ValueError("Swift direct Score levels must be integers; refuse rounding")
        labels = [str(int(c.ordinal)) for c in q.candidates]
    raw = {
        "type": q.type,
        "instructions": q.instruction,
        "criteria": {label: c.description for label, c in zip(labels, q.candidates, strict=True)},
    }
    return q, parse_question(raw), dict(zip(labels, (c.id for c in q.candidates), strict=True))


def _bound_values(item):
    return {
        "encoding": asdict(item.enc),
        "target": item.target,
        "ordinals": [str(level) for level in item.ordinals],
        "type": item.type,
        "sample_id": item.sample_id,
    }


def swift_sample_to_items(
    sample,
    tok,
    cfg,
    *,
    prompt_variant="min",
    state_format="pretty",
    chat_template_kwargs=None,
    context_limit=None,
):
    """Encode complete isolated questions with the existing Swift renderer.

    Every row uses the full chat template, never separately retokenized parts.
    Prefix IDs share an object only when the exact state-only prefix agrees.
    Supervision stays in original candidate order; display_order records the
    actual letter order. Metadata teacher/trace inputs are deliberately ignored.
    """
    if cfg.readout != "lm":
        raise ValueError("Swift direct encoding requires the native LM-only readout")
    if sample.metadata.get("modality", "text") != "text" or any(
        key in sample.metadata for key in ("media", "proposal_supervision")
    ):
        raise ValueError("Swift direct encoding requires original text-only inputs")
    native = getattr(tok, "hf", tok)
    questions = [swift_question(q) for q in sample.questions]
    prepared = prepare_swift_evidence_inputs(
        sample.state,
        [wire for _, wire, _ in questions],
        native,
        context_limit=cfg.max_seq_len if context_limit is None else context_limit,
        prompt_variant=prompt_variant,
        state_format=state_format,
        chat_template_kwargs=chat_template_kwargs,
    )
    validate_swift_evidence_inputs(prepared)
    prefix = list(prepared.inputs.prefix_ids)
    recipe = {
        "version": VERSION,
        **{
            key: copy.deepcopy(prepared.recipe[key])
            for key in (
                "readout",
                "prompt_variant",
                "state_format",
                "chat_template_kwargs",
                "chat_template_sha256",
                "tokenizer_sha256",
            )
        },
    }
    items = []
    for (q, wire, semantic), encoded, retained in zip(
        questions, prepared.inputs.questions, prepared.recipe["questions"], strict=True
    ):
        original_ids = [c.id for c in q.candidates]
        displayed_ids = [semantic[label] for label in wire.labels]
        indices = [displayed_ids.index(candidate_id) for candidate_id in original_ids]
        rendered = RenderedQuestion(
            suffix_ids=list(encoded.suffix_ids),
            option_spans=[encoded.option_spans[i] for i in indices],
            label_ids=[encoded.label_ids[i] for i in indices],
            display_order=[original_ids.index(candidate_id) for candidate_id in displayed_ids],
        )
        item = TrainItem(
            enc=EncodedQuestion(prefix, rendered, encoded.primitive),
            target=[q.target_distribution.get(c.id, 0.0) for c in q.candidates],
            ordinals=[
                c.ordinal if c.ordinal is not None else i for i, c in enumerate(q.candidates)
            ],
            type=q.type,
            flagged=sample.metadata.get("evidence_state", "intact") in FLAGGED_EVIDENCE,
            teacher=None,
            sample_id=str(sample.metadata.get("source_example_id", "")),
            family=sample.metadata.get("task_family", "unknown"),
            source=sample.metadata.get("source", "synthetic"),
            direct_distillation=True,
        )
        item.direct_input_binding = {
            "version": VERSION,
            "recipe": copy.deepcopy(recipe),
            "recipe_sha256": fingerprint(recipe),
            "question_id": q.id,
            "candidate_ids": original_ids,
            "displayed_candidate_ids": displayed_ids,
            "messages_sha256": retained["messages_sha256"],
            "input_token_ids_sha256": retained["input_token_ids_sha256"],
            "canonical_token_ids_sha256": retained["canonical_token_ids_sha256"],
            "row_sha256": fingerprint(_bound_values(item)),
        }
        item.direct_input_binding["binding_sha256"] = fingerprint(item.direct_input_binding)
        items.append(item)
    validate_direct_input_items(items)
    return items


def validate_direct_input_items(items):
    """Reject changed/mixed Swift rows before any training or prediction forward."""
    bindings = [item.direct_input_binding for item in items]
    if not any(binding is not None for binding in bindings):
        return
    if any(binding is None for binding in bindings):
        raise ValueError("Swift direct rows must not mix the legacy Ayaka input encoder")
    recipes = set()
    for item, binding in zip(items, bindings, strict=True):
        if (
            not isinstance(binding, dict)
            or binding.get("version") != VERSION
            or fingerprint(
                {key: value for key, value in binding.items() if key != "binding_sha256"}
            )
            != binding.get("binding_sha256")
            or fingerprint(binding.get("recipe")) != binding.get("recipe_sha256")
            or fingerprint(_bound_values(item)) != binding.get("row_sha256")
            or fingerprint(item.enc.prefix_ids + item.enc.rendered.suffix_ids)
            != binding.get("input_token_ids_sha256")
        ):
            raise ValueError("Swift direct prompt, canonical tokens or gold binding changed")
        if not item.direct_distillation or any(
            getattr(item, key) is not None
            for key in (
                "reasoning_positions",
                "reasoning_labels",
                "proposal_input_ids",
                "proposal_positions",
                "proposal_labels",
                "native_inputs",
            )
        ):
            raise ValueError("Swift direct rows must not contain trace, proposal or image inputs")
        recipes.add(binding["recipe_sha256"])
    if len(recipes) != 1:
        raise ValueError("Swift direct batch must use one exact serving input recipe")
