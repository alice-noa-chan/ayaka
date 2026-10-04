"""Evidence features on Swift's exact canonical serving prompt and token ids.

Only existing public Swift helpers are read. No serving default is changed and
no model/tokenizer is loaded. Token offsets split a state-only shared prefix
out of the complete Swift input without retokenizing or truncating its rows.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ..eval.read_artifact import fingerprint
from ..model.electra import PRIMITIVE_INDEX
from ..swift.prompt import Question, parse_question, render_question
from ..swift.readers import READOUT, token_input
from .evidence_features import (
    EvidenceInputs,
    EvidenceQuestionInputs,
    _validate_inputs,
    extract_evidence_features,
)
from .tokenizer_identity import backend_fingerprints, configuration_fingerprint


@dataclass(frozen=True)
class SwiftEvidenceInputs:
    inputs: EvidenceInputs
    recipe: dict
    recipe_sha256: str


def _content_position(prompt, content):
    # Some native templates trim only edge whitespace from user content. No
    # other rewriting is accepted, since it would obscure the evidence boundary.
    for value, removed in ((content, 0), (content.strip(), len(content) - len(content.lstrip()))):
        if value and prompt.count(value) == 1:
            return prompt.index(value), removed, len(value)
    raise ValueError("Swift chat template must preserve identifiable user content")


def _token_span(offsets, start, end):
    indices = [i for i, (a, b) in enumerate(offsets) if b > start and a < end and a < b]
    if not indices:
        raise ValueError("Swift candidate description has no aligned tokens")
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("Swift candidate offsets must be contiguous")
    return indices[0], indices[-1] + 1


def _description_texts(question, prompt_variant, state_format):
    """Derive option decorations and footer from the actual serving renderer."""
    blank, _ = render_question(
        "",
        Question(question.type, "", [""], [""]),
        prompt_variant=prompt_variant,
        state_format=state_format,
    )
    before, separator, footer = blank[-1]["content"].partition("A. ")
    if not separator:
        raise ValueError("Swift renderer does not expose a single-option description boundary")
    prefix = before + separator
    result = []
    for label, description in zip(question.labels, question.descriptions, strict=True):
        messages, _ = render_question(
            "",
            Question(question.type, "", [label], [description]),
            prompt_variant=prompt_variant,
            state_format=state_format,
        )
        user = messages[-1]["content"]
        if not user.startswith(prefix) or (footer and not user.endswith(footer)):
            raise ValueError("Swift option decorations changed the surrounding renderer")
        decorated = user[len(prefix) : len(user) - len(footer)]
        if not decorated.endswith(description):
            raise ValueError("Swift candidate description is not preserved by the renderer")
        result.append((decorated, len(decorated) - len(description)))
    return result


def prepare_swift_evidence_inputs(
    state,
    questions,
    tokenizer,
    *,
    context_limit=8192,
    prompt_variant="min",
    state_format="pretty",
    chat_template_kwargs=None,
):
    """Preserve Swift parse/order/render/tokenization and its canonical labels.

    A fast tokenizer with reliable character offsets is required. A token that
    straddles the state/question boundary stays in the suffix, so shared memory
    never includes question/options. Candidate features use overlapping token
    spans; they inherit the causal prompt's order dependence.
    """
    if type(context_limit) is not int or context_limit < 2 or not 1 <= len(questions) <= 64:
        raise ValueError("require a valid context limit and 1..64 Swift questions")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Swift evidence alignment requires a fast offset tokenizer")
    kwargs = (
        {"enable_thinking": False} if chat_template_kwargs is None else dict(chat_template_kwargs)
    )
    if kwargs.get("enable_thinking", False):
        raise ValueError("Swift evidence training requires direct non-thinking prompts")
    if any(
        key in kwargs
        for key in ("return_dict", "return_tensors", "tokenize", "add_generation_prompt")
    ):
        raise ValueError("chat kwargs must not override Swift tokenizer transport arguments")
    rows, maximum_prefix, contracts = [], [], []
    for raw in questions:
        q = raw if isinstance(raw, Question) else parse_question(raw)
        if (
            q.type not in PRIMITIVE_INDEX
            or not 2 <= len(q.labels) <= 26
            or len(q.labels) != len(q.descriptions)
            or any(not isinstance(s, str) or not s for s in q.labels)
            or len(set(q.labels)) != len(q.labels)
            or any(not isinstance(s, str) or not s.strip() for s in q.descriptions)
        ):
            raise ValueError("Swift evidence supports complete single-pass 2..26 candidate sets")
        messages, mapping = render_question(
            state, q, state_format=state_format, prompt_variant=prompt_variant
        )
        letters = list(mapping)
        # Transformers 5 returns BatchEncoding by default. This controls only
        # the capture envelope, not the rendered content or serving HF inputs.
        wire = token_input(tokenizer, messages, letters, {**kwargs, "return_dict": False})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        ids = wire["input_token_ids"]
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded["input_ids"]) != ids:
            raise ValueError("offset tokenizer ids differ from Swift's actual serving input")
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        if len(offsets) != len(ids) or any(
            len(pair) != 2
            or any(type(v) is not int for v in pair)
            or not 0 <= pair[0] <= pair[1] <= len(prompt)
            for pair in offsets
        ):
            raise ValueError("Swift tokenizer returned invalid character offsets")
        user = messages[-1]["content"]
        empty_messages, _ = render_question(
            "", q, state_format=state_format, prompt_variant=prompt_variant
        )
        tail = empty_messages[-1]["content"]
        if not user.endswith(tail) or not tail:
            raise ValueError("Swift renderer no longer exposes an exact state/question boundary")
        content_start, trimmed, retained = _content_position(prompt, user)
        state_end = content_start + len(user) - len(tail) - trimmed
        # Take a contiguous prefix, stopping before a straddling token or the
        # first token after the evidence. Template-only zero-offset tokens may
        # appear before the state; they carry no suffix evidence.
        prefix_end = 0
        for i, (a, b) in enumerate(offsets):
            if (a < b and b > state_end) or (a == b and prefix_end and a >= state_end):
                break
            prefix_end = i + 1
        if prefix_end < 1:
            raise ValueError("Swift prompt has no shareable state-only prefix")
        descriptions = _description_texts(q, prompt_variant, state_format)
        options = "\n".join(
            f"{letter}. {text}" for letter, (text, _) in zip(letters, descriptions, strict=True)
        )
        options_at = user.rfind(options)
        if options_at < len(user) - len(tail):
            raise ValueError("Swift options must be inside the question suffix")
        spans, cursor = [], content_start + options_at - trimmed
        for letter, description, (decorated, offset) in zip(
            letters, q.descriptions, descriptions, strict=True
        ):
            begin = cursor + len(f"{letter}. ") + offset
            # An edge-trimming template can remove the final option's trailing
            # whitespace. Never pool assistant-template tokens as its evidence.
            spans.append(
                _token_span(
                    offsets,
                    max(begin, content_start),
                    min(begin + len(description), content_start + retained),
                )
            )
            cursor += len(f"{letter}. ") + len(decorated) + 1
        label_ids = tuple(wire["canonical_token_ids"][letter][0] for letter in letters)
        rows.append((tuple(ids), spans, label_ids, PRIMITIVE_INDEX[q.type]))
        maximum_prefix.append(prefix_end)
        contracts.append(
            {
                "type": q.type,
                "labels": list(mapping.values()),
                "messages_sha256": fingerprint(messages),
                **wire,
            }
        )
    common = min(maximum_prefix)
    for tokens, _, _, _ in rows[1:]:
        for i in range(common):
            if tokens[i] != rows[0][0][i]:
                common = i
                break
    if common < 1:
        raise ValueError("Swift questions do not share a state-only token prefix")
    encoded_questions = tuple(
        EvidenceQuestionInputs(
            tokens[common:],
            tuple((a - common, b - common) for a, b in spans),
            label_ids,
            primitive,
            tuple(range(len(label_ids))),
        )
        for tokens, spans, label_ids, primitive in rows
    )
    inputs = EvidenceInputs(
        rows[0][0][:common], encoded_questions, context_limit, fingerprint(state)
    )
    _validate_inputs(inputs)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("Swift evidence requires exact tokenizer serialization")
    recipe = {
        "version": "ayaka-swift-evidence-inputs-1",
        "readout": READOUT,
        "prompt_variant": prompt_variant,
        "state_format": state_format,
        "chat_template_kwargs": kwargs,
        "capture_return_dict": False,
        "chat_template_sha256": fingerprint(tokenizer.chat_template),
        "tokenizer_sha256": backend_fingerprints(tokenizer)["json_string_sha256"],
        "tokenizer_config_sha256": configuration_fingerprint(tokenizer),
        "input_sha256": inputs.input_sha256,
        "questions": contracts,
        "shared_prefix_semantics": "system and state only; boundary-straddling tokens kept in suffix",
        "candidate_span_semantics": "description-overlapping tokens in the actual causal serving prompt",
    }
    return SwiftEvidenceInputs(inputs, recipe, fingerprint(recipe))


def validate_swift_evidence_inputs(prepared):
    if not isinstance(prepared, SwiftEvidenceInputs):
        raise ValueError("require bound Swift evidence inputs")
    # Revalidate the immutable inputs and retained serving rows before forward.
    if (
        fingerprint(prepared.recipe) != prepared.recipe_sha256
        or prepared.inputs.input_sha256 != prepared.recipe.get("input_sha256")
        or prepared.recipe.get("readout") != READOUT
    ):
        raise ValueError("Swift evidence serving-input binding changed")


def extract_swift_evidence_features(text, prepared, **options):
    validate_swift_evidence_inputs(prepared)
    features = extract_evidence_features(text, prepared.inputs, **options)
    features.metadata.update(
        prior_recipe=copy.deepcopy(prepared.recipe),
        prior_recipe_sha256=prepared.recipe_sha256,
        serving_readout=READOUT,
        prompt_variant=prepared.recipe["prompt_variant"],
    )
    return features
