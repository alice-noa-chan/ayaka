"""Opt-in text feature extraction for evidence-head experiments.

Uses the supplied, already loaded bare backbone. No downloads, model loading,
reasoning generation, production-default changes or hidden state truncation.
Bundles are unbound until a caller verifies weights/recipe/splits/targets.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from numbers import Real

import torch

from ayaka.backbone import native_logits, output_rows
from ayaka.eval.read_artifact import fingerprint
from ayaka.model.decision import PRIMITIVE_INDEX, at_least_fp32
from ayaka.prompt import QuestionView, render_prefix, render_question

from ..input_errors import ContextLimitError
from .evidence_cache import branch_evidence_cache, dynamic_kv_bytes


@dataclass(frozen=True)
class EvidenceQuestionInputs:
    suffix_ids: tuple[int, ...]
    option_spans: tuple[tuple[int, int], ...]
    label_ids: tuple[int, ...]
    primitive: int
    display_order: tuple[int, ...]


@dataclass(frozen=True)
class EvidenceInputs:
    prefix_ids: tuple[int, ...]
    questions: tuple[EvidenceQuestionInputs, ...]
    context_limit: int
    state_sha256: str

    @property
    def input_sha256(self) -> str:
        return fingerprint(asdict(self))


@dataclass
class EvidenceFeatures:
    tensors: dict[str, torch.Tensor]
    metadata: dict

    def head_inputs(self) -> dict[str, torch.Tensor]:
        return dict(self.tensors)


class FeatureExtractionError(ValueError):
    """Runtime failure with attempted work recorded; never triggers a retry."""

    def __init__(self, error, *, mode, calls, tokens, planned_tokens):
        super().__init__(f"{mode} feature extraction failed: {error}")
        self.progress = {
            "mode": mode,
            "attempted_forward_calls": calls,
            "attempted_forward_tokens": tokens,
            "planned_forward_tokens": planned_tokens,
            "complete": False,
            "reason": str(error),
        }


def prepare_evidence_inputs(
    state,
    questions: list[QuestionView],
    tokenizer,
    *,
    context_limit: int = 8192,
) -> EvidenceInputs:
    """Render the entire state; reject overlength/unsupported native candidates.

    This uses Ayaka's existing segmented prompt/tokenizer recipe, not Swift's
    messages template. Model comparisons must not conflate those two recipes.
    """
    if type(context_limit) is not int or context_limit < 2:
        raise ValueError("context_limit must be an integer >= 2")
    if not questions or len(questions) > 64:
        raise ValueError("require 1–64 questions per record")
    state_sha256 = fingerprint(state)
    for question in questions:
        if question.type not in PRIMITIVE_INDEX:
            raise ValueError("unsupported primitive")
        n = len(question.descriptions)
        if not 2 <= n <= 26:
            raise ValueError("native feature extraction requires 2–26 candidates")
        if question.type == "noul" and n != 2:
            raise ValueError("Noul requires exactly two candidates")
        if question.type == "score" and question.ordinals is not None:
            levels = question.ordinals
            if (
                len(levels) != n
                or any(
                    not isinstance(x, Real) or isinstance(x, bool) or not math.isfinite(x)
                    for x in levels
                )
                or len(set(levels)) != n
            ):
                raise ValueError("Score requires distinct finite ordinal levels")
    prefix = tuple(render_prefix(state, tokenizer))  # never supply max_state_tokens
    encoded = []
    for question in questions:
        rendered = render_question(question, tokenizer)
        if rendered.label_ids is None or len(set(rendered.label_ids)) != len(rendered.label_ids):
            raise ValueError("native candidates need distinct single-token readout ids")
        encoded.append(
            EvidenceQuestionInputs(
                tuple(rendered.suffix_ids),
                tuple(rendered.option_spans),
                tuple(rendered.label_ids),
                PRIMITIVE_INDEX[question.type],
                tuple(rendered.display_order),
            )
        )
    inputs = EvidenceInputs(prefix, tuple(encoded), context_limit, state_sha256)
    _validate_inputs(inputs)
    return inputs


def _validate_inputs(inputs: EvidenceInputs) -> None:
    if (
        not isinstance(inputs, EvidenceInputs)
        or type(inputs.context_limit) is not int
        or inputs.context_limit < 2
    ):
        raise ValueError("require typed evidence inputs and a valid context limit")
    if (
        type(inputs.prefix_ids) is not tuple
        or type(inputs.questions) is not tuple
        or not inputs.prefix_ids
        or not 1 <= len(inputs.questions) <= 64
    ):
        raise ValueError("require nonempty state prefix and 1–64 questions")
    if (
        not isinstance(inputs.state_sha256, str)
        or len(inputs.state_sha256) != 64
        or any(x not in "0123456789abcdef" for x in inputs.state_sha256)
    ):
        raise ValueError("state fingerprint must be a lowercase SHA256")
    if any(type(token) is not int or token < 0 for token in inputs.prefix_ids):
        raise ValueError("prefix token ids must be nonnegative integers")
    for question in inputs.questions:
        if not isinstance(question, EvidenceQuestionInputs):
            raise ValueError("require typed question inputs")
        if any(
            type(value) is not tuple
            for value in (
                question.suffix_ids,
                question.option_spans,
                question.label_ids,
                question.display_order,
            )
        ):
            raise ValueError("question input sequences must be immutable tuples")
        if (
            not question.suffix_ids
            or len(inputs.prefix_ids) + len(question.suffix_ids) >= inputs.context_limit
        ):
            raise ContextLimitError(
                "entire state/question must fit context with one answer token reserved"
            )
        if any(
            type(token) is not int or token < 0
            for token in question.suffix_ids + question.label_ids
        ):
            raise ValueError("question token ids must be nonnegative integers")
        k = len(question.label_ids)
        if not 2 <= k <= 26 or len(set(question.label_ids)) != k or len(question.option_spans) != k:
            raise ValueError("require 2–26 distinct native ids with aligned option spans")
        if (
            type(question.primitive) is not int
            or question.primitive not in (0, 1, 2)
            or (question.primitive == 0 and k != 2)
        ):
            raise ValueError("invalid typed candidate set")
        if any(type(x) is not int for x in question.display_order) or sorted(
            question.display_order
        ) != list(range(k)):
            raise ValueError("display order must be a candidate permutation")
        for span in question.option_spans:
            if (
                type(span) is not tuple
                or len(span) != 2
                or any(type(x) is not int for x in span)
                or not 0 <= span[0] < span[1] <= len(question.suffix_ids)
            ):
                raise ValueError("option spans must be nonempty and inside their own suffix")


def extract_evidence_features(
    text,
    inputs: EvidenceInputs,
    *,
    mode: str = "prefix_cache",
    cache_strategy: str = "deepcopy",
    lexical: bool = False,
    output_device="cpu",
    max_forward_tokens: int = 65536,
    max_feature_bytes: int = 256 * 1024 * 1024,
) -> EvidenceFeatures:
    """Extract one record with isolated suffixes and reusable state-only memory.

    ``prefix_cache`` runs the prefix once, then deep-copies its native cache for
    each suffix (including recurrent caches). Opt-in ``copy_on_write`` shares
    immutable KV tensors in stock, non-offloaded dynamic attention layers while
    copying their metadata; unsupported cache classes fail explicitly. It never
    crops discarded sliding states. ``full_rows`` independently runs
    prefix+suffix and serves as a reference. Both preserve final normalized
    states, actual output rows and uncalibrated native logits. Caps are validated
    before the first forward; feature bytes are not total model/KV/peak VRAM.
    Inference tensors are cloned into ordinary frozen tensors for head training.
    Unsupported cache behavior raises an error; it does not silently retry.
    """
    _validate_inputs(inputs)
    if mode not in {"prefix_cache", "full_rows"} or type(lexical) is not bool:
        raise ValueError("invalid feature extraction mode/lexical setting")
    if cache_strategy not in {"deepcopy", "copy_on_write"} or (
        mode == "full_rows" and cache_strategy != "deepcopy"
    ):
        raise ValueError("invalid evidence cache strategy for extraction mode")
    if any(type(cap) is not int or cap < 1 for cap in (max_forward_tokens, max_feature_bytes)):
        raise ValueError("resource caps must be positive integers")
    if text.training or any(module.training for module in text.modules()):
        raise ValueError("backbone must already be in eval mode")
    config = text.config
    hidden = config.hidden_size
    vocab = config.vocab_size
    if type(hidden) is not int or hidden < 1 or type(vocab) is not int or vocab < 1:
        raise ValueError("invalid backbone hidden/vocabulary dimensions")
    context = min(
        inputs.context_limit, getattr(config, "max_position_embeddings", inputs.context_limit)
    )
    if any(len(inputs.prefix_ids) + len(q.suffix_ids) >= context for q in inputs.questions):
        raise ValueError("inputs exceed backbone context; truncation is forbidden")
    all_ids = list(inputs.prefix_ids)
    for question in inputs.questions:
        all_ids.extend(question.suffix_ids + question.label_ids)
    if max(all_ids) >= vocab:
        raise ValueError("token ids exceed backbone vocabulary")
    qcount = len(inputs.questions)
    kmax = max(len(q.label_ids) for q in inputs.questions)
    prefix_length = len(inputs.prefix_ids)
    suffix_lengths = [len(q.suffix_ids) for q in inputs.questions]
    forward_tokens = sum(suffix_lengths) + prefix_length * (1 if mode == "prefix_cache" else qcount)
    if forward_tokens > max_forward_tokens:
        raise ValueError("planned forward tokens exceed cap")
    parameter = next(text.parameters())
    # Mixed-precision output rows/bias can promote native/lexical features.
    # Inspect metadata only; do not copy or scan the weights on their device.
    element_size = max(
        4,
        *(p.element_size() for p in text.parameters() if p.is_floating_point()),
        *(b.element_size() for b in text.buffers() if b.is_floating_point()),
    )
    # Conservatively include promoted span/lexical means and native scores.
    feature_upper = (
        (prefix_length * hidden + qcount * hidden + qcount * kmax * hidden * (1 + lexical))
        * element_size
        + qcount * kmax * (element_size + 1)
        + qcount * 9
        + prefix_length
    )
    if feature_upper > max_feature_bytes:
        raise ValueError("planned feature storage exceeds cap")
    destination = torch.device(output_device)
    source = parameter.device

    queries, candidates, priors, lexical_features = [], [], [], []
    memory = None
    cache = None
    prefix_kv_bytes = None
    calls = 0
    consumed_tokens = 0

    @contextmanager
    def accounted():
        try:
            yield
        except Exception as exc:
            raise FeatureExtractionError(
                exc, mode=mode, calls=calls, tokens=consumed_tokens, planned_tokens=forward_tokens
            ) from exc

    def forward(ids, past=None):
        nonlocal calls, consumed_tokens
        tensor = torch.tensor([ids], dtype=torch.long, device=source)
        calls += 1
        consumed_tokens += len(ids)
        result = text(
            input_ids=tensor,
            past_key_values=past,
            use_cache=mode == "prefix_cache",
            return_dict=True,
        )
        if (
            result.last_hidden_state.shape != (1, len(ids), hidden)
            or not torch.isfinite(result.last_hidden_state).all()
        ):
            raise ValueError("backbone did not return finite aligned final hidden states")
        return result

    # Copies outside inference_mode make features usable by Linear backward.
    def frozen(tensor):
        with torch.inference_mode(False):
            return tensor.detach().to(destination).clone()

    with accounted(), torch.inference_mode():
        if mode == "prefix_cache":
            pref = forward(inputs.prefix_ids)
            cache = pref.past_key_values
            if cache is None:
                raise ValueError("backbone has no native prefix cache")
            prefix_kv_bytes = dynamic_kv_bytes(cache)
            memory = frozen(pref.last_hidden_state)
            del pref
        for question in inputs.questions:
            if mode == "prefix_cache":
                result = forward(question.suffix_ids, branch_evidence_cache(cache, cache_strategy))
                suffix = result.last_hidden_state[0]
            else:
                result = forward(inputs.prefix_ids + question.suffix_ids)
                if memory is None:
                    memory = frozen(result.last_hidden_state[:, :prefix_length])
                suffix = result.last_hidden_state[0, prefix_length:]
            query = suffix[-1]
            ids = torch.tensor(question.label_ids, dtype=torch.long, device=source)
            prior = native_logits(text, query.expand(len(ids), -1), ids)
            if not torch.isfinite(prior).all():
                raise ValueError("native output-head logits are not finite")
            priors.append(frozen(prior))
            queries.append(frozen(query))
            candidate_features = torch.stack(
                [at_least_fp32(suffix[s:e]).mean(0) for s, e in question.option_spans]
            )
            if not torch.isfinite(candidate_features).all():
                raise ValueError("pooled candidate features are not finite")
            candidates.append(frozen(candidate_features))
            if lexical:
                option_ids = torch.tensor(question.suffix_ids, dtype=torch.long, device=source)
                lexical_feature = torch.stack(
                    [
                        at_least_fp32(output_rows(text, option_ids[s:e])).mean(0)
                        for s, e in question.option_spans
                    ]
                )
                if not torch.isfinite(lexical_feature).all():
                    raise ValueError("pooled lexical features are not finite")
                lexical_features.append(frozen(lexical_feature))
            del result, suffix, query

    with accounted(), torch.inference_mode(False):
        candidate_mask = torch.zeros(1, qcount, kmax, dtype=torch.bool, device=destination)
        padded_prior = torch.full(
            (1, qcount, kmax), -torch.inf, dtype=priors[0].dtype, device=destination
        )
        padded_candidates = torch.zeros(
            1, qcount, kmax, hidden, dtype=candidates[0].dtype, device=destination
        )
        padded_lexical = (
            torch.zeros_like(padded_candidates, dtype=lexical_features[0].dtype)
            if lexical
            else None
        )
        for qi, question in enumerate(inputs.questions):
            k = len(question.label_ids)
            candidate_mask[0, qi, :k] = True
            padded_prior[0, qi, :k] = priors[qi]
            padded_candidates[0, qi, :k] = candidates[qi]
            if lexical:
                padded_lexical[0, qi, :k] = lexical_features[qi]
        tensors = {
            "native_logits": padded_prior,
            "candidates": padded_candidates,
            "queries": torch.stack(queries)[None],
            "memory": memory,
            "memory_mask": torch.ones(1, prefix_length, dtype=torch.bool, device=destination),
            "candidate_mask": candidate_mask,
            "question_mask": torch.ones(1, qcount, dtype=torch.bool, device=destination),
            "primitive": torch.tensor(
                [[q.primitive for q in inputs.questions]], dtype=torch.long, device=destination
            ),
        }
        if lexical:
            tensors["lexical"] = padded_lexical
    stored_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    if stored_bytes > max_feature_bytes:
        raise FeatureExtractionError(
            "actual feature storage exceeds cap; no truncation applied",
            mode=mode,
            calls=calls,
            tokens=consumed_tokens,
            planned_tokens=forward_tokens,
        )
    metadata = {
        "artifact_kind": "unbound_evidence_features_v1",
        "promotable": False,
        "input_sha256": inputs.input_sha256,
        "state_sha256": inputs.state_sha256,
        "model_type": config.model_type,
        "feature_layer": "final_post_norm",
        "readout": "native_single_token",
        "label_ids": [list(q.label_ids) for q in inputs.questions],
        "mode": mode,
        "cache_type": type(cache).__name__ if cache is not None else None,
        "cache_strategy": cache_strategy if cache is not None else None,
        "logical_prefix_kv_bytes": prefix_kv_bytes,
        "cache_branch_kv_copy_bytes": (0 if cache_strategy == "copy_on_write" else prefix_kv_bytes),
        "prefix_tokens": prefix_length,
        "suffix_tokens": suffix_lengths,
        "forward_calls": calls,
        "forward_tokens": consumed_tokens,
        "planned_forward_tokens": forward_tokens,
        "stored_feature_bytes": stored_bytes,
        "planned_feature_bytes_upper_bound": feature_upper,
        "reasoning_tokens": 0,
    }
    return EvidenceFeatures(tensors, metadata)
