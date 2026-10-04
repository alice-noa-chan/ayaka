"""Candidate-order training augmentation and predeclared dev diagnostics.

Actual prompts are rerendered, never permuted frozen causal feature tensors.
Gold and ordinal meaning follow semantic labels. No dev/test value chooses a
training permutation; dev uses an explicit small fixed plan with no ensembling.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

import torch

from ..eval.read_artifact import fingerprint, resolve_target
from ..eval.v2 import typed_row
from ..primitives import QuestionSpec
from ..swift.prompt import Question, parse_question
from .swift_evidence import (
    SwiftEvidenceInputs,
    extract_swift_evidence_features,
    prepare_swift_evidence_inputs,
    validate_swift_evidence_inputs,
)


@dataclass(frozen=True)
class PermutedEvidenceInputs:
    prepared: SwiftEvidenceInputs
    canonical_labels: tuple[tuple[str, ...], ...]
    canonical_question_sha256: tuple[str, ...]
    display_to_canonical: tuple[tuple[int, ...], ...]
    targets: tuple[tuple[float, ...], ...]
    ordinals: tuple[tuple[int, ...] | None, ...]
    split: str
    record_id: str
    seed: int
    epoch: int
    binding_sha256: str

    def binding(self):
        return {
            "recipe_sha256": self.prepared.recipe_sha256,
            "canonical_labels": self.canonical_labels,
            "canonical_question_sha256": self.canonical_question_sha256,
            "display_to_canonical": self.display_to_canonical,
            "targets": self.targets,
            "ordinals": self.ordinals,
            "split": self.split,
            "record_id": self.record_id,
            "seed": self.seed,
            "epoch": self.epoch,
        }

    def supervision(self, features):
        """Gold tensors for the exact augmented prompt, including soft targets."""
        self.validate()
        if features.metadata.get("prior_recipe_sha256") != self.prepared.recipe_sha256:
            raise ValueError("permutation supervision requires features from its exact prompt")
        mask = features.tensors["candidate_mask"]
        counts = [len(row) for row in self.targets]
        if mask.shape != (1, len(counts), max(counts)) or mask.sum(-1).tolist() != [counts]:
            raise ValueError("permutation supervision candidate shape mismatch")
        target = torch.zeros(mask.shape, dtype=torch.float64, device=mask.device)
        levels = torch.zeros_like(target)
        for qi, row in enumerate(self.targets):
            target[0, qi, : len(row)] = torch.tensor(row, dtype=target.dtype, device=mask.device)
            if self.ordinals[qi] is not None:
                levels[0, qi, : len(row)] = torch.tensor(
                    self.ordinals[qi], dtype=levels.dtype, device=mask.device
                )
        return {"targets": target, "ordinals": levels}

    def validate(self):
        validate_swift_evidence_inputs(self.prepared)
        if fingerprint(self.binding()) != self.binding_sha256:
            raise ValueError("permutation gold/prompt binding changed")


def _parsed(questions):
    parsed = [q if isinstance(q, Question) else parse_question(q) for q in questions]
    for q in parsed:
        if q.type == "noul" and q.labels != ["false", "true"]:
            raise ValueError("canonical Noul labels must be false/true before augmentation")
        if q.type == "score":
            levels = [int(label) for label in q.labels]
            if levels != sorted(set(levels)):
                raise ValueError("canonical Score levels must be distinct and sorted")
    return parsed


def _orders(questions, orders):
    if len(orders) != len(questions):
        raise ValueError("require one complete display permutation per question")
    result = []
    for q, order in zip(questions, orders, strict=True):
        if any(type(i) is not int for i in order) or sorted(order) != list(range(len(q.labels))):
            raise ValueError("display order must be a complete candidate permutation")
        result.append(tuple(order))
    return tuple(result)


def prepare_permuted_evidence_inputs(
    state,
    questions,
    targets,
    tokenizer,
    *,
    split,
    record_id,
    seed=17,
    epoch=0,
    display_orders=None,
    **prompt_options,
):
    """One deterministic view per train epoch; reserved dev views must be explicit.

    Seed/order depends on record identity, epoch and question, never gold. Score
    and Noul permutations deliberately differ from canonical serving order and
    must be mapped back before reporting metrics. Test augmentation is rejected.
    """
    if split not in ("train", "dev"):
        raise ValueError("permutation augmentation supports train or explicit dev diagnostics only")
    if (
        not isinstance(record_id, str)
        or not record_id.strip()
        or type(seed) is not int
        or type(epoch) is not int
        or epoch < 0
    ):
        raise ValueError("require record identity, integer seed and nonnegative epoch")
    qs = _parsed(questions)
    if not qs or len(targets) != len(qs):
        raise ValueError("require aligned questions and gold distributions")
    gold = [resolve_target(q.labels, target) for q, target in zip(qs, targets, strict=True)]
    if display_orders is None:
        if split != "train":
            raise ValueError("dev diagnostics require predeclared display orders")
        display_orders = []
        for qi, q in enumerate(qs):
            digest = fingerprint(
                {"record": record_id, "seed": seed, "epoch": epoch, "question": qi}
            )
            order = list(range(len(q.labels)))
            random.Random(int(digest, 16)).shuffle(order)
            display_orders.append(order)
    orders = _orders(qs, display_orders)
    augmented = [
        Question(
            q.type, q.instruction, [q.labels[i] for i in order], [q.descriptions[i] for i in order]
        )
        for q, order in zip(qs, orders, strict=True)
    ]
    prepared = prepare_swift_evidence_inputs(state, augmented, tokenizer, **prompt_options)
    result = PermutedEvidenceInputs(
        prepared,
        tuple(tuple(q.labels) for q in qs),
        tuple(fingerprint(vars(q)) for q in qs),
        orders,
        tuple(
            tuple(g[q.labels[i]] for i in order)
            for q, g, order in zip(qs, gold, orders, strict=True)
        ),
        tuple(
            tuple(int(q.labels[i]) for i in order) if q.type == "score" else None
            for q, order in zip(qs, orders, strict=True)
        ),
        split,
        record_id,
        seed,
        epoch,
        "",
    )
    return replace(result, binding_sha256=fingerprint(result.binding()))


def extract_permuted_evidence_features(text, prepared, **options):
    prepared.validate()
    return extract_swift_evidence_features(text, prepared.prepared, **options)


def dev_permutation_plan(questions, *, max_views=3):
    """Canonical, reverse, then cyclic order; deduplicate complete record views."""
    if type(max_views) is not int or not 1 <= max_views <= 3:
        raise ValueError("dev permutation diagnostics allow 1..3 fixed views")
    qs = _parsed(questions)
    if not qs:
        raise ValueError("require dev questions")
    canonical = tuple(tuple(range(len(q.labels))) for q in qs)
    plans = [
        canonical,
        tuple(tuple(reversed(row)) for row in canonical),
        tuple(row[1:] + row[:1] for row in canonical),
    ]
    return list(dict.fromkeys(plans))[:max_views]


def permutation_diagnostics(prepared_views, probabilities, *, planned_max_views=3):
    """Report variation after semantic realignment; do not select a best view.

    This validates the declared input/gold bindings, not backend execution.
    Caller must provide every predeclared view and bind measured observations.
    """
    if not prepared_views or len(prepared_views) != len(probabilities):
        raise ValueError("require probabilities for every predeclared dev view")
    first = prepared_views[0]
    for view in prepared_views:
        view.validate()
        if (
            view.split != "dev"
            or view.canonical_labels != first.canonical_labels
            or view.canonical_question_sha256 != first.canonical_question_sha256
            or view.record_id != first.record_id
            or view.prepared.inputs.state_sha256 != first.prepared.inputs.state_sha256
        ):
            raise ValueError("permutation diagnostics require matched dev evidence and labels")
    identities = [view.display_to_canonical for view in prepared_views]
    if len(set(identities)) != len(identities):
        raise ValueError("dev views must not duplicate a display order")
    if identities != dev_permutation_plan(
        [
            Question(record["type"], "", list(labels), list(labels))
            for record, labels in zip(
                first.prepared.recipe["questions"], first.canonical_labels, strict=True
            )
        ],
        max_views=planned_max_views,
    ):
        raise ValueError("dev observations do not cover the complete fixed permutation plan")
    recipes = [view.prepared.recipe for view in prepared_views]
    for recipe in recipes[1:]:
        for key in (
            "readout",
            "prompt_variant",
            "state_format",
            "chat_template_kwargs",
            "chat_template_sha256",
            "tokenizer_sha256",
        ):
            if recipe[key] != recipes[0][key]:
                raise ValueError("permutation dev views use different rendering recipes")
    rows = []
    for qi, labels in enumerate(first.canonical_labels):
        aligned, gold, metrics = [], None, []
        kind = recipes[0]["questions"][qi]["type"]
        for view, observations in zip(prepared_views, probabilities, strict=True):
            if len(observations) != len(first.canonical_labels):
                raise ValueError("dev probabilities must cover every question")
            values = observations[qi]
            if (
                len(values) != len(labels)
                or any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in values)
                or not math.isclose(math.fsum(values), 1, abs_tol=1e-6, rel_tol=0)
            ):
                raise ValueError(
                    "permutation probabilities must be finite normalized distributions"
                )
            order = view.display_to_canonical[qi]
            p = [values[order.index(i)] for i in range(len(labels))]
            target = [view.targets[qi][order.index(i)] for i in range(len(labels))]
            if gold is not None and target != gold:
                raise ValueError("permutation dev views changed semantic gold")
            gold = target
            spec = QuestionSpec(
                kind,
                "",
                list(labels),
                [int(label) for label in labels] if kind == "score" else None,
            )
            metrics.append(typed_row(spec, p, target))
            aligned.append(p)
        tops = [max(range(len(labels)), key=p.__getitem__) for p in aligned]
        metric_names = (
            ("nll", "brier", "rps", "nmae") if kind == "score" else ("nll", "brier", "correct")
        )
        rows.append(
            {
                "question_index": qi,
                "type": kind,
                "labels": list(labels),
                "canonical_probs": aligned[0],
                "max_label_probability_range": max(
                    max(p[i] for p in aligned) - min(p[i] for p in aligned)
                    for i in range(len(labels))
                ),
                "max_pairwise_total_variation": max(
                    0.5 * math.fsum(abs(a - b) for a, b in zip(p, q, strict=True))
                    for p in aligned
                    for q in aligned
                ),
                "argmax_flips_from_canonical": sum(top != tops[0] for top in tops[1:]),
                "metrics_range": {
                    name: [min(m[name] for m in metrics), max(m[name] for m in metrics)]
                    for name in metric_names
                },
            }
        )
    return {
        "version": "ayaka-evidence-permutation-dev-1",
        "split": "dev",
        "recipe_sha256": [v.prepared.recipe_sha256 for v in prepared_views],
        "binding_sha256": [v.binding_sha256 for v in prepared_views],
        "views": len(prepared_views),
        "planned_max_views": planned_max_views,
        "rows": rows,
        "selected_view": None,
        "probability_ensemble": False,
        "execution_attested": False,
        "promotable": False,
    }
