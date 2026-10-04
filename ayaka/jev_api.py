"""TypeSafe Jev wire contract, independent of SDKs and inference backends.

The Jev aliases provide SDK interoperability; they do not identify Ayaka as Jev.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class ValidationError(ValueError):
    def __init__(self, message: str, field: str = "questions"):
        super().__init__(message)
        self.field = field


def error_body(message: str, field: str | None = None) -> dict:
    return {"error": message, **({"field": field} if field else {})}


def render_content(value: object, field: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValidationError("must contain JSON values", field) from exc
    raise ValidationError("must be a string, object or array", field)


@dataclass(frozen=True)
class ParsedQuestion:
    type: str
    instruction: str
    labels: list[str]
    descriptions: list[str]
    criteria_values: list[object] | None = None

    @property
    def legend(self) -> dict[str, object]:
        return dict(
            zip(
                self.labels,
                self.criteria_values if self.criteria_values is not None else self.descriptions,
                strict=True,
            )
        )


def parse_question(
    question: object,
    *,
    field="questions",
    noul_defaults=("false", "true"),
    sort_score=False,
    enforce_limits=True,
) -> ParsedQuestion:
    """Accept Jev questions plus the historical Choice-list/Score-map aliases."""
    if not isinstance(question, dict):
        raise ValidationError("question must be an object", field)
    kind = question.get("type")
    instruction = question.get("instructions", question.get("instruction"))
    if (
        "instructions" in question
        and "instruction" in question
        and question["instructions"] != question["instruction"]
    ):
        raise ValidationError("conflicting instructions and instruction", f"{field}.instructions")
    instruction = (
        "" if instruction is None else render_content(instruction, f"{field}.instructions")
    )
    criteria = question.get("criteria")
    cf = f"{field}.criteria"

    def description(value, label):
        if not enforce_limits and value in (None, ""):
            return label
        if not enforce_limits and not isinstance(value, (str, dict, list)):
            return str(value)
        return render_content(value, f"{cf}.{label}")

    if kind == "noul":
        if criteria is None:
            criteria = {}
        if isinstance(criteria, dict) and set(criteria) <= {"no", "yes"}:
            criteria = {({"no": "false", "yes": "true"}[k]): v for k, v in criteria.items()}
        if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
            raise ValidationError("noul criteria must describe true and false", cf)
        labels = ["false", "true"]
        descriptions = [
            render_content(criteria.get(k, d), f"{cf}.{k}")
            for k, d in zip(labels, noul_defaults, strict=True)
        ]
    elif kind in ("choice", "score"):
        limit = (255 if kind == "choice" else 10) if enforce_limits else float("inf")
        if not isinstance(criteria, (dict, list)) or not 2 <= len(criteria) <= limit:
            raise ValidationError(f"{kind} criteria need 2..{limit} options/levels", cf)
        if isinstance(criteria, dict):
            if any(not isinstance(k, str) for k in criteria):
                raise ValidationError("option labels must be strings", cf)
            labels = list(criteria)
            if kind == "score" and sort_score:
                try:
                    labels.sort(key=int)
                except ValueError as exc:
                    raise ValidationError("score keys must be integer-like", cf) from exc
                if len({int(k) for k in labels}) != len(labels):
                    raise ValidationError("score ordinals must be unique", cf)
            descriptions = [
                k if kind == "choice" and criteria[k] in (None, "") else description(criteria[k], k)
                for k in labels
            ]
        elif kind == "choice":
            labels = [render_content(v, cf) for v in criteria]
            descriptions = labels.copy()
        else:
            labels = [str(i) for i in range(len(criteria))]
            descriptions = [description(v, str(i)) for i, v in enumerate(criteria)]
        if len(set(labels)) != len(labels):
            raise ValidationError("option labels must be unique", cf)
    else:
        raise ValidationError(f"unknown question type: {kind!r}", f"{field}.type")
    values = (
        ([criteria[k] for k in labels] if isinstance(criteria, dict) else criteria)
        if kind == "score"
        else None
    )
    return ParsedQuestion(
        kind, instruction, labels, descriptions, values if kind == "score" else None
    )


def normalize_request(body: object, *, max_questions=128) -> dict:
    """Copy preferred ayaka extensions into internal legacy aliases.

    Remove the namespace after resolving it so recursive candidate scoring cannot
    reintroduce a generation policy from the original question.
    """
    if not isinstance(body, dict):
        raise ValidationError("body must be an object", "body")
    result = copy.deepcopy(body)
    questions = result.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValidationError("questions must be a non-empty object")
    if max_questions is not None and len(questions) > max_questions:
        raise ValidationError(f"maximum {max_questions} questions per request")
    # Missing state remains the historical empty-state shorthand.
    render_content(result.get("state", ""), "state")
    extension = result.pop("ayaka", {})
    if not isinstance(extension, dict):
        raise ValidationError("ayaka must be an object", "ayaka")
    options = result.setdefault("options", {})
    if not isinstance(options, dict):
        raise ValidationError("options must be an object", "options")

    def merge(target, source, key, field):
        if key in source:
            if key in target and json.dumps(target[key], sort_keys=True) != json.dumps(
                source[key], sort_keys=True
            ):
                raise ValidationError(f"conflicting {key} values", field)
            target[key] = source[key]

    merge(options, extension, "reasoning", "ayaka.reasoning")
    merge(result, extension, "media", "ayaka.media")
    extra_questions = extension.get("questions", {})
    if not isinstance(extra_questions, dict) or set(extra_questions) - set(questions):
        raise ValidationError("ayaka.questions must name existing questions", "ayaka.questions")
    for name, question in questions.items():
        if not isinstance(name, str) or not isinstance(question, dict):
            raise ValidationError("question must be a named object", f"questions.{name}")
        extra = extra_questions.get(name, {})
        if not isinstance(extra, dict):
            raise ValidationError(
                "each ayaka.questions entry must be an object", f"ayaka.questions.{name}"
            )
        for key in ("reasoning", "candidate_generation"):
            merge(question, extra, key, f"ayaka.questions.{name}.{key}")
        policy = question.get("candidate_generation", {})
        if (
            isinstance(policy, dict)
            and policy.get("mode") in ("open", "expand")
            and question.get("type") != "choice"
        ):
            raise ValidationError(
                "candidate generation supports Choice only",
                f"questions.{name}.candidate_generation",
            )
        if isinstance(policy, dict) and policy.get("mode") == "open":
            # The generation validator handles the absent partition.
            render_content(
                question.get("instructions", question.get("instruction", "")),
                f"questions.{name}.instructions",
            )
        else:
            parse_question(question, field=f"questions.{name}")
    return result


def _values(probabilities: Mapping[str, float] | Sequence[float]) -> list[float]:
    values = list(probabilities.values() if isinstance(probabilities, Mapping) else probabilities)
    if len(values) < 2 or any(not math.isfinite(p) or p < 0 for p in values):
        raise ValueError("confidence requires at least two finite nonnegative probabilities")
    if not math.isclose(math.fsum(values), 1.0, abs_tol=1e-6):
        raise ValueError("probabilities must sum to one")
    return values


def choice_confidence(probabilities: Mapping[str, float] | Sequence[float]) -> float:
    values = _values(probabilities)
    baseline = 1 / len(values)
    return min(1.0, max(0.0, (max(values) - baseline) / (1 - baseline)))


def score_confidence(probabilities: Mapping[str, float] | Sequence[float]) -> float:
    values = _values(probabilities)
    n = len(values)
    mode = values.index(max(values))  # First maximum, including tied adjacent levels.
    spread = sum(p * abs(i - mode) for i, p in enumerate(values))
    uniform_mad = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / uniform_mad)


def build_answer(
    kind: str, probabilities: dict[str, float], *, legend=None, ordinals=None, true_label="true"
) -> dict:
    if kind == "noul":
        return {"type": kind, "noul": float(probabilities[true_label])}
    probs = {k: float(v) for k, v in probabilities.items()}
    if kind == "choice":
        return {
            "type": kind,
            "choice": max(probs, key=probs.__getitem__),
            "probabilities": probs,
            "confidence": choice_confidence(probs),
        }
    if kind != "score":
        raise ValueError(f"unknown question type: {kind}")
    indices = ordinals if ordinals is not None else [int(k) for k in probs]
    return {
        "type": kind,
        "score": float(sum(i * p for i, p in zip(indices, probs.values(), strict=True))),
        "legend": legend if legend is not None else {k: k for k in probs},
        "probabilities": probs,
        "confidence": score_confidence(probs),
    }


@dataclass(frozen=True)
class ModelCatalog:
    served_name: str
    model_id: str | None = None
    description: str = "Ayaka decision model; Jev aliases are accepted for SDK compatibility."
    release_date: str = "2026-10-04"

    @property
    def actual_id(self) -> str:
        return self.model_id or self.served_name

    def resolve(self, name=None) -> str:
        if name is None:
            name = "jev-latest"
        if not isinstance(name, str) or name not in self.names:
            raise ValidationError(
                f"unknown model: {name!r}; accepted: {', '.join(self.names)}", "model"
            )
        return self.actual_id

    @property
    def names(self) -> list[str]:
        return list(dict.fromkeys(["jev-latest", "jev-preview", self.served_name, self.actual_id]))

    def listing(self) -> dict:
        return {
            "models": [
                {"name": n, "description": self.description, "release_date": self.release_date}
                for n in self.names
            ]
        }


def complete_answers(
    response: dict, body: dict, *, calibration="unfitted", route="direct", include_diagnostics=True
) -> dict:
    """Refresh required fields after any final joint-mass or routing transformation."""
    for name, answer in response.get("answers", {}).items():
        if answer["type"] == "choice":
            answer["confidence"] = choice_confidence(answer["probabilities"])
        elif answer["type"] == "score":
            question = parse_question(body["questions"][name])
            answer["legend"] = {k: question.legend[k] for k in answer["probabilities"]}
            answer["confidence"] = score_confidence(answer["probabilities"])
        extra = answer.setdefault("ayaka", {})
        diagnostics = response.get("reasoning", {}).get(name, {})
        candidates = response.get("candidate_generation", {}).get(name)
        generated = "candidates" in extra or candidates is not None
        extra.setdefault("route", diagnostics.get("route", route))
        extra["calibration"] = (
            "unvalidated_image"
            if "media" in body
            else "unvalidated_generated_partition"
            if generated
            else calibration
        )
        if diagnostics and include_diagnostics:
            extra.setdefault("diagnostics", diagnostics)
        if candidates:
            extra.setdefault("candidates", candidates)
    response.setdefault("ayaka", {})
    return response


def wire_response(response: dict) -> dict:
    """v1 internal callers retain old diagnostics; the HTTP wire has one namespace."""
    result = copy.deepcopy(response)
    if "answers" not in result:
        return result
    extra = result.setdefault("ayaka", {})
    for name, answer in result["answers"].items():
        diagnostics = result.get("reasoning", {}).get(name)
        if diagnostics:
            answer.setdefault("ayaka", {}).setdefault("diagnostics", diagnostics)
    for key in ("reasoning", "candidate_generation", "latency_ms"):
        if key in result:
            extra[key] = result.pop(key)
    usage = result.get("usage", {})
    breakdown = {k: usage.pop(k) for k in list(usage) if k not in ("input_tokens", "output_tokens")}
    if breakdown:
        extra.setdefault("usage", {}).update(breakdown)
    return result
