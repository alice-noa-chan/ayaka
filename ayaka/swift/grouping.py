"""Hierarchical readout for option sets larger than the letter alphabet."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .policy import normalize
from .prompt import InvalidQuestion, Question, render_options
from .readers import READOUT, LetterReader


@dataclass(frozen=True)
class QuestionRead:
    raw_probs: dict[str, float]
    input_tokens: int
    output_tokens: int
    latency_s: float
    readout: str = READOUT
    passes: int = 1
    candidate_log_masses: dict[str, float] | None = None
    pass_inputs: list[dict] = field(default_factory=list)


def option_groups(count: int, group_size: int) -> list[list[int]]:
    """Deterministic initial partitions, shared with pre-inference bindings."""
    validate_option_count(count, group_size)
    if count <= 26:
        return [list(range(count))]
    size, extra = divmod(count, math.ceil(count / group_size))
    result, offset = [], 0
    for i in range(math.ceil(count / group_size)):
        end = offset + size + (i < extra)
        result.append(list(range(offset, end)))
        offset = end
    return result


def read_question(
    reader: LetterReader,
    state: object,
    question: Question,
    *,
    group_size: int = 20,
    state_format: str = "pretty",
    prompt_variant: str = "min",
) -> QuestionRead:
    """Read near-equal groups, then weight each by its winner's final mass."""
    validate_option_count(len(question.labels), group_size)
    input_tokens = output_tokens = 0
    latency_s = 0.0
    passes = 0
    masses = None
    pass_inputs = []

    def read_indices(indices: list[int]) -> dict[str, float]:
        nonlocal input_tokens, output_tokens, latency_s, passes, masses
        messages, mapping = render_options(
            state,
            question.instruction,
            [question.labels[i] for i in indices],
            [question.descriptions[i] for i in indices],
            state_format=state_format,
            prompt_variant=prompt_variant,
        )
        result = reader.read(messages, list(mapping))
        pass_inputs.append(
            {
                "messages": messages,
                "labels": list(mapping.values()),
                "letters": list(mapping),
                "input_token_ids": result.input_token_ids,
                "canonical_token_ids": result.canonical_token_ids,
                "token_logits": {str(k): v for k, v in result.token_logits.items()},
            }
        )
        probs = normalize({letter: result.letter_probs[letter] for letter in mapping})
        input_tokens += result.input_tokens
        output_tokens += result.output_tokens
        latency_s += result.latency_s
        passes += 1
        masses = (
            {mapping[letter]: result.letter_log_masses[letter] for letter in mapping}
            if result.letter_log_masses is not None
            else None
        )
        return {mapping[letter]: p for letter, p in probs.items()}

    n = len(question.labels)
    if n <= 26:
        probs = read_indices(list(range(n)))
    else:
        groups, winners = [], []
        for indices in option_groups(n, group_size):
            conditional = read_indices(indices)
            groups.append(conditional)
            winner = max(conditional, key=conditional.__getitem__)
            winners.append(question.labels.index(winner))
        final = read_indices(winners)
        probs = {
            label: final[question.labels[winner]] * p
            for group, winner in zip(groups, winners, strict=True)
            for label, p in group.items()
        }
    return QuestionRead(
        probs,
        input_tokens,
        output_tokens,
        latency_s,
        "grouped_approx" if n > 26 else getattr(reader, "readout", "undeclared"),
        passes,
        masses if n <= 26 else None,
        pass_inputs,
    )


def validate_option_count(count: int, group_size: int) -> None:
    if not 2 <= group_size <= 26:
        raise ValueError("group_size must be in 2..26")
    if count > max(26, 26 * group_size):
        raise InvalidQuestion(f"too many options: maximum is {26 * group_size}")
