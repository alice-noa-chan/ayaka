"""Optional helpers around the official typesafe-sdk; no SDK fork.

Use ``system_one(client, ..., ayaka={...})`` or pass ``AyakaResponse`` as the
official client's response_model. Install ayaka[sdk-compat] to use these helpers.
"""

from __future__ import annotations

from typing import Annotated, Any

try:
    from pydantic import ConfigDict, Field, model_validator
    from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse
except ImportError as exc:
    raise ImportError("ayaka.client requires typesafe-sdk; install ayaka[sdk-compat]") from exc


class AyakaNoulAnswer(NoulAnswer):
    ayaka: dict[str, Any] = Field(default_factory=dict)


class AyakaChoiceAnswer(ChoiceAnswer):
    ayaka: dict[str, Any] = Field(default_factory=dict)


class AyakaScoreAnswer(ScoreAnswer):
    ayaka: dict[str, Any] = Field(default_factory=dict)


AyakaAnswer = Annotated[
    AyakaNoulAnswer | AyakaChoiceAnswer | AyakaScoreAnswer, Field(discriminator="type")
]


class AyakaResponse(SystemOneResponse):
    # The SDK lifts newly declared response fields from same-named questions.
    # Keeping the namespace as an extra prevents a question named "ayaka" from
    # replacing top-level metadata, while reusing the official decoder unchanged.
    model_config = ConfigDict(extra="allow")
    answers: dict[str, AyakaAnswer] = Field(default_factory=dict)

    @property
    def ayaka(self) -> dict[str, Any]:
        return (self.model_extra or {}).get("ayaka", {})

    @model_validator(mode="after")
    def validate_namespace(self):
        if not isinstance(self.ayaka, dict):
            raise ValueError("ayaka must be an object")
        return self


def extra_body(ayaka: dict, *, extra: dict | None = None) -> dict:
    """Build official-SDK extra_body, rejecting conflicting duplicate values."""
    if not isinstance(ayaka, dict):
        raise ValueError("ayaka must be an object")
    body = dict(extra or {})
    if "ayaka" in body and body["ayaka"] != ayaka:
        raise ValueError("conflicting ayaka extension values")
    body["ayaka"] = ayaka
    return body


def system_one(client, state, questions, *, ayaka=None, response_model=AyakaResponse, **kwargs):
    """Call an existing official SDK client, preserving namespaced extensions."""
    supplied = kwargs.pop("extra_body", None)
    body = extra_body(
        ayaka if ayaka is not None else (supplied or {}).get("ayaka", {}), extra=supplied
    )
    return client.system_one(
        state, questions, extra_body=body, response_model=response_model, **kwargs
    )
