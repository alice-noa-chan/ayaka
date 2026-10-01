"""Immutable, field-wise reasoning settings shared by HTTP and Python callers."""

from dataclasses import asdict, dataclass

EFFORT_TOKENS = {"low": 128, "medium": 384, "high": 1024}


@dataclass(frozen=True)
class ReasoningSettings:
    mode: str = "auto"
    effort: str = "medium"
    max_tokens: int | None = None

    def __post_init__(self):
        if self.mode not in ("off", "auto", "on"):
            raise ValueError("reasoning.mode must be off, auto, or on")
        if self.effort not in EFFORT_TOKENS:
            raise ValueError("reasoning.effort must be low, medium, or high")
        if self.max_tokens is not None and (
            type(self.max_tokens) is not int or not 0 <= self.max_tokens <= 1024
        ):
            raise ValueError("reasoning.max_tokens must be an integer in 0..1024")

    @property
    def budget(self):
        return (
            0
            if self.mode == "off"
            else (self.max_tokens if self.max_tokens is not None else EFFORT_TOKENS[self.effort])
        )

    def override(self, values):
        if not isinstance(values, dict):
            raise ValueError("reasoning settings must be an object")
        unknown = values.keys() - asdict(self).keys()
        if unknown:
            raise ValueError(f"unknown reasoning fields: {sorted(unknown)}")
        return ReasoningSettings(**(asdict(self) | values))

    def as_dict(self):
        return asdict(self) | {"budget": self.budget}


def resolve_settings(checkpoint=None, server=None, request=None, question=None):
    settings = ReasoningSettings()
    for values in (checkpoint, server, request, question):
        if values is not None:
            settings = settings.override(values)
    return settings
