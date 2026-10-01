import pytest

from ayaka.reasoning import ReasoningSettings, resolve_settings
from ayaka.serve import BadRequest, DecisionService


def test_precedence_is_per_field_and_immutable():
    s = resolve_settings(
        {"mode": "off", "effort": "low"},
        {"mode": "auto"},
        {"mode": "on", "effort": "high"},
        {"max_tokens": 64},
    )
    assert (s.mode, s.effort, s.budget) == ("on", "high", 64)
    assert resolve_settings().budget == 384
    assert ReasoningSettings(mode="off", effort="high").budget == 0
    assert ReasoningSettings(mode="on", max_tokens=0).budget == 0


@pytest.mark.parametrize(
    "values",
    [
        None,
        [],
        {"mode": "force"},
        {"effort": "max"},
        {"max_tokens": -1},
        {"max_tokens": 1025},
        {"max_tokens": True},
        {"max_tokens": 1.2},
        {"unknown": 1},
    ],
)
def test_invalid_settings(values):
    with pytest.raises(ValueError):
        ReasoningSettings().override(values)


def test_unsupported_backend_and_null_are_bad_requests():
    service = DecisionService(object(), "fake")
    body = {"questions": {"a": {"type": "noul"}}, "options": {"reasoning": {"mode": "on"}}}
    with pytest.raises(BadRequest, match="does not support"):
        service.handle(body)
    body["options"]["reasoning"] = None
    with pytest.raises(BadRequest, match="object"):
        service.handle(body)
