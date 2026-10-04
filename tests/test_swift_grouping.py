import pytest

from ayaka.swift.grouping import read_question
from ayaka.swift.policy import Policy
from ayaka.swift.prompt import InvalidQuestion, parse_question
from ayaka.swift.readers import FakeReader


def test_grouping_joint_probabilities_and_usage():
    question = parse_question(
        {"type": "choice", "criteria": {str(i): f"option {i}" for i in range(30)}}
    )
    a = {chr(65 + i): (0.3 if i == 0 else 0.7 / 14) for i in range(15)}
    b = {chr(65 + i): (0.6 if i == 14 else 0.4 / 14) for i in range(15)}
    reader = FakeReader([a, b, {"A": 0.8, "B": 0.2}])
    result = read_question(reader, "state", question)
    assert [len(letters) for _, letters in reader.calls] == [15, 15, 2]
    assert result.raw_probs["0"] == pytest.approx(0.8 * 0.3)
    assert result.raw_probs["29"] == pytest.approx(0.2 * 0.6)
    assert result.raw_probs["1"] == pytest.approx(0.8 * 0.7 / 14)
    assert result.raw_probs["15"] == pytest.approx(0.2 * 0.4 / 14)
    assert sum(result.raw_probs.values()) == pytest.approx(1)
    assert (result.input_tokens, result.output_tokens, result.latency_s) == (30, 3, 0.03)
    assert reader.calls[-1][0][1]["content"].endswith("A. option 0\nB. option 29")
    expected = {label: p**0.5 for label, p in result.raw_probs.items()}
    total = sum(expected.values())
    assert Policy(t_choice=2).apply("choice", result.raw_probs) == pytest.approx(
        {label: p / total for label, p in expected.items()}
    )


def test_near_equal_configured_groups():
    question = parse_question({"type": "score", "criteria": [str(i) for i in range(31)]})
    reader = FakeReader()
    result = read_question(reader, "", question, group_size=8)
    assert [len(letters) for _, letters in reader.calls] == [8, 8, 8, 7, 4]
    assert result.raw_probs["0"] == pytest.approx(1 / 4 / 8)
    assert result.raw_probs["30"] == pytest.approx(1 / 4 / 7)


def test_26_direct_and_group_final_limit():
    reader = FakeReader()
    read_question(
        reader, "", parse_question({"type": "choice", "criteria": [str(i) for i in range(26)]})
    )
    assert len(reader.calls) == 1
    with pytest.raises(InvalidQuestion, match="too many"):
        read_question(
            reader, "", parse_question({"type": "choice", "criteria": [str(i) for i in range(521)]})
        )
    assert len(reader.calls) == 1


@pytest.mark.parametrize("variant", ["min", "cygnet", "rules"])
def test_every_group_and_final_pass_uses_variant(variant):
    from ayaka.swift.prompt import SYSTEM_TEXTS

    reader = FakeReader()
    question = parse_question({"type": "choice", "criteria": [str(i) for i in range(30)]})
    read_question(reader, "state", question, prompt_variant=variant)
    assert len(reader.calls) == 3
    assert all(messages[0]["content"] == SYSTEM_TEXTS[variant] for messages, _ in reader.calls)
