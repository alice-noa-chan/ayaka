import json

import pytest

from ayaka.swift.prompt import InvalidQuestion, parse_question, render_question
from ayaka.swift.readers import FakeReader


def test_noul_false_first_and_default_texts():
    messages, mapping = render_question(
        "State", {"type": "noul", "criteria": {"true": "T", "false": "F"}}
    )
    assert mapping == {"A": "false", "B": "true"}
    assert messages[1]["content"].endswith("A. F\nB. T")
    messages, _ = render_question("", {"type": "noul"})
    assert messages[1]["content"].endswith("A. no\nB. yes")
    assert "only the option letter" in messages[0]["content"]


def test_choice_request_order_and_descriptions():
    question = {"type": "choice", "criteria": {"z": None, "a": "", "x": {"한": 2}, "y": "desc"}}
    messages, mapping = render_question("literal\nstate", question)
    assert list(mapping.values()) == ["z", "a", "x", "y"]
    assert messages[1]["content"].startswith("literal\nstate\n")
    assert messages[1]["content"].endswith('A. z\nB. a\nC. x: {"한":2}\nD. desc')


@pytest.mark.parametrize("state", [{"한": [1, 2]}, ["한", {"x": 1}]])
def test_state_json_formats(state):
    question = {"type": "choice", "criteria": ["a", "b"]}
    messages, _ = render_question(state, question)
    assert messages[1]["content"].startswith(json.dumps(state, indent=1, ensure_ascii=False))
    messages, _ = render_question(state, question, state_format="compact")
    assert messages[1]["content"].startswith(
        json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    )


def test_score_list_and_numeric_key_order():
    _, mapping = render_question("", {"type": "score", "criteria": ["high", "low"]})
    assert mapping == {"A": "0", "B": "1"}
    messages, mapping = render_question(
        "", {"type": "score", "criteria": {"10": "ten", "-1": "minus", "2": "two"}}
    )
    assert mapping == {"A": "-1", "B": "2", "C": "10"}
    assert messages[1]["content"].endswith("A. minus\nB. two\nC. ten")


@pytest.mark.parametrize(
    "question",
    [
        None,
        {},
        {"type": "choice", "criteria": ["a"]},
        {"type": "choice", "criteria": ["a", "a"]},
        {"type": "score", "criteria": {"x": "a", "y": "b"}},
        {"type": "score", "criteria": {"1": "a", "01": "b"}},
        {"type": "noul", "instructions": ["bad"]},
    ],
)
def test_invalid_questions(question):
    with pytest.raises(InvalidQuestion):
        parse_question(question)


def test_single_prompt_option_limit():
    with pytest.raises(InvalidQuestion, match="26"):
        render_question("", {"type": "choice", "criteria": [str(i) for i in range(27)]})


@pytest.mark.parametrize("state", ["state\n \n", {"한": [1, 2]}, [1, "한"]])
@pytest.mark.parametrize("state_format", ["pretty", "compact"])
def test_min_is_byte_identical_to_original(state, state_format):
    instruction = "choose\n \n"
    text = (
        state
        if isinstance(state, str)
        else (
            json.dumps(state, indent=1, ensure_ascii=False)
            if state_format == "pretty"
            else json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        )
    )
    expected = [
        {"role": "system", "content": "Answer with only the option letter."},
        {"role": "user", "content": f"{text}\n\n{instruction}\nA. 아니요\nB. yes"},
    ]
    question = {
        "type": "choice",
        "instructions": instruction,
        "criteria": {"n": "아니요", "y": "yes"},
    }
    for options in ({}, {"prompt_variant": "min"}):
        messages, mapping = render_question(state, question, state_format=state_format, **options)
        assert mapping == {"A": "n", "B": "y"}
        assert (
            json.dumps(messages, ensure_ascii=False).encode()
            == json.dumps(expected, ensure_ascii=False).encode()
        )


@pytest.mark.parametrize("state", ["state\n ", {"한": [1, 2]}])
def test_cygnet_measured_layout_and_false_first(state):
    messages, mapping = render_question(
        state,
        {"type": "noul", "instructions": "question\n ", "criteria": {"true": "T", "false": "F"}},
        prompt_variant="cygnet",
    )
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
    assert messages == [
        {
            "role": "system",
            "content": (
                "You are a calibration engine. You never answer in prose. You are given a state, a question and "
                "a numbered set of options, and you choose exactly one option. You reply with that option's "
                "LETTER and nothing else — a single character, no words, no punctuation, no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{text.rstrip()}\n\nquestion\n\nOptions:\nA. F\nB. T\n\n"
                "Answer with the letter of exactly one option, and nothing else:"
            ),
        },
    ]
    assert mapping == {"A": "false", "B": "true"}


def test_rules_budget_and_fake_reader_rendering():
    question = {"type": "choice", "criteria": ["accept", "reject"]}
    minimum, _ = render_question("state", question)
    rules, mapping = render_question("state", question, prompt_variant="rules")

    def whitespace_tokens(messages):
        return sum(len(message["content"].split()) for message in messages)

    assert 0 < whitespace_tokens(rules) - whitespace_tokens(minimum) <= 80
    assert rules[1] == minimum[1]
    for term in (
        "definitions",
        "exceptions",
        "amendments",
        "effective dates",
        "arithmetic",
        "governing text",
    ):
        assert term in rules[0]["content"]
    assert "JevBench" not in rules[0]["content"]
    reader = FakeReader()
    reader.read(rules, list(mapping))
    assert reader.calls == [(rules, ["A", "B"])]
    with pytest.raises(ValueError, match="prompt_variant"):
        render_question("state", question, prompt_variant="unknown")


@pytest.mark.parametrize(
    "question,options",
    [
        ({"type": "choice", "criteria": {"x": "description", "y": "y"}}, "A. x: description\nB. y"),
        ({"type": "choice", "criteria": {"x": {"한": 2}, "y": None}}, 'A. x: {"한":2}\nB. y'),
        ({"type": "noul"}, "A. false: no\nB. true: yes"),
        ({"type": "score", "criteria": {"8": "high", "2": "low"}}, "A. low\nB. high"),
    ],
)
def test_labeled_layout_preserves_mapping_and_min_system(question, options):
    messages, mapping = render_question("state", question, prompt_variant="labeled")
    minimum, original_mapping = render_question("state", question)
    assert mapping == original_mapping
    assert messages[0] == minimum[0]
    assert messages[1]["content"] == f"state\n\n\n{options}"
