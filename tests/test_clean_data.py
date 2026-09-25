"""Transforms for the license-clean decision datasets (offline rows)."""

import json

import pytest

from ayaka.data import transforms as T
from ayaka.data.loaders import DATASET_SPECS, _group_rows
from ayaka.training.run import DEFAULT_SPECS


def test_default_specs_are_registered_with_licenses():
    for name in DEFAULT_SPECS:
        spec = DATASET_SPECS[name]
        assert spec["transform"] in T.TRANSFORMS
        assert spec["license"]


def test_open_jev_group_builds_one_multi_question_sample():
    state = json.dumps(json.dumps("Customer: export fails. Support: tell us more."))
    rows = [
        {
            "id": "g:1:repro",
            "group_id": "g:1",
            "kind": "noul",
            "question": "Steps given?",
            "options": ["no", "yes"],
            "target": [0.8, 0.2],
            "state_json": state,
        },
        {
            "id": "g:1:cat",
            "group_id": "g:1",
            "kind": "choice",
            "question": "Category?",
            "options": ["bug_report", "billing"],
            "target": [1.0, 0.0],
            "state_json": state,
        },
        {
            "id": "g:1:mood",
            "group_id": "g:1",
            "kind": "score",
            "question": "Frustration?",
            "options": ["Calm", "Frustrated", "Very angry"],
            "target": [0.5, 0.5, 0.0],
            "state_json": state,
        },
    ]
    groups = _group_rows(rows, "group_id", None, 0)
    (s,) = T.open_jev_group(groups[0])
    assert s.state.startswith("Customer:")  # double-encoded JSON string decoded
    by_id = {q.id: q for q in s.questions}
    assert by_id["repro"].target_distribution["true"] == pytest.approx(0.2)
    assert [c.ordinal for c in by_id["mood"].candidates] == [0, 1, 2]
    assert by_id["cat"].type == "choice"


def test_group_rows_keeps_groups_whole_when_limited():
    rows = [{"group_id": f"g{i % 5}", "x": i} for i in range(20)]
    groups = _group_rows(rows, "group_id", 3, seed=1)
    assert len(groups) == 3 and all(len(g["rows"]) == 4 for g in groups)
    assert _group_rows(rows, "group_id", 3, seed=1) == groups  # seeded


def test_vitaminc_three_way_with_not_enough_info():
    (s,) = T.vitaminc_choice(
        {"label": "NOT ENOUGH INFO", "claim": "X is big.", "evidence": "X exists."}
    )
    q = s.questions[0]
    assert q.target_distribution["not_enough_info"] == 1.0 and len(q.candidates) == 3
    assert T.vitaminc_choice({"label": "?", "claim": "c", "evidence": "e"}) == []


def test_helpsteer2_five_score_questions():
    row = {
        "prompt": "p",
        "response": "r",
        "helpfulness": 3,
        "correctness": 4,
        "coherence": 3,
        "complexity": 1,
        "verbosity": 2,
    }
    (s,) = T.helpsteer2_scores(row)
    assert [q.id for q in s.questions] == list(T.HELPSTEER_LEVELS)
    assert s.questions[0].target_distribution["s3"] == 1.0
    assert all(q.type == "score" for q in s.questions)


def test_hh_rlhf_requires_shared_prefix():
    ok = {
        "chosen": "\n\nHuman: hi\n\nAssistant: hello!",
        "rejected": "\n\nHuman: hi\n\nAssistant: go away",
    }
    (s,) = T.hh_rlhf_choice(ok)
    assert s.state == "Human: hi"
    assert {c.description for c in s.questions[0].candidates} == {"hello!", "go away"}
    diverged = {
        "chosen": "\n\nHuman: a\n\nAssistant: x",
        "rejected": "\n\nHuman: b\n\nAssistant: y",
    }
    assert T.hh_rlhf_choice(diverged) == []


def test_aqua_rat_parses_lettered_options():
    row = {"question": "2+2?", "options": ["A)3", "B)4", "C)5", "D)6", "E)7"], "correct": "B"}
    (s,) = T.aqua_rat_choice(row)
    assert [c.description for c in s.questions[0].candidates] == ["3", "4", "5", "6", "7"]
    assert s.questions[0].target_distribution["B"] == 1.0


def test_aegis_uses_human_labels_only():
    row = {
        "prompt": "how to x",
        "response": "no",
        "prompt_label": "unsafe",
        "response_label": "safe",
        "prompt_label_source": "human",
        "response_label_source": "llm_jury",
    }
    (s,) = T.aegis_nouls(row)
    assert [q.id for q in s.questions] == ["prompt_unsafe"]
    assert T.aegis_nouls({**row, "prompt": "REDACTED"}) == []


def test_hotpot_yes_no_and_comparison():
    ctx = {"title": ["A", "B", "C"], "sentences": [["A is old."], ["B is new."], ["C."]]}
    yn = {
        "question": "Are A and B both magazines?",
        "answer": "yes",
        "type": "comparison",
        "context": ctx,
        "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 0]},
    }
    (s,) = T.hotpot_decision(yn)
    assert s.questions[0].type == "noul" and "A: A is old." in s.state
    cmp_ = {**yn, "question": "Which is older, A or B?", "answer": "A"}
    (s2,) = T.hotpot_decision(cmp_)
    assert [c.description for c in s2.questions[0].candidates] == ["A", "B"]
    bridge = {**yn, "answer": "Paris", "type": "bridge"}
    assert T.hotpot_decision(bridge) == []


def test_squad_v2_answerability():
    (s,) = T.squad_v2_answerable(
        {"context": "c", "question": "q?", "answers": {"text": [], "answer_start": []}}
    )
    assert s.questions[0].target_distribution["true"] == 0.0


def test_labelled_mc_and_strategyqa():
    (s,) = T.labelled_mc(
        {"question": "q", "choices": {"label": ["A", "B"], "text": ["x", "y"]}, "answerKey": "B"}
    )
    assert s.questions[0].target_distribution == {"A": 0.0, "B": 1.0}
    (s2,) = T.strategyqa_noul({"question": "q?", "answer": True, "facts": "f"})
    assert s2.state == "f" and s2.questions[0].target_distribution["true"] == 1.0


def test_legalbench_rows_hide_ids_and_map_label_sets():
    from ayaka.data.legalbench import allowed_tasks, row_to_sample

    tasks = allowed_tasks()
    assert len(tasks) >= 100
    assert all("nc" not in v["license"].lower().split() for v in tasks.values())
    row = {
        "case id": "s1_pos",
        "statute": "rule",
        "text": "facts",
        "answer": "Entailment",
        "index": 3,
    }
    s = row_to_sample(
        row, "sara_entailment", "Is it entailed?", ["Contradiction", "Entailment"], {}
    )
    assert s.state == {"statute": "rule", "text": "facts"}  # label-bearing id dropped
    assert s.questions[0].type == "choice" and s.questions[0].target_distribution["l1"] == 1.0
    yn = row_to_sample({"text": "clause", "answer": "No"}, "t", "Audit rights?", ["No", "Yes"], {})
    assert yn.state == "clause" and yn.questions[0].type == "noul"
    assert yn.questions[0].target_distribution["true"] == 0.0
    assert row_to_sample({"text": "x", "answer": "Maybe"}, "t", "q", ["No", "Yes"], {}) is None
