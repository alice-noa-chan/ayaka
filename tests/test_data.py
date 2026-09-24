import random

import pytest

from ayaka.data.augment import (
    add_distractor,
    add_nota,
    delete_evidence_text,
    evidence_deletion_variant,
    group_shared_state,
    permute_candidates,
    remove_candidate,
)
from ayaka.data.dedup import dedup_samples, group_split, sample_dedup_key
from ayaka.data.mixture import MixtureSampler, language_weights
from ayaka.data.schema import Candidate, Question, Sample, uniform
from ayaka.data.transforms import (
    boolq_noul,
    chaos_nli,
    intent_choice,
    jev_direct,
    multilabel_nouls,
    multirc_nouls,
    nli_choice,
    ordinal_score,
    sts_score,
)


def _choice(qid="q", k=3):
    cands = [Candidate(f"c{i}", f"desc{i}") for i in range(k)]
    return Question(
        qid, "choice", "pick?", cands, {"c0": 1.0, **{f"c{i}": 0.0 for i in range(1, k)}}
    )


def _sample(state="s", questions=None):
    return Sample(state=state, questions=questions or [_choice()], metadata={"language": "en"})


# ------------------------------------------------------------------ schema


def test_question_validation():
    with pytest.raises(ValueError, match="unknown primitive"):
        Question("q", "bogus", "i", [], {})
    with pytest.raises(ValueError, match="missing candidates"):
        _choice().__class__("q", "choice", "i", [Candidate("a", "a")], {"b": 1.0})
    with pytest.raises(ValueError, match="sums to"):
        Question("q", "choice", "i", [Candidate("a", "a")], {"a": 0.5})
    with pytest.raises(ValueError, match="ordinal"):
        Question("q", "score", "i", [Candidate("a", "a")], {"a": 1.0})


def test_noul_factory():
    q = Question.noul("q", "is it?", 0.7)
    assert [c.id for c in q.candidates] == ["false", "true"]
    assert q.target_distribution == {"false": pytest.approx(0.3), "true": pytest.approx(0.7)}


def test_sample_roundtrip():
    s = _sample(state={"a": 1})
    s2 = Sample.from_json(s.to_json())
    assert s2.state == s.state and s2.questions[0].type == "choice"


# ------------------------------------------------------------------- dedup


def test_dedup_ignores_candidate_order():
    q1 = Question(
        "q", "choice", "i", [Candidate("a", "x"), Candidate("b", "y")], {"a": 1.0, "b": 0.0}
    )
    q2 = Question(
        "q", "choice", "i", [Candidate("b", "y"), Candidate("a", "x")], {"b": 0.0, "a": 1.0}
    )
    s = _sample("st", [q1])
    assert sample_dedup_key(s, 0) == sample_dedup_key(_sample("st", [q2]), 0)


def test_dedup_samples_drops_dupes():
    s1 = _sample("st")
    s2 = _sample("st")  # identical
    out = dedup_samples([s1, s2])
    assert len(out) == 1


def test_group_split_no_leakage():
    samples = []
    for i in range(10):
        s = _sample(f"state{i}")
        s.metadata["generator_template_id"] = f"tmpl{i % 3}"
        samples.append(s)
    train, val = group_split(samples, val_frac=0.4, seed=1)
    train_groups = {s.metadata["generator_template_id"] for s in train}
    val_groups = {s.metadata["generator_template_id"] for s in val}
    assert not (train_groups & val_groups)


# -------------------------------------------------------------- transforms


def test_nli_transform():
    row = {"premise": "a man sleeps", "hypothesis": "a man is awake", "label": 2}
    [s] = nli_choice(row, metadata={"source": "snli"})
    q = s.questions[0]
    assert q.type == "choice" and len(q.candidates) == 3
    assert q.target_distribution == {"c0": 0.0, "c1": 0.0, "c2": 1.0}
    assert s.state == {"premise": "a man sleeps", "hypothesis": "a man is awake"}
    assert nli_choice({**row, "label": -1}) == []


def test_chaos_nli_soft_target():
    row = {"premise": "p", "hypothesis": "h", "label_count": [70, 20, 10]}
    [s] = chaos_nli(row)
    assert s.questions[0].target_distribution["c0"] == pytest.approx(0.7)
    assert s.questions[0].target_distribution["c2"] == pytest.approx(0.1)


def test_intent_choice_with_oos_nota():
    onto = {"billing": "user asks about billing", "tech": "user needs tech help"}
    [s] = intent_choice(
        {"text": "refund please", "label": "billing"},
        text_key="text",
        label_key="label",
        ontology=onto,
    )
    assert s.questions[0].target_distribution["billing"] == 1.0
    [s2] = intent_choice(
        {"text": "asdfgh", "label": "oos"},
        text_key="text",
        label_key="label",
        ontology=onto,
        oos_label="oos",
    )
    nota = [c for c in s2.questions[0].candidates if c.is_nota]
    assert nota and s2.questions[0].target_distribution["__nota__"] == 1.0


def test_boolq_and_multirc():
    [s] = boolq_noul(
        {"passage": "earth orbits sun", "question": "does earth orbit sun?", "answer": True}
    )
    assert s.questions[0].target_distribution["true"] == 1.0
    [s2] = multirc_nouls(
        {
            "paragraph": "para",
            "question": "who?",
            "answers": [("alice", True), ("bob", False)],
        }
    )
    assert len(s2.questions) == 2
    assert s2.questions[0].target_distribution["true"] == 1.0
    assert s2.questions[1].target_distribution["true"] == 0.0


def test_multilabel_fractional_nouls():
    row = {"text": "comment", "labels": {"toxic": 0.8, "insult": 0.1}}
    [s] = multilabel_nouls(
        row,
        text_key="text",
        labels_key="labels",
        label_names={"toxic": "toxicity", "insult": "insult"},
        instruction_template="is this {label}?",
        fractional=True,
    )
    assert s.questions[0].target_distribution["true"] == pytest.approx(0.8)


def test_sts_adjacent_mass():
    [s] = sts_score({"sentence1": "a", "sentence2": "b", "score": 3.4}, n_levels=6)
    d = s.questions[0].target_distribution
    assert d["l3"] == pytest.approx(0.6) and d["l4"] == pytest.approx(0.4)
    assert [c.ordinal for c in s.questions[0].candidates] == list(range(6))
    [s2] = ordinal_score(
        {"text": "r", "label": 4},
        text_key="text",
        label_key="label",
        n_levels=5,
        instruction="stars?",
    )
    assert s2.questions[0].target_distribution["l4"] == 1.0


def test_jev_direct_passthrough():
    row = {
        "state": {"x": 1},
        "questions": [
            {
                "id": "q1",
                "type": "choice",
                "instruction": "pick",
                "candidates": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
                "target_distribution": {"a": 0.9, "b": 0.1},
            }
        ],
    }
    [s] = jev_direct(row)
    assert s.questions[0].target_distribution["a"] == pytest.approx(0.9)


# ----------------------------------------------------------------- mixture


def test_language_weights_guardrails():
    w = language_weights({"en": 1000, "ko": 50, "ja": 50})
    assert w["en"] <= 0.60 + 1e-6
    assert w["ko"] >= 0.20 - 1e-6
    assert w["ja"] >= 0.20 - 1e-6
    assert sum(w.values()) == pytest.approx(1.0)


def test_mixture_plan_family_quota():
    pools = {
        ("direct_jev", "en"): list(range(100)),
        ("nli", "en"): list(range(100)),
        ("choice", "ko"): list(range(100)),
    }
    s = MixtureSampler(seed=0)
    alloc = s.plan(pools, 100)
    total = sum(alloc.values())
    assert 80 <= total <= 120  # rounding slack
    assert alloc[("direct_jev", "en")] > alloc[("nli", "en")]


# ----------------------------------------------------------------- augment


def test_permute_candidates_moves_mass():
    q = _choice(k=4)
    rng = random.Random(7)
    p = permute_candidates(q, rng)
    assert sorted(c.id for c in p.candidates) == ["c0", "c1", "c2", "c3"]
    assert p.target_distribution["c0"] == 1.0
    assert len(p.candidates) == 4


def test_remove_candidate_to_nota():
    cands = [Candidate("a", "A"), Candidate("b", "B"), Candidate("n", "nota", is_nota=True)]
    q = Question("q", "choice", "i", cands, {"a": 0.8, "b": 0.1, "n": 0.1})
    q2 = remove_candidate(q, "a")
    assert q2.target_distribution["n"] == pytest.approx(0.9)
    q3 = remove_candidate(q, "n")
    assert q3.target_distribution["a"] == pytest.approx(0.8 / 0.9)


def test_add_nota_and_distractor():
    q = _choice(k=2)
    q2 = add_nota(q)
    assert q2.candidates[-1].is_nota and q2.target_distribution["__nota__"] == 0.0
    rng = random.Random(0)
    q3 = add_distractor(q, "distractor", rng)
    assert len(q3.candidates) == 3
    assert sum(q3.target_distribution.values()) == pytest.approx(1.0)


def test_evidence_deletion_retargets_uniform():
    s = _sample("Sentence one is here. Sentence two is here. Sentence three.")
    rng = random.Random(3)
    v = evidence_deletion_variant(s, rng, target="uniform")
    assert v.metadata["evidence_state"] == "deleted"
    for q in v.questions:
        assert q.target_distribution == uniform(q.candidates)
    assert len(delete_evidence_text("a. b. c. d.", rng)) > 0


def test_group_shared_state_merges_questions():
    s1 = _sample("same state", [_choice("q1")])
    s2 = _sample("same state", [_choice("q2")])
    s3 = _sample("other", [_choice("q3")])
    merged = group_shared_state([s1, s2, s3])
    assert len(merged) == 2
    sizes = sorted(len(m.questions) for m in merged)
    assert sizes == [1, 2]


# ---------------------------------------------------------- jev distill


def test_jev_distill_transform_kinds():
    from ayaka.data.transforms import jev_distill

    noul = jev_distill(
        {
            "kind": "noul",
            "options": ["false", "true"],
            "target": [0.2, 0.8],
            "state": "s",
            "question": "q?",
        }
    )[0]
    q = noul.questions[0]
    assert [c.id for c in q.candidates] == ["false", "true"]
    assert q.target_distribution["true"] == pytest.approx(0.8)
    assert noul.metadata["task_family"] == "direct_jev"

    score = jev_distill(
        {
            "kind": "score",
            "options": ["0", "1", "2"],
            "target": [0.1, 0.1, 0.9],
            "state": "s",
            "question": "rate",
        }
    )[0]
    sq = score.questions[0]
    assert [c.ordinal for c in sq.candidates] == [0, 1, 2]
    assert sum(sq.target_distribution.values()) == pytest.approx(1.0)  # renormalized

    choice = jev_distill(
        {
            "kind": "choice",
            "options": ["escalate", "wait"],
            "target": [0.7, 0.3],
            "state": "s",
            "question": "next?",
        }
    )[0]
    assert [c.description for c in choice.questions[0].candidates] == ["escalate", "wait"]


def test_hf_jsonl_spec_samples_uniformly(tmp_path, monkeypatch):
    import json

    import huggingface_hub

    from ayaka.data import loaders

    path = tmp_path / "train.jsonl"
    with open(path, "w") as f:
        for i in range(100):
            f.write(
                json.dumps(
                    {
                        "id": f"r{i}",
                        "kind": "noul",
                        "options": ["false", "true"],
                        "target": [0.5, 0.5],
                        "state": f"state {i}",
                        "question": "q?",
                    }
                )
                + "\n"
            )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(path))
    samples, man = loaders.load_spec_samples("jev_distill", limit=10, dedup=False, seed=3)
    assert len(samples) == 10
    ids = [s.metadata["source_example_id"] for s in samples]
    assert ids != [f"r{i}" for i in range(10)]  # not a head slice
    again, _ = loaders.load_spec_samples("jev_distill", limit=10, dedup=False, seed=3)
    assert [s.metadata["source_example_id"] for s in again] == ids  # seeded
    assert man.split == "train"


# ------------------------------------------------------------- decontam


def test_decontaminator_drops_overlapping_samples():
    from ayaka.data.decontam import Decontaminator

    bench = "Policy: refunds require a receipt and purchase within 30 days. A customer bought 12 days ago but has no receipt."
    d = Decontaminator([bench, "Short exact item text"], n=8)
    leak = _sample(
        "Background. refunds require a receipt and purchase within 30 days. A customer asked."
    )
    clean = _sample("An unrelated incident report about a failing disk in rack four.")
    exact = _sample("short   EXACT item text")
    kept, dropped = d.filter([leak, clean, exact])
    assert kept == [clean] and dropped == 2


def test_decontaminator_loads_vendored_jevbench():
    import json
    import os

    from ayaka.data.decontam import Decontaminator, jevbench_public_dir

    d = Decontaminator.from_jevbench()
    with open(os.path.join(jevbench_public_dir(), "hard.jsonl"), encoding="utf-8") as f:
        rec = json.loads(f.readline())
    assert d.text_hit("Preamble. " + rec["state"] + " Trailing words.")
    assert not d.text_hit(
        "A completely unrelated sentence about sailing boats and the weather at sea today, nothing else."
    )


def test_reading_mc_skips_long_articles_instead_of_truncating():
    from ayaka.data.transforms import reading_mc

    row = {
        "article": "word " * 50,
        "question": "What?",
        "options": ["a", "b", "c", "d"],
        "answer": 2,
    }
    kw = {
        "article_key": "article",
        "question_key": "question",
        "options_key": "options",
        "label_key": "answer",
    }
    s = reading_mc(row, **kw, max_chars=1000)[0]
    q = s.questions[0]
    assert q.instruction == "What?" and q.target_distribution["c2"] == 1.0
    assert s.metadata["task_family"] == "long_context"
    assert reading_mc(row, **kw, max_chars=100) == []
    assert reading_mc({**row, "options": ["a", "a", "b", "c"]}, **kw) == []
