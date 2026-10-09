"""Opt-in StrategyQA rehearsal: shared held-out task view, raw gold and plan3 inventory."""

import copy

import pytest
from test_direct_natural import raw_registry

from ayaka.data.direct_natural import (
    FINAL_VERSION,
    STRATEGYQA_VIEW,
    NaturalGoldRegistry,
    task_view,
)
from ayaka.data.natural_training_v2 import EXTRA_SOURCES, FINAL_SOURCES, valid_provenance
from ayaka.data.transforms import strategyqa_noul
from ayaka.eval.read_artifact import fingerprint
from ayaka.training.direct_corpus_plan import (
    FINAL_VERSION as PLAN_FINAL_VERSION,
)
from ayaka.training.direct_corpus_plan import (
    SOURCE_WIDTHS,
    validate_settings,
)


def strategy_rows(n=40):
    return [
        {
            "qid": f"q{i}",
            "term": f"term {i}",
            "description": f"description {i}",
            "question": f"Original implicit yes/no question {i}?",
            "answer": bool(i % 2),
            "facts": f"Supporting fact for original question {i}.",
        }
        for i in range(n)
    ]


def extra_registry(rows=None):
    raw = copy.deepcopy(raw_registry().raw)
    rows = strategy_rows() if rows is None else rows
    repo, revision, filename, _ = EXTRA_SOURCES["strategyqa"]
    raw["strategyqa"] = {
        "rows": rows,
        "features": {},
        "provenance": {
            "repo": repo,
            "revision": revision,
            "file": filename,
            "sha256": fingerprint(rows),
        },
    }
    return NaturalGoldRegistry(raw, extra_sources=("strategyqa",))


def test_strategyqa_training_view_matches_the_heldout_transform():
    row = strategy_rows()[3]
    state, questions, view = task_view("strategyqa", row)
    (expected,) = strategyqa_noul(row)
    assert state == expected.state
    assert questions == expected.questions
    assert view == STRATEGYQA_VIEW


def test_strategyqa_gold_is_rechecked_from_the_raw_boolean():
    registry = extra_registry()
    sample = registry.sample("strategyqa", 3)
    assert registry(sample, sample.questions[0]) == {"false": 0.0, "true": 1.0}
    assert sample.metadata["license"] == "MIT"
    assert sample.metadata["language"] == "en"
    assert valid_provenance(sample.metadata)
    assert registry.binding["version"] == FINAL_VERSION
    assert registry.source_names == set(FINAL_SOURCES) - {"contract_nli"}


def test_strategyqa_gold_rejects_changed_evidence_and_non_boolean_answers():
    registry = extra_registry()
    sample = registry.sample("strategyqa", 2)
    changed = copy.deepcopy(sample)
    changed.state = "A different supporting fact."
    with pytest.raises(ValueError):
        registry(changed, changed.questions[0])
    rows = strategy_rows()
    rows[0]["answer"] = 1
    broken = extra_registry(rows)
    with pytest.raises(ValueError):
        broken(broken.sample("strategyqa", 0), broken.sample("strategyqa", 0).questions[0])


def test_unknown_extra_sources_and_unrequested_extras_are_rejected():
    with pytest.raises(ValueError):
        NaturalGoldRegistry(raw_registry().raw, extra_sources=("hotpotqa",))
    raw = extra_registry().raw
    with pytest.raises(ValueError):
        NaturalGoldRegistry(raw)  # StrategyQA rows without opting in


def test_final_plan_settings_require_the_complete_final_inventory():
    quota = {"train": 4, "router_train": 1, "dev": 1, "calibration": 1, "test": 1}
    settings = {
        "natural_sample_quotas": {source: dict(quota) for source in FINAL_SOURCES},
        "authored_per_type": dict(quota),
        "epochs": 1,
        "rows_per_step": 8,
        "seed": 1,
        "minimum_english_question_fraction": 0.5,
    }
    assert validate_settings(settings) is settings
    assert SOURCE_WIDTHS["strategyqa"] == (1, "noul")
    assert PLAN_FINAL_VERSION == "ayaka-direct-corpus-plan-3"
    partial = copy.deepcopy(settings)
    del partial["natural_sample_quotas"]["contract_nli"]
    with pytest.raises(ValueError):
        validate_settings(partial)
