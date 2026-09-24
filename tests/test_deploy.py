"""Tests for dataset loaders, manifests, the run driver, and the
beam deployment entrypoint (sec 37/49). HF/datasets/beta9 are optional
deps — loader internals are exercised with a fake dataset object and
jsonl specs; the beam module is import-checked only when beta9 exists.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ayaka.data.loaders import (
    DATASET_SPECS,
    _apply,
    _group_multirc,
    _rows_to_samples,
    load_spec_samples,
)
from ayaka.data.manifest import DatasetManifest, write_manifest

# ------------------------------------------------------------- fake HF ds


class _ClassLabel:
    def __init__(self, names):
        self.names = names


class _Sequence:
    def __init__(self, feature):
        self.feature = feature


class _Info:
    version = "1.0.0"


class FakeDataset:
    """Minimal stand-in exposing .features like a HF Dataset."""

    def __init__(self, features):
        self.features = features
        self.info = _Info()


@pytest.fixture
def intent_ds():
    return FakeDataset({"intent": _ClassLabel(["card_arrival", "lost_card", "oos"])})


@pytest.fixture
def multilabel_ds():
    return FakeDataset({"labels": _Sequence(_ClassLabel(["admiration", "anger"]))})


@pytest.fixture
def nli_ds():
    return FakeDataset({"label": _ClassLabel(["entailment", "neutral", "contradiction"])})


# -------------------------------------------------------------- _apply


def test_apply_resolves_ontology(intent_ds):
    kw = {"text_key": "text", "label_key": "intent", "ontology_from_features": "intent"}
    out = _apply("intent", kw, {"text": "hi", "intent": 1}, intent_ds, {})
    (s,) = out
    q = s.questions[0]
    assert q.type == "choice"
    assert len(q.candidates) == 3
    assert q.candidates[1].description == "lost card"
    assert q.target_distribution["1"] == 1.0


def test_apply_oos_label_maps_to_id(intent_ds):
    kw = {
        "text_key": "text",
        "label_key": "intent",
        "ontology_from_features": "intent",
        "oos_label": "oos",
    }
    (s,) = _apply("intent", kw, {"text": "hi", "intent": 2}, intent_ds, {})
    q = s.questions[0]
    nota = [c for c in q.candidates if c.is_nota]
    assert len(nota) == 1
    assert q.target_distribution["__nota__"] == 1.0


def test_apply_labels_from_features(nli_ds):
    (s,) = _apply(
        "nli",
        {"labels_from_features": "label"},
        {"premise": "p", "hypothesis": "h", "label": 0},
        nli_ds,
        {},
    )
    q = s.questions[0]
    assert [c.description for c in q.candidates] == ["entailment", "neutral", "contradiction"]
    assert q.target_distribution["c0"] == 1.0


def test_apply_multilabel_list_to_dict(multilabel_ds):
    kw = {
        "text_key": "text",
        "labels_key": "labels",
        "label_names_from_features": "labels",
        "label_list_to_dict": True,
        "instruction_template": "Does this text express {label}?",
    }
    (s,) = _apply("multilabel", kw, {"text": "t", "labels": [1]}, multilabel_ds, {})
    assert len(s.questions) == 2
    assert s.questions[0].target_distribution["true"] == 0.0
    assert s.questions[1].target_distribution["true"] == 1.0


def test_apply_score_from_labels():
    ds = FakeDataset({})
    row = {
        "sentence1": "a",
        "sentence2": "b",
        "labels": {"label": 3, "real-label": 3.5},
    }
    kw = {
        "sent1_key": "sentence1",
        "sent2_key": "sentence2",
        "score_key": "labels",
        "score_from_labels": "real-score",
        "n_levels": 6,
    }
    (s,) = _apply("sts", kw, row, ds, {})
    q = s.questions[0]
    assert q.type == "score"
    assert q.target_distribution["l3"] == pytest.approx(0.5)
    assert q.target_distribution["l4"] == pytest.approx(0.5)


# ------------------------------------------------------- multirc grouping


def test_group_multirc():
    rows = [
        {
            "idx": {"paragraph": 0, "question": 0},
            "paragraph": "p",
            "question": "q?",
            "answer": "a1",
            "label": 1,
        },
        {
            "idx": {"paragraph": 0, "question": 0},
            "paragraph": "p",
            "question": "q?",
            "answer": "a2",
            "label": 0,
        },
        {
            "idx": {"paragraph": 0, "question": 1},
            "paragraph": "p",
            "question": "q2?",
            "answer": "a3",
            "label": 1,
        },
    ]
    groups = list(_group_multirc(rows))
    assert len(groups) == 2
    assert groups[0]["answers"] == [("a1", True), ("a2", False)]
    assert groups[1]["answers"] == [("a3", True)]


def test_rows_to_samples_multirc_grouped():
    spec = {"transform": "multirc_grouped", "kwargs": {}}
    rows = [
        {
            "idx": {"paragraph": 0, "question": 0},
            "paragraph": "p",
            "question": "q?",
            "answer": "a1",
            "label": 1,
        },
        {
            "idx": {"paragraph": 0, "question": 0},
            "paragraph": "p",
            "question": "q?",
            "answer": "a2",
            "label": 0,
        },
    ]
    out = _rows_to_samples(spec, rows, None, {}, None)
    assert len(out) == 1
    assert len(out[0].questions) == 2  # shared-state independent nouls


# ----------------------------------------------------------- jsonl specs


def test_load_spec_jsonl_jev(tmp_path):
    row = {
        "state": {"text": "hello"},
        "questions": [
            {
                "type": "noul",
                "instruction": "is this fine?",
                "candidates": [
                    {"id": "false", "description": "no"},
                    {"id": "true", "description": "yes"},
                ],
                "target_distribution": {"false": 0.2, "true": 0.8},
            }
        ],
    }
    p = tmp_path / "jev.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    spec_name = "tmp_jev"
    DATASET_SPECS[spec_name] = {
        "jsonl": str(p),
        "transform": "jev",
        "kwargs": {},
        "family": "direct_jev",
        "lang": "en",
        "license": "test",
        "label_schema": "direct",
    }
    try:
        samples, manifest = load_spec_samples(spec_name, dedup=False)
    finally:
        del DATASET_SPECS[spec_name]
    assert len(samples) == 1
    assert samples[0].questions[0].target_distribution["true"] == 0.8
    assert manifest.dataset_id == spec_name
    assert manifest.task_family == "direct_jev"


# ---------------------------------------------------------------- manifest


def test_manifest_roundtrip(tmp_path):
    recs = [
        DatasetManifest(
            dataset_id="d1",
            source_url="hf://x/y/z",
            revision="abc123",
            config="y",
            split="z",
            license="MIT",
            language="en",
            task_family="nli",
            primitive_mapping="nli",
            original_label_schema="e/n/c",
        )
    ]
    path = tmp_path / "m.jsonl"
    write_manifest(recs, path)
    loaded = json.loads(path.read_text().strip())
    assert loaded["dataset_id"] == "d1"
    assert loaded["transformation_version"]
    assert loaded["dedup_version"]


# ---------------------------------------------------------------- beam app


def test_beam_app_importable():
    if importlib.util.find_spec("beam") is None:
        pytest.skip("beam sdk not installed in this env (beam CLI env only)")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        import beam_train  # noqa: F401
    finally:
        sys.path.pop(0)


def test_pipeline_set_parsing():
    from ayaka.pipeline import parse_sets

    got = parse_sets(
        ["steps=20", "liger=true", "specs=jev_distill,quality", 'spec_limits={"jev_distill":10}']
    )
    assert got == {
        "steps": 20,
        "liger": True,
        "specs": ["jev_distill", "quality"],
        "spec_limits": {"jev_distill": 10},
    }
    with pytest.raises(SystemExit):
        parse_sets(["not_a_field=1"])
