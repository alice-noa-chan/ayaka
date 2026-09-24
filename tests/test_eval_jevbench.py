import json
import os

import pytest
import torch

from ayaka.config import tiny_config
from ayaka.eval.jevbench import (
    TIERS,
    compare_references,
    data_dir,
    intelligence_proxy,
    load_jsonl,
    record_to_item,
    reference_outcomes,
    run_jevbench,
)
from ayaka.model.electra import ElectraDecisionModel
from ayaka.tokenization import ToyTokenizer


def test_public_items_vendored():
    counts = {t: len(load_jsonl(os.path.join(data_dir(), f"{t}.jsonl"))) for t in TIERS}
    assert counts == {"easy": 48, "original": 72, "hard": 111}


def test_all_public_records_map_without_leaking_answers():
    for t in TIERS:
        for rec in load_jsonl(os.path.join(data_dir(), f"{t}.jsonl")):
            item = record_to_item(rec)
            assert item.expected in item.labels
            assert len(item.spec.candidates) == len(item.labels)
            # the model sees descriptions/instruction/state only
            assert "expected" not in json.dumps(item.spec.__dict__)


def test_noul_mapping_false_first():
    rec = {
        "id": "x",
        "expected": "yes",
        "labels": ["yes", "no"],
        "question": {
            "type": "noul",
            "instructions": "ok?",
            "criteria": {"false": "F desc", "true": "T desc"},
        },
        "state": "s",
    }
    item = record_to_item(rec)
    assert item.labels == ["no", "yes"]
    assert item.spec.candidates == ["F desc", "T desc"]


def test_score_mapping_ordinals():
    rec = {
        "id": "x",
        "expected": "2",
        "labels": ["0", "1", "2"],
        "question": {"type": "score", "instructions": "rate", "criteria": ["bad", "ok", "great"]},
        "state": "s",
    }
    item = record_to_item(rec)
    assert item.spec.ordinals == [0, 1, 2]
    assert item.spec.candidates == ["bad", "ok", "great"]


def test_intelligence_proxy_chance_corrected():
    acc = {"easy": 1.0, "original": 1.0, "hard": 1.0}
    ch = {"easy": 0.25, "original": 0.4, "hard": 0.3}
    assert intelligence_proxy(acc, ch) == pytest.approx(100.0)
    assert intelligence_proxy({k: ch[k] for k in ch}, ch) == pytest.approx(0.0)


def test_reference_outcomes_cover_public_items():
    refs = reference_outcomes()
    assert "jev-1.13.0" in refs
    ids = {r["id"] for t in TIERS for r in load_jsonl(os.path.join(data_dir(), f"{t}.jsonl"))}
    assert ids <= set(refs["jev-1.13.0"]["outcomes"])


def test_run_jevbench_tiny_smoke(tmp_path):
    m = ElectraDecisionModel.from_config(tiny_config(), dtype=torch.float32).eval()
    rep = run_jevbench(m, ToyTokenizer(), out_path=str(tmp_path / "r.json"), limit=2, verbose=False)
    s = rep["summary"]
    assert set(s["accuracy"]) == set(TIERS)
    assert "jev-1.13.0" in s["references"]
    for tier in TIERS:
        for r in rep["tiers"][tier]["results"]:
            assert sum(r["probs"].values()) == pytest.approx(1.0, abs=1e-4)
    assert (tmp_path / "r.json").exists()
    refs = compare_references(rep["tiers"])
    assert all(0 <= a <= 1 for a in refs["jev-1.13.0"]["accuracy"].values())
