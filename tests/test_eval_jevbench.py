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
from ayaka.model.decision import AyakaDecisionModel
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
    m = AyakaDecisionModel.from_config(tiny_config(), dtype=torch.float32).eval()
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


@pytest.mark.parametrize(
    ("axes", "published"),
    [
        # JevBench v1.4.2.1 published rows: (I, C, S, K) -> composite
        ((53.1, 76.3, 83.3, 52.0), 63.29),  # Jev 1.13.0
        ((53.0, 75.5, 93.5, 55.8), 65.8),  # Plumb-4B
        ((44.2, 54.9, 77.0, 64.8), 45.1),  # system-one-open (Intelligence < 50 penalty)
    ],
)
def test_jevbench_score_reproduces_published_composites(axes, published):
    from ayaka.eval.jevbench import jevbench_score

    assert jevbench_score(*axes) == pytest.approx(published, abs=0.15)


def test_speed_and_cost_axes_match_published_values():
    from ayaka.eval.jevbench import cost_axis, speed_axis, speed_score

    assert speed_score(0.1) == 100 and speed_score(1.0) == pytest.approx(80)
    # Jev 1.13.0 (hosted API, no adjustment): p50 0.652 s, p95 0.722 s -> 83.3
    assert speed_axis(0.6524, 0.7222, self_hosted=False) == pytest.approx(83.3, abs=0.1)
    # self-hosted: x2 + 0.15 s
    assert speed_axis(0.1, 0.1) == pytest.approx(speed_score(0.35))
    assert cost_axis(0.03991) == pytest.approx(52.0, abs=0.1)  # Jev 1.13.0
    assert cost_axis(0.01488) == pytest.approx(64.8, abs=0.1)  # system-one-open (E2B)


def test_leaderboard_estimate_uses_original_tier_latency_and_backbone_cost():
    from ayaka.eval.jevbench import leaderboard_estimate, speed_axis

    tiers = {
        "easy": {"accuracy": 1.0, "chance": 0.3, "latency_p50_s": 9.0, "latency_p95_s": 9.0},
        "original": {"accuracy": 0.9, "chance": 0.3, "latency_p50_s": 0.1, "latency_p95_s": 0.2},
        "hard": {"accuracy": 0.5, "chance": 0.3, "latency_p50_s": 9.0, "latency_p95_s": 9.0},
    }
    est = leaderboard_estimate(tiers, backbone="google/gemma-4-12B-it")
    assert est["speed"] == pytest.approx(speed_axis(0.1, 0.2))
    assert est["cost"] is not None and est["jevbench_score"] is None
    assert leaderboard_estimate(tiers, "google/gemma-4-12B-it", calibration=70.0)["jevbench_score"]
    assert leaderboard_estimate(tiers, backbone="unknown")["cost"] is None
