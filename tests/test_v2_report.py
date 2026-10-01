import copy
import json

import pytest

from ayaka.data.reasoning_v2 import curriculum
from ayaka.eval.reasoning_v2 import dataset_signature
from ayaka.eval.v2 import typed_row
from ayaka.experiments.report import collect, compact_evaluation
from ayaka.primitives import QuestionSpec
from ayaka.training.batching import _noul_canonical


def measured():
    samples = [s for s, _ in curriculum("dev", 2)]
    rows = []
    for s in samples:
        for original in s.questions:
            q = _noul_canonical(original)
            spec = QuestionSpec(
                q.type,
                q.instruction,
                [c.description for c in q.candidates],
                [c.ordinal for c in q.candidates] if q.type == "score" else None,
            )
            target = [q.target_distribution.get(c.id, 0) for c in q.candidates]
            rows.append(
                {
                    **typed_row(spec, target, target),
                    "id": f"{s.metadata['source_example_id']}/{q.id}",
                    "budget": 0,
                    "reasoning_tokens": 0,
                    "route": "direct",
                    "error": None,
                    "finish_reason": "disabled",
                    "latency_s": 0.1,
                }
            )
    low = [
        {**r, "budget": 128, "reasoning_tokens": 9, "route": "reasoning", "finish_reason": "eos"}
        for r in rows
    ]
    return samples, {
        "status": "complete",
        "dataset_signature": dataset_signature(samples),
        "rows": {"off": rows, "low": low},
        "reports": {},
    }


def test_recompute_uses_underlying_cases_without_changing_measured_rows():
    samples, report = measured()
    original = copy.deepcopy(report)
    result = compact_evaluation(report, samples)
    assert report == original
    assert result["paired"]["low"]["independent_cases"] == 2
    assert result["paired"]["low"]["cc_delta_95ci"] == [0, 0]
    assert result["reports"]["low"]["reasoning_tokens"] == 54
    assert result["reports"]["low"]["finish_reasons"] == {"eos": 6}
    assert result["reports"]["off"]["reasoning_tokens"] == 0
    assert result["reports"]["low"]["official_composite"] is None


def test_recompute_rejects_different_preparation_or_case_identity():
    samples, report = measured()
    report["dataset_signature"] = "wrong"
    with pytest.raises(ValueError, match="prepared split"):
        compact_evaluation(report, samples)
    report["dataset_signature"] = dataset_signature(samples)
    report["rows"]["off"][0]["cluster_id"] = "other"
    with pytest.raises(ValueError, match="identity changed"):
        compact_evaluation(report, samples)


def test_collector_preserves_incomplete_status_and_hashes_sources(tmp_path):
    samples, report = measured()
    report["status"] = "incomplete"
    (tmp_path / "dev.json").write_text(json.dumps([{"sample": s.to_json()} for s in samples]))
    (tmp_path / "screen").mkdir()
    source = tmp_path / "screen" / "example.json"
    source.write_text(json.dumps(report))
    result = collect(tmp_path)
    assert result["screen"]["example"]["status"] == "incomplete"
    assert len(result["source_artifact_sha256"]["screen/example.json"]) == 64
    assert "selection" not in result
    assert "cluster_id" not in json.loads(source.read_text())["rows"]["off"][0]
