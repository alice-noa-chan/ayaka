import copy

import pytest
import torch

from ayaka.collate import EncodedQuestion, full_rows
from ayaka.config import tiny_config
from ayaka.eval.recovery_v2 import (
    fit_report_calibrations,
    promotion_screen,
    recalibrate_report,
    verified_trace_readout,
)
from ayaka.eval.v2 import typed_row
from ayaka.model.decision import PRIMITIVE_INDEX, AyakaDecisionModel
from ayaka.model.ragged import ragged_softmax
from ayaka.primitives import QuestionSpec
from ayaka.reasoning_pipeline import TraceGenerator, readout_suffix
from ayaka.tokenization import ToyTokenizer


def report(split, count=240, probability=0.9):
    rows = []
    for i in range(count):
        kind = ("choice", "noul", "score")[i % 3]
        spec = QuestionSpec(kind, "", ["zero", "one"], [9, 3] if kind == "score" else None)
        target = [float(i % 4 != 0), float(i % 4 == 0)]
        probs = [probability, 1 - probability]
        row = typed_row(spec, probs, target)
        row.update(
            id=str(i),
            cluster_id=str(i),
            type=kind,
            ordinals=spec.ordinals,
            split=split,
            model_id="a" * 64,
            modality="text",
            partition="fixed",
            route="direct",
            budget=0,
            target=target,
            probs=probs,
            family="calendar",
            language=("en", "ko", "ja")[(i // 3) % 3],
        )
        rows.append(row)
    return {"complete": True, "split": split, "model_id": "a" * 64, "rows": {"off": rows}}


def test_calibration_is_fit_only_on_reserved_split_and_recomputes_score_metrics():
    fits = fit_report_calibrations(report("calibration"))
    original = report("dev")
    before = copy.deepcopy(original)
    result = recalibrate_report(original, fits)
    assert original == before
    score = next(row for row in result["rows"]["off"] if row["type"] == "score")
    expected = typed_row(
        QuestionSpec("score", "", ["zero", "one"], [9, 3]), score["probs"], score["target"]
    )
    assert score["rps"] == expected["rps"]
    assert score["expected_position"] == expected["expected_position"]
    assert result["calibration"] == "reserved_split_scoped_temperature"
    with pytest.raises(ValueError, match="reserved"):
        fit_report_calibrations(report("test"))
    foreign = report("dev")
    foreign["rows"]["off"][0]["model_id"] = "b" * 64
    with pytest.raises(ValueError, match="binding"):
        recalibrate_report(foreign, fits)


def test_screen_refuses_regression_uncertain_small_runs_and_test_selection():
    baseline = report("dev")
    result = promotion_screen(baseline, copy.deepcopy(baseline), replicates=30)
    assert not result["screen_passed"] and result["final_test_required"]
    assert "overall_gain_below_5_points" in result["failures"]
    assert "overall_gain_not_supported_by_paired_95ci" in result["failures"]
    with pytest.raises(ValueError, match="dev"):
        promotion_screen(baseline, report("test"), replicates=30)
    candidate = copy.deepcopy(baseline)
    candidate["rows"]["off"][0]["target"] = [1, 0]
    with pytest.raises(ValueError, match="unmatched"):
        promotion_screen(baseline, candidate, replicates=30)
    result = promotion_screen(report("dev", 24), report("dev", 24), replicates=30)
    assert "fewer_than_200_independent_dev_cases" in result["failures"]


def test_screen_accepts_large_paired_gain_with_better_probability_quality():
    baseline, candidate = report("dev"), report("dev")
    for left, right in zip(baseline["rows"]["off"], candidate["rows"]["off"], strict=True):
        target = left["target"]
        wrong = [0.1 if value else 0.9 for value in target]
        better = [0.9 if value else 0.1 for value in target]
        left.update(
            typed_row(QuestionSpec(left["type"], "", ["a", "b"], left["ordinals"]), wrong, target),
            probs=wrong,
        )
        right.update(
            typed_row(
                QuestionSpec(right["type"], "", ["a", "b"], right["ordinals"]), better, target
            ),
            probs=better,
        )
    result = promotion_screen(baseline, candidate, replicates=30)
    assert result["screen_passed"] and result["final_test_required"]


def test_verified_trace_cache_matches_full_forward_without_generation():
    torch.set_num_threads(1)
    model = AyakaDecisionModel.from_config(tiny_config(version=2), dtype=torch.float32).eval()
    tok = ToyTokenizer()
    generator = TraceGenerator(model, tok, apply_temperature=False)
    spec = QuestionSpec("choice", "Select", ["a", "b", "c"])
    notes = "Two days after Monday is Wednesday."
    ids, _ = generator.prepare(generator.messages_for("Today is Monday", spec))
    rendered = readout_suffix(tok, spec)
    item = EncodedQuestion(ids + tok.encode(notes), rendered, PRIMITIVE_INDEX[spec.type])
    with torch.no_grad():
        output = model(full_rows([item], tok.pad_id), apply_temperature=False)
        expected = ragged_softmax(output.logits, output.cand_cu).tolist()
    generator.generate_trace = lambda *a, **kw: pytest.fail("oracle must not generate")
    assert verified_trace_readout(generator, "Today is Monday", spec, notes) == pytest.approx(
        expected, abs=1e-5
    )
    generator.max_context = 1
    with pytest.raises(ValueError, match="fit"):
        verified_trace_readout(generator, "Today is Monday", spec, notes)
