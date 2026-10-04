import json
from dataclasses import replace

import pytest

from ayaka.swift.collect import adapt_jevbench, iter_dataset
from ayaka.swift.readers import FakeReader
from scripts.swift import parity
from scripts.swift.build_parity_cohort import build


def item():
    return adapt_jevbench(
        {
            "id": "fixture:parity",
            "state": "CPU",
            "split": "calibration",
            "public": False,
            "labels": ["a", "b"],
            "expected": "b",
            "question": {"type": "choice", "criteria": ["a", "b"]},
        }
    )


def test_r14_tail_counterexample_fails_centered_log_mass():
    result = parity.compare(
        [item()], FakeReader([{"A": 1, "B": 1e-40}]), FakeReader([{"A": 1, "B": 1e-9}]), n=1
    )
    assert result["max_abs"] < 0.02 and result["argmax_agreement"] == 1
    assert result["centered_log_mass_max_abs"] == pytest.approx(71.38013788181541 / 2)
    assert result["centered_log_mass_max_abs_threshold_nats"] == 0.05
    assert result["passed"] is False and result["comparison_valid"] is False


@pytest.mark.parametrize("change", ["prompt", "ids", "missing", "nonfinite"])
def test_parity_requires_identical_tokens_and_complete_finite_gather(change):
    reference = FakeReader().read([], ["A", "B"])
    changed = reference
    if change == "prompt":
        changed = replace(changed, input_token_ids=[123])
    elif change == "ids":
        changed = replace(
            changed,
            canonical_token_ids={"A": [0], "B": [1]},
            token_logits={0: reference.token_logits[65], 1: reference.token_logits[66]},
        )
    elif change == "missing":
        changed = replace(changed, token_logits={65: 0})
    else:
        changed = replace(changed, token_logits={65: float("inf"), 66: 0})

    class Reader:
        def __init__(self, result):
            self.result = result

        def read(self, *args):
            return self.result

    if change in ("missing", "nonfinite"):
        with pytest.raises(ValueError, match="incomplete|finite"):
            parity.compare([item()], Reader(reference), Reader(changed))
    else:
        assert parity.compare([item()], Reader(reference), Reader(changed))["passed"] is False


def test_common_raw_logit_offset_does_not_fail_parity():
    from ayaka.swift.readers import ReadResult, logmass_probs

    masses = {"A": 1000.0, "B": 998.0}
    result = ReadResult(
        logmass_probs(masses),
        1,
        1,
        0.0,
        masses,
        [7],
        {"A": [65], "B": [66]},
        {65: 1000.0, 66: 998.0},
    )
    shifted = replace(
        result, letter_log_masses={"A": 0.0, "B": -2.0}, token_logits={65: 0.0, 66: -2.0}
    )

    class Reader:
        def __init__(self, r):
            self.result = r

        def read(self, *args):
            return self.result

    assert parity.compare([item()], Reader(result), Reader(shifted))["passed"]


def test_stored_cohort_is_deterministic_nonpublic_and_covers_stress_cases():
    rows = build()
    stored = [json.loads(line) for line in parity.COHORT.read_text(encoding="utf-8").splitlines()]
    assert rows == stored == build()
    cohort = list(iter_dataset([parity.COHORT]))
    assert {2, 20, 26} <= {len(i.question.labels) for i in cohort}
    assert all(not i.public and i.split == "calibration" for i in cohort)
    assert any(r["provenance"]["skew_fixture"] for r in rows)
    assert sum(bool(r["provenance"].get("permutation_of")) for r in rows) == 2
    original = next(r for r in rows if r["id"] == "parity:twentysix")
    for row in rows[-2:]:
        assert row["state"] == original["state"]
        assert row["question"]["criteria"] == original["question"]["criteria"]
        assert row["labels"] != original["labels"]
        adapted = next(i for i in cohort if i.id == row["id"])
        assert adapted.question.labels == row["labels"]
        assert adapted.question.descriptions == [
            original["question"]["criteria"][label] for label in row["labels"]
        ]


def test_two_phase_parity_preserves_failed_diagnostic(tmp_path, monkeypatch):
    class HF(FakeReader):
        def _load(self):
            pass

    monkeypatch.setattr(parity, "HFReader", lambda *a, **kw: HF())
    monkeypatch.setattr(
        parity,
        "VLLMChatReader",
        lambda *a, **kw: FakeReader(
            lambda m, letters: {k: (0.999999999 if k == letters[0] else 1e-9) for k in letters}
        ),
    )
    reference = tmp_path / "reference.json"
    output = tmp_path / "parity.json"
    common = ["--model", "fixture", "--revision", "a" * 40, "--prompt-variants", "min"]
    assert parity.main([*common, "--phase", "reference", "--output", str(reference)]) == 0
    saved = json.loads(reference.read_text())
    assert saved["hf_load_s"] >= 0 and len(saved["rows"]) == 7
    assert parity.main([*common, "--reference", str(reference), "--output", str(output)]) == 1
    saved = json.loads(output.read_text())
    assert saved["comparison_valid"] is False
    assert len(saved["variants"]["min"]["samples"]) == 7
    assert saved["hf_load_s"] >= 0


@pytest.mark.parametrize("failure", ["wire", "nonfinite", "interrupt"])
def test_partial_comparison_is_saved_before_next_read_and_on_failure(
    tmp_path, monkeypatch, failure
):
    class HF(FakeReader):
        def _load(self):
            pass

    monkeypatch.setattr(parity, "HFReader", lambda *a, **kw: HF())
    reference = tmp_path / "reference.json"
    output = tmp_path / "parity.json"
    common = ["--model", "fixture", "--revision", "a" * 40, "--prompt-variants", "min"]
    assert parity.main([*common, "--phase", "reference", "--output", str(reference)]) == 0

    class FailingReader(FakeReader):
        def read(self, messages, letters):
            if self.calls:
                saved = json.loads(output.read_text())
                assert saved["comparison_valid"] is False
                assert saved["variants"]["min"]["samples"][0]["complete_finite"] is True
                if failure == "interrupt":
                    raise KeyboardInterrupt("CPU interrupt fixture")
                if failure == "wire":
                    raise ValueError("missing requested canonical token ids")
                result = super().read(messages, letters)
                return replace(result, token_logits={65: float("nan")})
            return super().read(messages, letters)

    monkeypatch.setattr(parity, "VLLMChatReader", lambda *a, **kw: FailingReader())
    assert parity.main([*common, "--reference", str(reference), "--output", str(output)]) == 1
    saved = json.loads(output.read_text())
    assert saved["complete"] is False and saved["comparison_valid"] is False
    assert len(saved["variants"]["min"]["samples"]) == 2
    assert saved["error"]
    assert "NaN" not in output.read_text()


def test_missing_cohort_still_writes_invalid_artifact(tmp_path):
    output = tmp_path / "parity.json"
    assert (
        parity.main(
            [
                str(tmp_path / "missing.jsonl"),
                "--model",
                "fixture",
                "--revision",
                "a" * 40,
                "--output",
                str(output),
            ]
        )
        == 1
    )
    saved = json.loads(output.read_text())
    assert saved["comparison_valid"] is False and saved["complete"] is False
    assert "FileNotFoundError" in saved["error"]


def test_centered_log_mass_checks_underflowed_probabilities():
    from ayaka.swift.readers import logmass_probs

    left = FakeReader().read([], ["A", "B"])
    left = replace(
        left,
        letter_probs=logmass_probs({"A": 0.0, "B": -1000.0}),
        letter_log_masses={"A": 0.0, "B": -1000.0},
        token_logits={65: 0.0, 66: -1000.0},
    )
    right = replace(
        left,
        letter_log_masses={"A": 0.0, "B": -1010.0},
        token_logits={65: 0.0, 66: -1010.0},
    )

    class Reader:
        def __init__(self, result):
            self.result = result

        def read(self, *args):
            return self.result

    result = parity.compare([item()], Reader(left), Reader(right))
    assert result["max_abs"] == 0 and result["argmax_agreement"] == 1
    assert result["centered_log_mass_max_abs"] == 5
    assert result["comparison_valid"] is False
