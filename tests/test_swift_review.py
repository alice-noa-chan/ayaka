"""Regression checks for the accepted Swift review findings."""

import json
import math
from dataclasses import replace

import pytest

from ayaka.swift.collect import adapt_canonical, adapt_jevbench, collect, load_reads
from ayaka.swift.evaluate import cluster_strata, evaluate, paired_bootstrap
from ayaka.swift.fit import fit_policy, nll
from ayaka.swift.policy import Policy
from ayaka.swift.readers import FakeReader, canonical_letter_ids, top_letter_probs
from ayaka.swift.score import score_reads
from scripts.swift.parity import compare


def reads():
    return [
        {
            "id": kind,
            "split": "calibration",
            "public": False,
            "type": kind,
            "labels": list(probs),
            "raw_probs": probs,
            "gold": gold,
        }
        for kind, probs, gold in [
            ("choice", {"a": 0.8, "b": 0.2}, "a"),
            ("noul", {"false": 0.4, "true": 0.6}, "true"),
            ("score", {"0": 0.2, "1": 0.8}, "1"),
        ]
    ]


def item(**changes):
    return adapt_jevbench(
        {
            "id": "q",
            "split": "calibration",
            "state": "original",
            "labels": ["a", "b"],
            "expected": "a",
            "question": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
            **changes,
        }
    )


def test_default_and_all_emitted_probability_arms():
    assert Policy().commit_margin is None
    assert Policy().apply("noul", {"false": 0.4, "true": 0.6})["true"] == 0.6
    report = evaluate(reads(), Policy(), B=10)
    for arm in report["ablation"].values():
        assert set(arm["per_type_probability"]) == {"choice", "noul", "score"}
        for metrics in arm["per_type_probability"].values():
            assert {"NLL", "Brier", "ECE"} <= metrics.keys()
        assert set(arm["per_type_CC"]) == {"choice", "noul", "score"}
    raw = report["ablation"]["raw"]["per_type_probability"]["noul"]
    committed = report["ablation"]["temps_commit"]["per_type_probability"]["noul"]
    assert raw["NLL"] == pytest.approx(-math.log(0.6))
    assert committed["NLL"] == pytest.approx(-math.log(0.801))
    assert raw["Brier"] == pytest.approx(0.32)
    assert raw["ECE"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "change",
    [
        "state",
        "instruction",
        "labels",
        "target",
        "split",
        "cluster",
        "revision",
        "backend",
        "mode",
        "variant",
        "format",
        "group_size",
        "kwargs",
        "readout",
    ],
)
def test_resume_preflights_mismatches_before_any_reader_calls(tmp_path, change):
    original = item()
    output = tmp_path / "reads.jsonl"
    collect(iter([original]), FakeReader(), output, model="fixture", revision="a" * 40)
    requested = original
    reader = FakeReader()
    options = {"model": "fixture", "revision": "a" * 40}
    if change == "state":
        requested = replace(original, state="changed")
    elif change == "instruction":
        requested = replace(original, question=replace(original.question, instruction="changed"))
    elif change == "labels":
        requested = replace(
            original,
            question=replace(original.question, labels=["b", "a"], descriptions=["B", "A"]),
        )
    elif change == "target":
        requested = replace(original, gold_distribution={"a": 0.2, "b": 0.8})
    elif change == "split":
        requested = replace(original, split="dev")
    elif change == "cluster":
        requested = replace(original, cluster_id="changed")
    elif change == "revision":
        options["revision"] = "b" * 40
    elif change == "backend":
        reader.backend = "different"
    elif change == "mode":
        reader.logprobs_mode = "different"
    elif change == "variant":
        options["prompt_variant"] = "rules"
    elif change == "format":
        options["state_format"] = "compact"
    elif change == "group_size":
        options["group_size"] = 19
    elif change == "kwargs":
        reader.chat_template_kwargs = {"enable_thinking": True}
    elif change == "readout":
        reader.readout = "alias_sum"
    with pytest.raises(ValueError, match="cached read differs"):
        collect(iter([replace(original, id="unseen-first"), requested]), reader, output, **options)
    assert reader.calls == []
    assert len(load_reads([output])) == 1


def test_collect_refuses_duplicate_and_legacy_reads(tmp_path):
    reader = FakeReader()
    path = tmp_path / "reads.jsonl"
    with pytest.raises(ValueError, match="duplicate"):
        collect(iter([item(), item()]), reader, path)
    assert not reader.calls
    path.write_text(json.dumps({"id": "q", "raw_probs": {"a": 0.5, "b": 0.5}}) + "\n")
    with pytest.raises(ValueError, match="binding"):
        collect(iter([item()]), reader, path)
    assert not reader.calls


def test_adapters_preserve_source_lineage_split_and_public_group():
    canonical = {
        "metadata": {
            "split": "calibration",
            "source_lineage": "underlying-case",
            "case_facts_sha256": "a" * 64,
            "source_example_id": "english",
        },
        "questions": [
            {
                "type": "choice",
                "candidates": [{"id": "a"}, {"id": "b"}],
                "target_distribution": {"a": 1, "b": 0},
            }
        ],
    }
    first = adapt_canonical(canonical)[0]
    canonical["metadata"]["source_example_id"] = "korean"
    second = adapt_canonical(canonical)[0]
    assert first.id != second.id
    assert first.split == second.split == "calibration"
    assert first.cluster_id == second.cluster_id == "underlying-case"
    assert first.lineage_ids == ("a" * 64, "underlying-case")
    assert item(split="public", group="g").cluster_id == "g"
    assert item(split="public").cluster_id == "q"
    assert item(source="cygnet", group="g").cluster_id == "q"


@pytest.mark.parametrize("split", [None, "dev", "test", "public", "train", "private"])
def test_fitting_requires_explicit_calibration_and_diagnostic_is_not_promotable(split):
    rows = [{**row, "split": split} for row in reads()]
    with pytest.raises(ValueError, match="calibration only"):
        fit_policy(rows)
    policy = fit_policy(rows, exploratory=True, diagnostic=True)
    assert policy.promotable is False


def test_soft_target_counts_validation_and_underflow():
    row = {
        "raw_probs": {"a": 0.4, "b": 0.6},
        "gold": "b",
        "gold_distribution": {"a": 0.4, "b": 0.6},
    }
    assert nll([row], 1) == pytest.approx(-0.4 * math.log(0.4) - 0.6 * math.log(0.6))
    saturated = {
        "raw_probs": {"a": 1, "b": 0},
        "gold": "a",
        "gold_distribution": {"a": 0.5, "b": 0.5},
        "candidate_log_masses": {"a": 0, "b": -1000},
    }
    assert nll([saturated], 1) == 500
    assert nll([saturated], 2) == 250
    emitted = Policy(t_choice=10).apply(
        "choice", saturated["raw_probs"], candidate_log_masses=saturated["candidate_log_masses"]
    )
    assert emitted["b"] == pytest.approx(math.exp(-100), rel=1e-12, abs=0)
    assert math.isinf(nll([{**saturated, "candidate_log_masses": None}], 1))
    rows = reads()
    rows[0]["gold_distribution"] = {"a": 0.6, "b": 0.4}
    policy = fit_policy(rows, exploratory=True)
    assert policy.search["hard_n"] == 2 and policy.search["soft_n"] == 1
    for bad in ({"a": 0.5}, {"a": 1, "unknown": 0}, {"a": math.nan, "b": 1}, {"a": True, "b": 0}):
        with pytest.raises(ValueError):
            nll([{**row, "gold_distribution": bad}], 1)


def test_top21_fails_safely_and_canonical_ids_use_the_rendered_prefix():
    letters = list("ABCDEFGHIJKLMNOPQRSTU")
    with pytest.raises(ValueError, match="missing canonical.*U"):
        top_letter_probs([{"token": letter, "logprob": 0} for letter in letters[:-1]], letters)

    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return {"assistant:": [7], "assistant:A": [7, 42], "assistant:B": [7, 43]}[text]

        def __len__(self):
            raise AssertionError("canonical mode must not enumerate the vocabulary")

    assert canonical_letter_ids(Tokenizer(), "assistant:", ["A", "B"]) == {"A": [42], "B": [43]}


def test_grouped_reads_are_separate_from_single_pass_fit_and_evaluate(tmp_path):
    labels = [f"option-{i}" for i in range(27)]
    grouped_item = item(
        labels=labels, expected=labels[0], question={"type": "choice", "criteria": labels}
    )
    path = tmp_path / "grouped.jsonl"
    collect(iter([grouped_item]), FakeReader(), path)
    grouped = load_reads([path])[0]
    assert grouped["readout"] == "grouped_approx" and grouped["passes"] == 3
    assert len(grouped["pass_bindings"]) == grouped["passes"]
    assert [len(bound["letters"]) for bound in grouped["pass_bindings"]] == [14, 13, 2]
    assert len(grouped["binding"]["messages"]) == 2
    rows = [*reads(), grouped]
    result = evaluate(rows, Policy(), B=10)
    assert result["n"] == 3 and result["grouped_excluded_n"] == 1
    assert result["grouped_approx"]["n"] == 1
    assert result["I"] == score_reads(reads())["I"]
    fitted = fit_policy(rows, exploratory=True)
    assert fitted.search["grouped_excluded_n"] == 1
    assert fitted.search["hard_n"] == 3
    assert fitted.search["grouped_approx"]["n"] == 1


def test_canonical_score_ordinal_diagnostics_are_separate():
    record = {
        "questions": [
            {
                "type": "score",
                "candidates": [
                    {"id": "low", "ordinal": 0},
                    {"id": "mid", "ordinal": 1},
                    {"id": "high", "ordinal": 100},
                ],
                "target_distribution": {"mid": 1},
            }
        ]
    }
    adapted = adapt_canonical(record)[0]
    row = {
        "type": "score",
        "adapter": adapted.adapter,
        "labels": adapted.question.labels,
        "gold": adapted.gold,
        "gold_distribution": adapted.gold_distribution,
        "raw_probs": {"0": 0.99, "1": 0, "100": 0.01},
    }
    result = score_reads([row])
    assert result["per_type_CC"]["score"] == pytest.approx(-47)
    assert result["ordinal_value_diagnostics"]["nMAE"] == 0
    assert result["ordinal_value_diagnostics"]["CC"] == 100


def test_cluster_bootstrap_keeps_correlated_typed_and_language_rows_together():
    rows = []
    for cluster, gold in [("first", "true"), ("second", "false")]:
        for base in reads():
            count = 10 if base["type"] == "noul" else 1
            for i in range(count):
                rows.append(
                    {
                        **base,
                        "id": f"{cluster}/{base['type']}/{i}",
                        "cluster_id": cluster,
                        "source": f"language-{i % 2}",
                        "gold": gold if base["type"] == "noul" else base["gold"],
                    }
                )
    result = paired_bootstrap(rows, Policy(), Policy(commit_margin=0), B=500)
    assert result["unit"] == "cluster" and result["cluster_count"] == 2
    assert result["ci_95"] == pytest.approx([0, 200 / 3])
    cases, _ = cluster_strata([{**row, "public": True, "split": "public"} for row in rows])
    assert len(cases) == len(rows)


def test_parity_reports_error_thresholds_and_agreement(tmp_path, monkeypatch):
    same = compare([item(), item(id="q2")], FakeReader(), FakeReader(), n=2)
    assert same["passed"] and same["max_abs"] == same["mean_abs"] == 0
    assert same["argmax_agreement"] == 1
    failing = compare(
        [item()], FakeReader([{"A": 0.51, "B": 0.49}]), FakeReader([{"A": 0.48, "B": 0.52}]), n=1
    )
    assert failing["max_abs"] == pytest.approx(0.03)
    assert failing["mean_abs"] == pytest.approx(0.03)
    assert not failing["passed"] and failing["argmax_agreement"] == 0


@pytest.mark.parametrize(
    "left,right,max_abs,agreement",
    [
        ({"A": 0.7, "B": 0.3}, {"A": 0.67, "B": 0.33}, 0.03, 1),
        ({"A": 0.505, "B": 0.495}, {"A": 0.495, "B": 0.505}, 0.01, 0),
    ],
)
def test_parity_enforces_both_thresholds(left, right, max_abs, agreement):
    result = compare([item()], FakeReader([left]), FakeReader([right]), n=1)
    assert result["max_abs"] == pytest.approx(max_abs)
    assert result["argmax_agreement"] == agreement
    assert not result["passed"]
