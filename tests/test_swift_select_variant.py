import json
from dataclasses import replace

import pytest

from ayaka.swift.collect import adapt_jevbench, collect, load_reads
from ayaka.swift.policy import Policy
from ayaka.swift.prompt import PROMPT_VARIANTS
from ayaka.swift.readers import FakeReader, ReadResult
from ayaka.swift.score import composite, cost_score, speed_axis
from scripts.swift.select_variant import (
    choose_variant,
    group_reads,
    main,
    paired_case_bootstrap,
    select_variants,
)


def fixture_reads(tmp_path):
    cal, dev = [], []
    for split in ("calibration", "dev"):
        items = []
        for case in range(4):
            for kind, labels in (
                ("choice", ["a", "b"]),
                ("noul", ["no", "yes"]),
                ("score", ["0", "1", "2"]),
            ):
                items.append(
                    adapt_jevbench(
                        {
                            "id": f"{split}-{case}/{kind}",
                            "case_id": f"{split}-{case}",
                            "source": "private-v2"
                            if split == "calibration" or case < 2
                            else "private-secondary",
                            "public": False,
                            "split": split,
                            "tier": "hard",
                            "state": f"private fixture {split}-{case}",
                            "question": {
                                "type": kind,
                                "criteria": {label: label for label in labels},
                            },
                            "labels": labels,
                            "expected": labels[-1],
                        }
                    )
                )
        for variant in PROMPT_VARIANTS:

            def read(messages, letters):
                tokens = sum(len(message["content"].split()) for message in messages)
                probs = {letter: 0.1 / (len(letters) - 1) for letter in letters}
                probs[letters[-1]] = 0.9
                return ReadResult(probs, tokens, 1, 0.01)

            path = tmp_path / f"{split}.{variant}.jsonl"
            reader = FakeReader(read)
            reader.backend = "hf"
            reader.logprobs_mode = "raw_logits"
            collect(
                iter(items),
                reader,
                path,
                model="fixture",
                revision="a" * 40,
                prompt_variant=variant,
            )
            (cal if split == "calibration" else dev).extend(load_reads([path]))
    return cal, dev


def latency(variant, seconds=0.04):
    return {
        "prompt_variant": variant,
        "model": "fixture",
        "revision": "a" * 40,
        "complete": True,
        "concurrency": 1,
        "units": "seconds",
        "completed_reads": 200,
        "requested_reads": 200,
        "p50_s": seconds,
        "p95_s": seconds * 2,
        "seed": 15,
        "samples": [{"id": str(i)} for i in range(200)],
    }


@pytest.mark.parametrize(
    "ci,selected,fallback",
    [
        ([0.1, 1.0], "rules", False),
        ([-1.0, 1.0], "min", True),
        ([0.0, 1.0], "min", True),
        ([-1.0, 0.0], "min", True),
    ],
)
def test_selection_highest_A_and_token_fallback(ci, selected, fallback):
    reports = {
        "min": {"composite_A": 70, "mean_input_tokens": 600},
        "rules": {"composite_A": 71, "mean_input_tokens": 650},
        "cygnet": {"composite_A": 69, "mean_input_tokens": 580},
    }
    selection = choose_variant(reports, ci)
    assert selection["selected"] == selected
    assert selection["token_fallback"] is fallback
    assert selection["highest_A"] == "rules"
    assert selection["runner_up"] == "min"
    reports["min"]["mean_input_tokens"] = 650
    assert choose_variant(reports, [-1, 1])["selected"] == "rules"


def test_select_fits_calibration_reports_axes_and_writes_policies(tmp_path, capsys):
    cal, dev = fixture_reads(tmp_path)
    result = select_variants(cal, list(reversed(dev)), B=30)
    assert result["selection"]["selected"] == "min"
    assert result["calibration_n_per_variant"] == result["dev_n_per_variant"] == 12
    for variant, report in result["variants"].items():
        assert result["policies"][variant]["prompt_variant"] == variant
        assert "NLL n=12" in result["policies"][variant]["fitted_on"]
        assert report["Cost"] == pytest.approx(
            cost_score(report["mean_input_tokens"] * 0.0403 / 1000)
        )
        assert report["composite_A"] == composite(report["I"], report["C"], 91, report["Cost"])
        assert report["speed_source"] == "estimated_speed_axis"
    assert result["variants"]["min"]["paired_vs_min"]["ci_95_A"] == [0, 0]
    assert result["bootstrap"]["B"] == 30 and result["bootstrap"]["seed"] == 15
    assert result["bootstrap"]["case_count"] == 4
    output = tmp_path / "selection.json"
    policies = tmp_path / "policies"
    main(
        [
            "--calibration",
            *[str(tmp_path / f"calibration.{v}.jsonl") for v in PROMPT_VARIANTS],
            "--dev",
            *[str(tmp_path / f"dev.{v}.jsonl") for v in PROMPT_VARIANTS],
            "--output",
            str(output),
            "--policy-dir",
            str(policies),
            "--bootstrap",
            "30",
        ]
    )
    assert json.loads(output.read_text()) == result
    assert Policy.load(policies / "policy.json").prompt_variant == "min"
    assert "Selected min" in capsys.readouterr().out


def test_paired_case_draws_keep_questions_together_and_use_same_draws(tmp_path):
    _, rows = fixture_reads(tmp_path)
    dev = group_reads(rows, "dev")
    for variant in dev:
        dev[variant] = [dict(row, input_tokens=600) for row in dev[variant]]
    policies = {v: Policy(prompt_variant=v, commit_margin=None) for v in dev}
    result = paired_case_bootstrap(dev, policies, dict.fromkeys(dev, 91))
    assert result["B"] == 2000 and result["seed"] == 15
    assert result["unit"] == "case" and result["case_count"] == 4
    for delta in result["vs_min"].values():
        assert delta == {"delta_A": 0, "delta_I": 0, "ci_95_A": [0, 0], "ci_95_I": [0, 0]}
    dev["rules"] = [
        dict(
            row,
            raw_probs={label: 1 / len(row["labels"]) for label in row["labels"]},
            candidate_log_masses=dict.fromkeys(row["labels"], 0.0),
        )
        for row in dev["rules"]
    ]
    changed = paired_case_bootstrap(dev, policies, dict.fromkeys(dev, 91), B=30)
    assert changed["vs_min"]["rules"]["delta_I"] < 0
    assert changed["vs_min"]["rules"]["ci_95_I"][1] < 0


def test_select_uses_variant_latency_and_input_only_cost(tmp_path):
    cal, dev = fixture_reads(tmp_path)
    probes = [latency(v, (i + 1) * 0.03) for i, v in enumerate(PROMPT_VARIANTS)]
    result = select_variants(cal, dev, latency=probes, usd_in_per_m=0.08, B=10)
    for p in probes:
        report = result["variants"][p["prompt_variant"]]
        assert report["S"] == speed_axis(p["p50_s"], p["p95_s"])
        assert report["speed_source"] == "serial_latency_probe"
        assert report["usd_per_1000_decisions"] == report["mean_input_tokens"] * 0.08 / 1000


def test_case_bootstrap_retains_correlated_question_errors():
    # Each of two cases contains all three primitives. Only the first case
    # improves, so whole-case draws must retain the 0 and 200 I-delta endpoints.
    baseline, candidate = [], []
    for case in range(2):
        for kind, labels in (
            ("choice", ["a", "b"]),
            ("noul", ["false", "true"]),
            ("score", ["0", "1"]),
        ):
            row = {
                "id": f"case-{case}/{kind}",
                "case_id": f"case-{case}",
                "source": "private",
                "type": kind,
                "tier": "hard",
                "labels": labels,
                "gold": labels[0],
                "raw_probs": {labels[0]: 0, labels[1]: 1},
                "input_tokens": 600,
            }
            baseline.append(row)
            candidate.append(
                dict(row, raw_probs={labels[0]: 1, labels[1]: 0}) if case == 0 else row.copy()
            )
    variants = {"min": baseline, "rules": candidate}
    policies = {v: Policy(prompt_variant=v, commit_margin=None) for v in variants}
    result = paired_case_bootstrap(variants, policies, dict.fromkeys(variants, 91))
    delta = result["vs_min"]["rules"]
    assert result["case_count"] == 2
    assert delta["delta_I"] == 100
    assert delta["ci_95_I"] == [0, 200]


@pytest.mark.parametrize(
    "field,value",
    [
        ("public", True),
        ("public", None),
        ("source", "data/jevbench_public/hard.jsonl"),
        ("split", "public"),
    ],
)
def test_selector_refuses_public_on_either_split(tmp_path, field, value):
    cal, dev = fixture_reads(tmp_path)
    for rows in (cal, dev):
        before = rows[0].copy()
        rows[0][field] = value
        with pytest.raises(ValueError, match="REFUSING public"):
            select_variants(cal, dev, B=1)
        rows[0] = before


@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "gold", "type", "case", "revision", "overlap"]
)
def test_selector_refuses_unpaired_or_contaminated_reads(tmp_path, problem):
    cal, dev = fixture_reads(tmp_path)
    if problem == "missing":
        dev.pop()
    elif problem == "duplicate":
        dev.append(dev[0].copy())
    elif problem == "gold":
        dev[-1]["gold"] = "0"
    elif problem == "type":
        dev[-1]["type"] = "choice"
    elif problem == "case":
        dev[-1]["case_id"] = "different"
    elif problem == "revision":
        dev[-1]["revision"] = "b" * 40
    elif problem == "overlap":
        dev = cal
    with pytest.raises(ValueError):
        select_variants(cal, dev, B=1)


@pytest.mark.parametrize(
    "problem", ["partial", "missing", "revision", "variant", "sample", "adjusted"]
)
def test_selector_refuses_bad_latency(tmp_path, problem):
    cal, dev = fixture_reads(tmp_path)
    probes = [latency(v) for v in PROMPT_VARIANTS]
    if problem == "partial":
        probes[0]["complete"] = False
    elif problem == "missing":
        probes.pop()
    elif problem == "revision":
        probes[0]["revision"] = "b" * 40
    elif problem == "variant":
        probes[0]["prompt_variant"] = "unknown"
    elif problem == "sample":
        probes[0]["samples"].reverse()
    elif problem == "adjusted":
        probes[0]["self_hosted_adjustment"] = {"applied": True}
    with pytest.raises(ValueError):
        select_variants(cal, dev, latency=probes, B=1)


def test_case_metadata_survives_collection(tmp_path):
    from ayaka.swift.collect import adapt_canonical

    record = {
        "id": "case",
        "metadata": {"case_id": "explicit-case"},
        "questions": [
            {
                "id": "q",
                "type": "choice",
                "candidates": [
                    {"id": "a"},
                    {"id": "b"},
                ],
                "target_distribution": {"a": 1, "b": 0},
            }
        ],
    }
    items = adapt_canonical(record)
    assert items[0].case_id == "explicit-case"
    output = tmp_path / "case.jsonl"
    collect(iter([replace(items[0], case_id="original-case")]), FakeReader(), output)
    assert load_reads([output])[0]["case_id"] == "original-case"
