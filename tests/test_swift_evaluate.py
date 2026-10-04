import json

import pytest

from ayaka.swift.evaluate import case_bootstrap_composite, evaluate, main, paired_bootstrap
from ayaka.swift.policy import Policy
from ayaka.swift.score import composite, score_reads


def reads():
    rows = []
    for i in range(2):
        for kind, probs, gold in [
            ("choice", {"a": 0.9, "b": 0.1}, "a"),
            ("noul", {"false": 0.4, "true": 0.6}, "true"),
            ("score", {"0": 0.1, "1": 0.9}, "1"),
        ]:
            rows.append(
                {
                    "id": f"{kind}-{i}",
                    "type": kind,
                    "tier": "standard",
                    "labels": list(probs),
                    "raw_probs": probs,
                    "gold": gold,
                    "input_tokens": 100,
                    "output_tokens": 1,
                }
            )
    return rows


def test_ablations_and_paired_bootstrap():
    result = evaluate(reads(), Policy(t_choice=2, commit_margin=0.0), B=20)
    assert result["ablation"]["raw"]["I"] == pytest.approx(80 / 3)
    assert result["ablation"]["temps_only"]["I"] == pytest.approx(80 / 3)
    assert result["ablation"]["temps_commit"]["I"] == pytest.approx(280 / 3)
    assert result["ablation"]["fitted_policy"]["I"] == result["I"]
    assert result["paired_bootstrap"]["delta_I"] == pytest.approx(200 / 3)
    assert result["paired_bootstrap"]["ci_95"] == pytest.approx([200 / 3, 200 / 3])
    assert result["paired_bootstrap"]["seed"] == 15
    assert result["paired_bootstrap"]["B"] == 20
    # The selected policy may disable commit while the ablation still enables it.
    disabled = evaluate(reads(), Policy(noul_commit=False), B=10)
    assert disabled["I"] == pytest.approx(80 / 3)
    assert disabled["paired_bootstrap"]["delta_I"] == 0
    assert disabled["ablation"]["temps_commit"]["I"] == pytest.approx(280 / 3)


def test_fitted_margin_differs_from_unconditional_commit_ablation():
    result = evaluate(reads(), Policy(commit_margin=0.15), B=10)
    assert list(result["ablation"]) == ["raw", "temps_only", "temps_commit", "fitted_policy"]
    assert result["ablation"]["fitted_policy"]["I"] == result["ablation"]["temps_only"]["I"]
    assert result["ablation"]["temps_commit"]["I"] > result["I"]


def test_temperature_ablations_use_nll_fit_before_joint_search():
    rows = reads()
    for item in rows:
        if item["type"] == "noul":
            item["raw_probs"] = {"false": 0.1, "true": 0.9}
    policy = Policy(
        t_noul=1,
        commit_margin=None,
        search={"nll_temperatures": {"t_choice": 1, "t_noul": 2, "t_score": 1}},
    )
    ablation = evaluate(rows, policy, B=10)["ablation"]
    assert ablation["fitted_policy"]["I"] == ablation["raw"]["I"]
    assert ablation["temps_only"]["I"] < ablation["fitted_policy"]["I"]
    assert ablation["temps_commit"]["I"] == ablation["fitted_policy"]["I"]
    assert ablation["temps_commit"]["C"] != ablation["fitted_policy"]["C"]


def test_paired_bootstrap_stratified_pairs_and_default_count():
    rows = reads()
    rows[1]["gold"] = "false"
    before = Policy(noul_commit=False)
    after = Policy(commit_margin=0.0)
    first = paired_bootstrap(rows, before, after)
    assert first == paired_bootstrap(rows, before, after)
    assert first["B"] == 2000
    assert first["delta_I"] == pytest.approx(100 / 3)
    assert first["ci_95"] == pytest.approx([0, 200 / 3])


def test_cli_json_composites_and_ablation_table(tmp_path, capsys):
    path = tmp_path / "reads.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in reads()), encoding="utf-8")
    policy_path = tmp_path / "policy.json"
    Policy().save(policy_path)
    output = tmp_path / "evaluation.json"
    main(
        [
            str(path),
            "--policy",
            str(policy_path),
            "--out",
            str(output),
            "--p50",
            "0.039807",
            "--p95",
            "0.099066",
            "--usd-in-per-m",
            "0.1",
            "--usd-out-per-m",
            "0.5",
            "--bootstrap",
            "10",
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["S"] == pytest.approx(90.9726, abs=1e-3)
    assert report["usd_per_1000_decisions"] == pytest.approx(0.0105)
    assert report["composite_A"] > 0
    assert report["composite_B"] > 0
    assert report["choice_ece_scope"] == "all_choice_fallback"
    text = capsys.readouterr().out
    assert "Ablation" in text and "temps_commit (margin 0)" in text and "fitted_policy" in text


def test_evaluation_ablations_and_composite_bootstrap_use_hard_choice_ece():
    rows = reads()
    rows[0]["tier"] = "hard"
    rows[0]["raw_probs"] = {"a": 0.8, "b": 0.2}
    rows[0]["gold_distribution"] = {"a": 0.7, "b": 0.3}
    rows[3]["gold"] = "b"
    rows[3]["raw_probs"] = {"a": 1, "b": 0}
    policy = Policy(t_choice=2, commit_margin=None)
    report = evaluate(rows, policy, p50=0.04, p95=0.1, usd_in_per_m=0.1, usd_out_per_m=0.5, B=10)
    for result in [report, *report["ablation"].values()]:
        assert result["choice_ece_scope"] == "hard"
        detail = result["calibration_details"]["choice"]
        assert detail["n"] == 2 and detail["n_ece_hard"] == detail["n_ece"] == 1
        assert result["composite_A"] == pytest.approx(
            composite(result["I"], result["C"], result["S"], result["Cost"])
        )
    # sqrt(.8)/(sqrt(.8)+sqrt(.2))=.6667, so hard ECE=1/3 and TVD=1/30.
    choice = report["calibration_details"]["choice"]
    assert choice["ECE"] == pytest.approx(1 / 3)
    assert choice["mean_TVD"] == pytest.approx(1 / 30)
    bootstrap = case_bootstrap_composite(rows, policy, speed=report["S"], cost=report["Cost"], B=20)
    assert bootstrap["choice_ece_scope"] == "hard"
    assert bootstrap["composite_A"] == pytest.approx(report["composite_A"])


def test_missing_arguments_and_partial_data():
    with pytest.raises(ValueError, match="both --p50"):
        evaluate(reads(), Policy(), p50=0.1)
    with pytest.raises(ValueError, match="both input"):
        evaluate(reads(), Policy(), usd_in_per_m=1)
    partial = evaluate(reads()[:1], Policy())
    assert partial["paired_bootstrap"] is None


@pytest.mark.parametrize("explicit_case_id", [False, True])
def test_composite_bootstrap_resamples_whole_cases(explicit_case_id):
    rows = reads()
    for index, item in enumerate(rows):
        case_id = str(index // 3)
        item["id"] = f"case-{case_id}/{item['type']}"
        item["cluster_id"] = case_id
        if explicit_case_id:
            item["id"] = str(index)
            item["case_id"] = case_id
        if index >= 3:
            item["gold"] = item["labels"][0]
    policy = Policy(commit_margin=0.0)

    def objective(sample):
        result = score_reads(sample, policy)
        return composite(result["I"], result["C"], 91.0, 56.4)

    # Two correlated cases admit only all-good, mixed, or all-bad case draws.
    possible = [objective(rows[:3] * 2), objective(rows), objective(rows[3:] * 2)]
    report = case_bootstrap_composite(rows, policy, speed=91, cost=56.4, B=1000)
    assert report["choice_ece_scope"] == "all_choice_fallback"
    assert report["case_count"] == 2
    assert report["composite_A"] == objective(rows)
    assert report["ci_95"] == pytest.approx([min(possible), max(possible)])
    assert report == case_bootstrap_composite(rows, policy, speed=91, cost=56.4, B=1000)


def test_case_bootstrap_rejects_invalid_count_and_missing_types():
    with pytest.raises(ValueError, match="positive"):
        case_bootstrap_composite(reads(), Policy(), speed=91, cost=56.4, B=0)
    with pytest.raises(ValueError, match="choice, noul and score"):
        case_bootstrap_composite(reads()[:1], Policy(), speed=91, cost=56.4)
