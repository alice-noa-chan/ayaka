import json

import pytest

from scripts.swift import matched_2x2


def cohort(n=4):
    return {
        str(i): {
            "tier": "hard" if i % 2 else "easy",
            "type": "choice",
            "family": "rules" if i % 2 else "fact",
            "expected": "a",
            "labels": ["a", "b"],
        }
        for i in range(n)
    }


def cell(values):
    return {str(i): {"correct": bool(v)} for i, v in enumerate(values)}


def write_rows(path, values):
    path.write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")


def test_marginals_interaction_and_paired_family_counts():
    cells = dict(
        zip(
            matched_2x2.CELLS,
            (
                cell([1, 0, 0, 0]),
                cell([1, 1, 0, 0]),
                cell([0, 1, 1, 0]),
                cell([1, 1, 1, 1]),
            ),
            strict=True,
        )
    )
    report = matched_2x2.analyze(cells, cohort())
    overall = report["groups"]["all"]
    assert overall["effects"] == pytest.approx(
        {
            "weights_native": 0.25,
            "weights_swift": 0.5,
            "format_readout_frozen": 0.25,
            "format_readout_lora": 0.5,
            "weights_marginal": 0.375,
            "format_readout_marginal": 0.375,
            "interaction": 0.25,
        }
    )
    assert [v["correct"] for v in overall["accuracy"].values()] == [1, 2, 2, 4]
    assert "tier_type/hard/choice" in report["groups"]
    assert report["groups"]["tier/hard"]["effects"]["interaction"] == 0
    weights = overall["contrasts"]["weights_native"]
    assert (weights["left_only"], weights["right_only"]) == (1, 2)
    assert weights["by_family"]["fact"]["left_only"] == 1
    assert weights["by_family"]["fact"]["right_only"] == 1
    assert weights["by_family"]["rules"]["right_only"] == 1
    assert weights["discordant_ids"] == {"left_only": ["0"], "right_only": ["1", "2"]}
    assert report["used_for_fit_or_selection"] is False and report["used_for_gates"] is False
    assert "selects nothing" in matched_2x2.markdown(report)


@pytest.mark.parametrize("a,b,expected", [(0, 0, 1), (0, 5, 1 / 16), (1, 7, 18 / 256), (3, 3, 1)])
def test_exact_mcnemar_known_tables(a, b, expected):
    assert matched_2x2.mcnemar_exact(a, b) == expected
    assert matched_2x2.mcnemar_exact(b, a) == expected


def test_reads_native_and_cygnet_normalize_noul_and_match_ids(tmp_path):
    items = cohort(1)
    items["0"].update(type="noul", expected="yes", labels=["no", "yes"])
    swift = tmp_path / "swift.jsonl"
    native = tmp_path / "native.json"
    cygnet = tmp_path / "cygnet.jsonl"
    write_rows(swift, [{"id": "0", "gold": "true", "raw_probs": {"false": 0.1, "true": 0.9}}])
    native.write_text(
        json.dumps(
            {
                "tiers": {
                    "easy": {
                        "results": [
                            {
                                "id": "0",
                                "expected": "yes",
                                "pred": "yes",
                                "ok": True,
                                "probs": {"no": 0.1, "yes": 0.9},
                            },
                        ]
                    }
                }
            }
        )
    )
    # Published `ok` means transport success and must not override `correct`.
    write_rows(cygnet, [{"task_id": "0", "predicted": "no", "correct": False, "ok": True}])
    assert matched_2x2.load_cell(swift, items)["0"]["correct"] is True
    assert matched_2x2.load_cell(native, items)["0"]["correct"] is True
    assert matched_2x2.load_cell(cygnet, items)["0"]["correct"] is False


@pytest.mark.parametrize("bad", ["missing", "extra", "duplicate", "gold", "probabilities"])
def test_bad_per_item_file_refused(tmp_path, bad):
    items = cohort(2)
    rows = [{"id": str(i), "pred": "a", "expected": "a"} for i in range(2)]
    if bad == "missing":
        rows.pop()
    elif bad == "extra":
        rows.append({"id": "extra", "pred": "a"})
    elif bad == "duplicate":
        rows.append(rows[0])
    elif bad == "gold":
        rows[0]["expected"] = "b"
    else:
        rows[0]["probs"] = {"a": float("nan"), "b": 0.5}
    path = tmp_path / "rows.jsonl"
    write_rows(path, rows)
    with pytest.raises(ValueError):
        matched_2x2.load_cell(path, items)


def test_analysis_refuses_id_intersection():
    cells = {name: cell([1, 0, 1, 0]) for name in matched_2x2.CELLS}
    del cells["lora_swift"]["3"]
    with pytest.raises(ValueError, match="id-set mismatch"):
        matched_2x2.analyze(cells, cohort())


def test_mixed_swift_variants_refused(tmp_path):
    path = tmp_path / "swift.jsonl"
    write_rows(
        path,
        [
            {"id": str(i), "raw_probs": {"a": 0.75, "b": 0.25}, "prompt_variant": variant}
            for i, variant in enumerate(("min", "cygnet"))
        ],
    )
    with pytest.raises(ValueError, match="mixed Swift recipes"):
        matched_2x2.load_cell(path, cohort(2))


def test_different_swift_factor_variants_refused():
    cells = {name: cell([1, 0, 1, 0]) for name in matched_2x2.CELLS}
    for name, variant in (("frozen_swift", "min"), ("lora_swift", "cygnet")):
        for row in cells[name].values():
            row["prompt_variant"] = variant
    with pytest.raises(ValueError, match="Swift cells differ in prompt_variant"):
        matched_2x2.analyze(cells, cohort())


def test_cli_synthetic_files_produce_tables_and_input_hashes(tmp_path):
    directory = tmp_path / "public"
    directory.mkdir()
    inputs = []
    for tier, n in (("easy", 48), ("original", 72), ("hard", 111)):
        rows = [
            {
                "id": f"{tier}-{i}",
                "labels": ["a", "b"],
                "expected": "a",
                "family": "synthetic",
                "question": {"type": "choice"},
            }
            for i in range(n)
        ]
        write_rows(directory / (tier + ".jsonl"), rows)
        inputs += rows
    arguments = ["--public-data-dir", str(directory), "--output", str(tmp_path / "report.json")]
    for index, name in enumerate((*matched_2x2.CELLS, "cygnet_reference", "historical_v1")):
        path = tmp_path / (name + ".jsonl")
        write_rows(
            path,
            [
                {"id": row["id"], "pred": "a" if i % 4 <= index else "b"}
                for i, row in enumerate(inputs)
            ],
        )
        arguments += ["--" + name.replace("_", "-"), str(path)]
    report = matched_2x2.main(arguments)
    assert report["groups"]["all"]["n"] == 231
    assert report["groups"]["tier/hard"]["n"] == 111
    assert report["groups"]["all"]["accuracy"]["lora_swift"]["correct"] == 231
    assert all(len(row["sha256"]) == 64 for row in report["inputs"].values())
    assert len(report["public_ids_sha256"]) == 64
    assert json.loads((tmp_path / "report.json").read_text())["role"] == "diagnostic"
    assert "cygnet_reference | historical_v1" in (tmp_path / "report.md").read_text()


def test_reference_columns_and_continuity_discordance():
    items = cohort()
    cells = {name: cell([1, 0, 1, 0]) for name in matched_2x2.CELLS}
    cells["cygnet_reference"] = cell([1, 1, 1, 0])
    cells["historical_v1"] = cell([0, 0, 1, 1])
    report = matched_2x2.analyze(cells, items)
    row = report["groups"]["all"]["contrasts"]["historical_v1_to_cygnet"]
    assert (row["left_only"], row["right_only"]) == (1, 2)
    table = matched_2x2.markdown(report)
    assert "lora_swift | cygnet_reference | historical_v1" in table
    assert "McNemar does not apply" in report["limitations"]
