import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "hard_dev_gate", Path(__file__).resolve().parents[1] / "scripts/swift/hard_dev_gate.py"
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def read(identifier, public=False):
    return {"id": identifier, "public": public, "readout": "canonical_letter_raw"}


@pytest.fixture
def dirs(tmp_path):
    hard = tmp_path / "data/dev.jsonl"
    write(hard, [{"id": "doc/1", "questions": [{"id": "q"}, {"id": "r"}]}])
    write(
        tmp_path / "v2/min/v2_dev.reads.jsonl",
        [read("v2/a"), {**read("g"), "readout": "grouped_approx"}],
    )
    return tmp_path, gate.hard_ids(hard)


def test_hard_reads_must_cover_the_frozen_file_exactly_once(dirs):
    root, expected = dirs
    assert expected == {"doc/1/q", "doc/1/r"}
    write(root / "hard/min/hard_dev.reads.jsonl", [read("doc/1/q"), read("doc/1/r")])
    v2, hard = gate.load_variant(root / "v2", root / "hard", "min", "dev", expected)
    assert [r["id"] for r in v2] == ["v2/a"] and len(hard) == 2
    for rows in ([read("doc/1/q")], [read("doc/1/q"), read("doc/1/q"), read("doc/1/r")]):
        write(root / "hard/min/hard_dev.reads.jsonl", rows)
        with pytest.raises(ValueError, match="exactly once"):
            gate.load_variant(root / "v2", root / "hard", "min", "dev", expected)


def test_public_reads_are_refused(dirs):
    root, expected = dirs
    write(root / "hard/min/hard_dev.reads.jsonl", [read("doc/1/q", True), read("doc/1/r")])
    with pytest.raises(ValueError, match="public"):
        gate.load_variant(root / "v2", root / "hard", "min", "dev", expected)


def test_misaligned_variants_are_refused():
    with pytest.raises(ValueError, match="aligned"):
        gate.aligned([read("a"), read("b")], [read("b"), read("a")])


def test_duplicate_hard_row_ids_are_refused(tmp_path):
    path = tmp_path / "dev.jsonl"
    write(path, [{"id": "doc/1", "questions": [{"id": "q"}]}] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        gate.hard_ids(path)
