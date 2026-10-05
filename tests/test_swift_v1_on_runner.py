import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "v1_on_runner", Path(__file__).resolve().parents[1] / "scripts/swift/v1_on_runner.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

from ayaka.swift.collect import iter_dataset  # noqa: E402


@dataclass
class Result:
    probs: list
    extras: dict = field(default_factory=dict)


class Inner:
    tok = "tok"

    def decide(self, state, specs, device=None):
        n = len(specs[0].candidates)
        return [Result([1 / n] * n)]


class Route:
    """Calls the single pass first, then returns different routed probabilities."""

    def __init__(self):
        self.original = Inner()

    def decide(self, state, specs, device=None):
        base = self.original.decide(state, specs, device=device)[0]
        n = len(base.probs)
        probs = [0.0] * (n - 1) + [1.0]
        return [Result(probs, {"evidence": {"route": "reasoned", "worked_steps": "abc"}})]


def dataset(tmp_path):
    rows = [
        {
            "id": "doc/1",
            "state": "Order placed on 3 May; shipped 9 May.",
            "split": "test",
            "tier": "standard",
            "metadata": {"split": "test", "source": "synth", "source_lineage": "c1"},
            "questions": [
                {
                    "id": "q",
                    "type": "noul",
                    "instruction": "Shipped within 7 days?",
                    "candidates": [
                        {"id": "true", "description": "yes"},
                        {"id": "false", "description": "no"},
                    ],
                    "target_distribution": {"true": 1.0, "false": 0.0},
                }
            ],
        }
    ]
    path = tmp_path / "cohort.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return list(iter_dataset([path]))


def test_records_route_and_single_pass_baseline_and_resumes(tmp_path):
    items = dataset(tmp_path)
    out = tmp_path / "v1.jsonl"
    assert runner.run_rows(Route(), items, out) == 1
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["labels"] == ["false", "true"]
    assert row["raw_probs"] == {"false": 0.0, "true": 1.0}
    assert row["baseline_probs"] == {"false": 0.5, "true": 0.5}
    assert row["route"] == "reasoned" and row["worked_steps_chars"] == 3
    assert row["gold"] == "true" and row["public"] is False
    assert runner.run_rows(Route(), items, out) == 0
