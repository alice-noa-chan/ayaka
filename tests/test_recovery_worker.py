from ayaka.data.language_v2 import language_curriculum
from ayaka.data.recovery_v2 import recovery_curriculum
from scripts.runpod_v2.recovery import selection


def test_worker_selection_is_deterministic_and_keeps_rich_translations_together():
    old = language_curriculum("dev", 32)
    for sample in old:
        sample.metadata["source_lineage"] = sample.metadata["source_example_id"]
    samples = old + recovery_curriculum("dev", 4)
    selected = selection(samples, 32, 2)
    repeated = selection(samples, 32, 2)
    assert [s.metadata["source_example_id"] for s in selected] == [
        s.metadata["source_example_id"] for s in repeated
    ]
    old_selection = [
        s for s in selected if not s.metadata["source_example_id"].startswith("recovery-1/")
    ]
    assert len(old_selection) == 96
    assert len({s.metadata["source_lineage"] for s in old_selection}) == 96
    rich = [s for s in selected if s.metadata["source_example_id"].startswith("recovery-1/")]
    assert len(rich) == 6 and sum(len(s.questions) for s in rich) == 18
    assert {s.metadata["oracle_facts"]["index"] for s in rich} == {0, 1}


def test_failed_primary_screen_never_opens_fresh_final_test(tmp_path, monkeypatch):
    from scripts.runpod_v2 import recovery

    calls = []
    monkeypatch.setattr(recovery, "validate_bundle", lambda _: ({}, {"dev": [], "calibration": []}))
    monkeypatch.setattr(recovery, "selection", lambda *a: [])
    monkeypatch.setattr(
        recovery, "measure", lambda checkpoint, name, *a, **kw: (calls.append(name) or {}, {})
    )
    monkeypatch.setattr(recovery, "promotion_screen", lambda *a: {"screen_passed": False})
    from ayaka.training import run_v2

    def train(args):
        assert args[args.index("--steps") + 1] == "200"
        assert "--init-checkpoint" in args
        calls.append("fixed-200-step-pilot")

    monkeypatch.setattr(run_v2, "main", train)
    result = recovery.main(["--kit", str(tmp_path), "--out", str(tmp_path / "out")])
    assert calls == ["v1", "current-v2", "fixed-200-step-pilot", "pilot"]
    assert result["complete"] and not result["test_evaluated"] and not result["release_promoted"]
