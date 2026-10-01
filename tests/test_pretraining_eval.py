from ayaka.data.multimodal_v2 import image_curriculum
from ayaka.eval.pretraining_v2 import evaluate_tracks
from ayaka.multimodal import decode_media
from ayaka.primitives import DecisionResult


def test_evaluation_tracks_pass_only_evidence_and_preserve_pairs_and_budgets():
    samples = image_curriculum("dev", 1)

    class Decision:
        from ayaka.tokenization import ToyTokenizer

        tok = ToyTokenizer()
        calls = []

        def prepare_media(self, state, media):
            return decode_media(state, media)

        def decide(self, state, questions, reasoning):
            self.calls.append((state, questions, reasoning))
            q, setting = questions[0], reasoning[0]
            # No oracle annotations or labels are exposed through state/spec.
            assert "oracle" not in str(state) and "target_distribution" not in str(state)
            probabilities = [1 / len(q.candidates)] * len(q.candidates)
            return [
                DecisionResult(
                    q.type,
                    probabilities,
                    {},
                    extras={
                        "reasoning": {
                            "route": "direct" if setting.mode == "off" else "reasoned",
                            "generated_tokens": 0 if setting.mode == "off" else 1,
                        }
                    },
                )
            ]

    decision = Decision()
    report = evaluate_tracks(decision, samples, modes=("off", "high"))
    assert report["complete"] and report["official_composite"] is None
    assert report["image_text_pairs"]["off"]["n"] == 12
    assert report["image_text_pairs"]["off"]["mean_image_minus_text_nll"] == 0
    assert all(
        settings[0].budget == 1024 for _, _, settings in decision.calls if settings[0].mode == "on"
    )
    assert all(row["split"] == "dev" for row in report["rows"]["high"])


def test_cli_hard_deadline_encloses_model_loading_and_report_writing(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from ayaka import multimodal
    from ayaka.eval import pretraining_v2
    from ayaka.training import run_v2, scoped_calibration

    active = []
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "report.json"

    @contextmanager
    def deadline(seconds):
        assert seconds == 30
        active.append(True)
        yield
        assert out.is_file()
        active.pop()

    def load(*args, **kwargs):
        assert active
        return object()

    monkeypatch.setattr(run_v2, "job_deadline", deadline)
    monkeypatch.setattr(run_v2, "source_matches", lambda manifest: True)
    monkeypatch.setattr(multimodal, "load_image_decision", load)
    monkeypatch.setattr(scoped_calibration, "checkpoint_fingerprint", lambda path: "a" * 64)
    monkeypatch.setattr(
        pretraining_v2,
        "validate_bundle",
        lambda path: ({"code_revision": "pinned"}, {"dev": []}),
    )
    monkeypatch.setattr(
        pretraining_v2,
        "evaluate_tracks",
        lambda *args, **kwargs: {"complete": True, "rows": {}},
    )
    result = pretraining_v2.main(
        [
            "--bundle",
            str(bundle),
            "--checkpoint",
            "unused",
            "--out",
            str(out),
            "--max-evaluation-seconds",
            "30",
        ]
    )
    assert result["complete"] and not active
