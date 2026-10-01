import pytest
from test_routing_v2 import rows

from ayaka.multimodal import ImageState
from ayaka.primitives import DecisionResult, QuestionSpec
from ayaka.routing import fit_router
from ayaka.tokenization import ToyTokenizer
from ayaka.training.scoped_router import ScopedRouter


def test_scoped_image_router_uses_stable_text_features_and_exact_binding(tmp_path):
    router = ScopedRouter("a" * 64, "image", fit_router(rows("router_train"), rows("dev")))
    baseline, spec = (
        DecisionResult("choice", [0.8, 0.2], {}),
        QuestionSpec("choice", "Type?", ["a", "b"]),
    )
    # Distinct image objects have no repr/address contribution to route features.
    assert router.should_reason(
        ImageState("evidence", [object()]), spec, baseline, ToyTokenizer()
    ) == router.should_reason(ImageState("evidence", [object()]), spec, baseline, ToyTokenizer())
    with pytest.raises(ValueError, match="modality"):
        router.should_reason("text", spec, baseline, ToyTokenizer())
    with pytest.raises(ValueError, match="binding"):
        router.validate_binding("b" * 64, "image", "fixed")
    path = tmp_path / "router.json"
    router.save(path)
    loaded = ScopedRouter.load(path)
    assert loaded.router.validation == router.router.validation


def test_router_without_verified_benefit_cannot_be_attached():
    with pytest.raises(ValueError, match="promotion"):
        ScopedRouter(
            "a" * 64, "image", fit_router(rows("router_train", gain=-0.4), rows("dev", gain=-0.4))
        )


def test_forced_high_bypasses_a_validated_image_router_and_direct_readout(monkeypatch):
    from test_multimodal import build, media

    from ayaka.multimodal import ImageDecision, decode_media
    from ayaka.reasoning import ReasoningSettings
    from ayaka.reasoning_pipeline import Trace, controlled_decision

    _, model, tok, backend = build("gemma4")
    decision = ImageDecision(controlled_decision(model, tok), backend)
    decision.images.router = ScopedRouter(
        "a" * 64, "image", fit_router(rows("router_train"), rows("dev"))
    )
    monkeypatch.setattr(
        decision.images.router,
        "should_reason",
        lambda *args, **kwargs: pytest.fail("forced request entered router"),
    )
    monkeypatch.setattr(
        decision.images.original,
        "decide",
        lambda *args, **kwargs: pytest.fail("forced request ran direct classification"),
    )
    budgets = []

    def generate(messages, budget, reserve):
        budgets.append(budget)
        return Trace(text="Complete.", token_ids=[1], finish_reason="eos", prefill_tokens=12)

    monkeypatch.setattr(decision.images.generator, "generate_trace", generate)
    monkeypatch.setattr(decision.images.generator, "readout", lambda trace, spec: [0.99, 0.01])
    result = decision.decide(
        decode_media("Simple document", media()),
        [QuestionSpec("choice", "Type?", ["letter", "receipt"])],
        reasoning=[ReasoningSettings(mode="on", effort="high")],
    )[0]
    assert budgets == [1024]
    assert result.extras["reasoning"]["route"] == "reasoned"
    assert result.extras["reasoning"]["generated_tokens"] == 1
