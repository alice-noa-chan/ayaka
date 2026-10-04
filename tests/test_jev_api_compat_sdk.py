"""Official SDK 0.7.2 against the same tiny CPU export used by v1 tests."""

import threading

import pytest

sdk = pytest.importorskip("typesafe_sdk", exc_type=ImportError)

import torch  # noqa: E402
from test_jev_api_compat import questions, running_server  # noqa: E402

from ayaka.client import AyakaResponse, extra_body, system_one  # noqa: E402
from ayaka.config import tiny_config  # noqa: E402
from ayaka.export import export_model, load_exported  # noqa: E402
from ayaka.jev_api import choice_confidence, score_confidence  # noqa: E402
from ayaka.model.electra import ElectraDecisionModel  # noqa: E402
from ayaka.primitives import Decision  # noqa: E402
from ayaka.serve import parse_question  # noqa: E402


@pytest.fixture(scope="module")
def tiny_decision(tmp_path_factory):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = ElectraDecisionModel.from_config(tiny_config(), dtype=torch.float32)
    with torch.no_grad():
        model.gate.fill_(0.5)
        model.temperature[1] = 1.3
    model.eval().requires_grad_(False)
    exported = export_model(model, str(tmp_path_factory.mktemp("sdk") / "electra-tiny"), "tiny")
    del model
    loaded, tok = load_exported(exported, device="cpu", dtype=torch.float32)
    return Decision(loaded, tok)


@pytest.fixture
def sdk_server(tiny_decision):
    with running_server(tiny_decision) as base:
        yield base, tiny_decision


def typed_questions(structured=False):
    return {
        name: {"noul": sdk.Noul, "choice": sdk.Choice, "score": sdk.Score}[q["type"]](
            **{k: v for k, v in q.items() if k != "type"}
        )
        for name, q in questions(structured).items()
    }


@pytest.mark.parametrize("structured", [False, True])
def test_official_sdk_all_primitives_default_model_and_models(sdk_server, structured):
    base, decision = sdk_server
    state = [{"message": "test"}]
    specs = questions(structured)
    direct = decision.decide(state, [parse_question(q)[0] for q in specs.values()])
    with sdk.TypeSafeClient(
        api_key="test-key", base_url=base, retry=sdk.RetryPolicy(max_retries=0)
    ) as client:
        result = client.system_one(state=state, questions=typed_questions(structured))
        assert result.model == "ayaka-test-1.0.0"
        assert result.nouls["n"].noul == pytest.approx(direct[0].probs[1], abs=1e-5)
        assert list(result.choices["c"].probabilities.values()) == pytest.approx(
            direct[1].probs, abs=1e-5
        )
        assert list(result.scores["s"].probabilities.values()) == pytest.approx(
            direct[2].probs, abs=1e-5
        )
        assert result.choices["c"].confidence == pytest.approx(choice_confidence(direct[1].probs))
        assert result.scores["s"].confidence == pytest.approx(score_confidence(direct[2].probs))
        assert result.scores["s"].legend == dict(enumerate(specs["s"]["criteria"]))
        assert result.usage.input_tokens > 0 and result.usage.output_tokens == 0
        listing = client.models.list()
        assert {m.name for m in listing.models} == {
            "jev-latest",
            "jev-preview",
            "ayaka-test",
            "ayaka-test-1.0.0",
        }
        assert all(m.description and m.release_date for m in listing.models)


def test_official_sdk_extra_body_and_helper_roundtrip(sdk_server):
    base, _ = sdk_server
    typed = typed_questions(True)
    typed["ayaka"] = sdk.Noul(instructions="A question named ayaka?")
    with sdk.TypeSafeClient(
        api_key="test-key", base_url=base, retry=sdk.RetryPolicy(max_retries=0)
    ) as client:
        baseline = client.system_one("test", typed)
        extended = client.system_one(
            "test", typed, extra_body={"ayaka": {}}, response_model=AyakaResponse
        )
        assert isinstance(extended, sdk.SystemOneResponse)
        assert extended.nouls["ayaka"].noul == baseline.nouls["ayaka"].noul
        assert extended.choices["c"].probabilities == baseline.choices["c"].probabilities
        assert set(extended.ayaka) == {"latency_ms"}
        wrapped = system_one(client, "test", typed, ayaka={"reasoning": "on"})
        assert wrapped.answers == extended.answers
        assert set(wrapped.ayaka) == {"latency_ms"}
    assert extra_body({}, extra={"custom": 1}) == {"custom": 1, "ayaka": {}}
    with pytest.raises(ValueError, match="conflicting"):
        extra_body({}, extra={"ayaka": {"x": 1}})


def test_official_sdk_401_and_422_without_retries(sdk_server):
    base, _ = sdk_server
    policy = sdk.RetryPolicy(max_retries=0)
    with sdk.TypeSafeClient(api_key="wrong", base_url=base, retry=policy) as client:
        for call in (lambda: client.system_one("test", typed_questions()), client.models.list):
            with pytest.raises(sdk.TypeSafeAPIError) as exc:
                call()
            assert exc.value.status == 401 and exc.value.request_id
    with sdk.TypeSafeClient(api_key="test-key", base_url=base, retry=policy) as client:
        for kwargs in (
            {"model": "unknown"},
            {"extra_body": {"questions": {}}},
            {"extra_body": {"ayaka": []}},
        ):
            with pytest.raises(sdk.TypeSafeAPIError) as exc:
                client.system_one("test", typed_questions(), **kwargs)
            assert exc.value.status == 422 and exc.value.request_id


def test_official_sdk_429_without_retries_and_recovery(sdk_server, monkeypatch):
    base, decision = sdk_server
    entered, release = threading.Event(), threading.Event()
    original = decision.decide
    pending = []

    def blocked(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)

    def first_request():
        with sdk.TypeSafeClient(
            api_key="test-key", base_url=base, retry=sdk.RetryPolicy(max_retries=0)
        ) as client:
            pending.append(client.system_one("test", typed_questions()))

    monkeypatch.setattr(decision, "decide", blocked)
    thread = threading.Thread(target=first_request)
    thread.start()
    try:
        assert entered.wait(5)
        with sdk.TypeSafeClient(
            api_key="test-key", base_url=base, retry=sdk.RetryPolicy(max_retries=0)
        ) as client:
            with pytest.raises(sdk.TypeSafeAPIError) as exc:
                client.system_one("test", typed_questions())
            assert exc.value.status == 429 and exc.value.headers["retry-after"] == "1"
            assert exc.value.request_id
    finally:
        release.set()
        thread.join(10)
    assert len(pending) == 1
    with sdk.TypeSafeClient(
        api_key="test-key", base_url=base, retry=sdk.RetryPolicy(max_retries=0)
    ) as client:
        assert client.system_one("test", typed_questions()).model == "ayaka-test-1.0.0"
