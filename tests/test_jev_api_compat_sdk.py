"""Official 0.7.2 SDK integration, skipped cleanly when the optional SDK is absent."""

import pytest

sdk = pytest.importorskip("typesafe_sdk", exc_type=ImportError)

import test_jev_api_compat as compat  # noqa: E402

api_server = compat.api_server
questions = compat.questions
request = compat.request

from ayaka.client import AyakaResponse, extra_body, system_one  # noqa: E402


@pytest.mark.parametrize("structured", [False, True])
def test_official_sdk_default_model_all_primitives_and_models(api_server, structured):
    base, _, _ = api_server
    with sdk.TypeSafeClient(
        api_key="test-key", base_url=base, retry=sdk.RetryPolicy(max_retries=0)
    ) as client:
        specs = questions(structured)
        typed = {
            name: {"noul": sdk.Noul, "choice": sdk.Choice, "score": sdk.Score}[q["type"]](
                **{k: v for k, v in q.items() if k != "type"}
            )
            for name, q in specs.items()
        }
        result = client.system_one(state=[{"message": "test"}], questions=typed)
        assert result.model == "ayaka-test-1.0.0"
        assert result.nouls["n"].noul == pytest.approx(0.5)
        assert result.choices["c"].confidence == pytest.approx(0)
        assert result.scores["s"].legend[0] == (["low", {"rule": True}] if structured else "low")
        assert {m.name for m in client.models.list().models} >= {"jev-latest", "jev-preview"}
        extension = {"reasoning": {"mode": "off"}}
        extended = client.system_one(
            state="test",
            questions=typed,
            extra_body=extra_body(extension),
            response_model=AyakaResponse,
        )
        assert isinstance(extended, sdk.SystemOneResponse)
        assert extended.choices["c"].ayaka["route"] == "direct"
        assert extended.scores["s"].ayaka["calibration"] == "unfitted"
        wrapped = system_one(client, "test", typed, ayaka=extension)
        assert wrapped.answers["n"].ayaka["calibration"] == "unfitted"
        status, raw, _ = request(base, {"state": "test", "questions": specs, "ayaka": extension})
        assert status == 200 and raw["answers"]["c"]["ayaka"] == extended.answers["c"].ayaka


def test_sdk_http_errors_with_retries_disabled(api_server):
    base, _, runtime = api_server
    policy = sdk.RetryPolicy(max_retries=0)
    with sdk.TypeSafeClient(api_key="wrong", base_url=base, retry=policy) as client:
        with pytest.raises(sdk.TypeSafeAuthenticationError) as exc:
            client.system_one("x", {"q": sdk.Noul()})
        assert exc.value.status == 401 and exc.value.request_id
        with pytest.raises(sdk.TypeSafeAuthenticationError):
            client.models.list()
    with sdk.TypeSafeClient(api_key="test-key", base_url=base, retry=policy) as client:
        with pytest.raises(sdk.TypeSafeUnprocessableEntityError) as exc:
            client.system_one("x", {"q": sdk.Noul()}, model="unknown")
        assert exc.value.body["field"] == "model"
        assert runtime.slots.acquire(blocking=False)
        try:
            with pytest.raises(sdk.TypeSafeRateLimitError) as exc:
                client.system_one("x", {"q": sdk.Noul()})
            assert exc.value.status == 429 and exc.value.retry_after_ms == 1000
        finally:
            runtime.slots.release()


def test_namespace_question_name_and_helper_conflicts(api_server):
    base, _, _ = api_server
    with sdk.TypeSafeClient(
        api_key="test-key", base_url=base, retry=sdk.RetryPolicy(max_retries=0)
    ) as client:
        result = system_one(
            client,
            "test",
            {"ayaka": sdk.Noul()},
            extra_body={"ayaka": {"reasoning": {"mode": "off"}}},
        )
        assert result.ayaka == (
            {"usage": {"reasoning_tokens": 0}} if "usage" in result.ayaka else {}
        )
        assert result.answers["ayaka"].ayaka["calibration"] == "unfitted"
        assert result.request_id
        assert result.raw_http_response.status_code == 200
    with pytest.raises(ValueError, match="conflicting"):
        extra_body({"reasoning": {"mode": "off"}}, extra={"ayaka": {"reasoning": {"mode": "on"}}})
