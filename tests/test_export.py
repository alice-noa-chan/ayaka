"""Export (bf16 + int8), quantized runtime, and the TypeSafe server."""

import json
import threading
import urllib.request

import pytest
import torch

from ayaka.config import tiny_config
from ayaka.export import export_model, load_exported, parity
from ayaka.model.electra import ElectraDecisionModel
from ayaka.primitives import Decision, QuestionSpec
from ayaka.quant import Int8Embedding, dequantize_rows, quantize_rows
from ayaka.serve import BadRequest, DecisionService, parse_question, serve
from ayaka.tokenization import ToyTokenizer
from ayaka.training.batching import sample_to_items
from ayaka.training.run import synthetic_pools

TOK = ToyTokenizer()
STATE = {"ticket": {"id": 7, "text": "my parcel never arrived"}, "tier": "gold"}
QUESTIONS = [
    QuestionSpec("choice", "What does the user want?", ["track order", "refund", "cancel"]),
    QuestionSpec("noul", "Is the customer premium?", ["not premium", "premium"]),
    QuestionSpec("score", "How urgent?", ["low", "medium", "high"], ordinals=[0, 1, 2]),
]


@pytest.fixture(scope="module")
def model():
    m = ElectraDecisionModel.from_config(tiny_config(), dtype=torch.float32)
    with torch.no_grad():
        m.gate.fill_(0.5)
        m.temperature[1] = 1.3
    return m.eval().requires_grad_(False)


def test_quantize_rows_roundtrip():
    w = torch.randn(64, 32) * torch.linspace(0.01, 3, 64)[:, None]
    q, s = quantize_rows(w)
    assert q.dtype == torch.int8 and s.shape == (64,)
    err = (dequantize_rows(q, s, torch.float32) - w).abs().max(dim=1).values
    assert torch.all(err <= s * 0.5 + 1e-6)  # at most half a quantization step per row


def test_full_export_roundtrip_is_exact(model, tmp_path):
    out = export_model(model, str(tmp_path / "electra-tiny"), "tiny")
    loaded, tok = load_exported(out, dtype=torch.float32)
    a = Decision(model, TOK).decide(STATE, QUESTIONS)
    b = Decision(loaded, tok).decide(STATE, QUESTIONS)
    for ra, rb in zip(a, b, strict=True):
        assert rb.probs == pytest.approx(ra.probs, abs=1e-5)
    assert (tmp_path / "electra-tiny" / "README.md").exists()
    meta = json.loads((tmp_path / "electra-tiny" / "export_meta.json").read_text())
    assert meta["quantized"] is False


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_int8_export_close_to_full(model, tmp_path, dtype):
    out = export_model(model, str(tmp_path / f"q-{dtype}"), "tiny", quantize=True)
    q, tok = load_exported(out, dtype=dtype)
    text = q.text_model()
    assert isinstance(text.embed_tokens, Int8Embedding)
    assert isinstance(text.embed_tokens_per_layer, Int8Embedding)
    # linears are dynamic-int8 on CPU (or bf16-dequantized where unsupported)
    assert not any(t.is_meta for t in list(text.parameters()) + list(text.buffers()))
    ref = Decision(model, TOK).decide(STATE, QUESTIONS)
    got = Decision(q, tok).decide(STATE, QUESTIONS)
    for ra, rb in zip(ref, got, strict=True):
        assert sum(rb.probs) == pytest.approx(1.0, abs=1e-3)
        assert max(abs(x - y) for x, y in zip(ra.probs, rb.probs, strict=True)) < 0.08
    # file is smaller than the full export
    full = export_model(model, str(tmp_path / f"f-{dtype}"), "tiny")

    def size(d):
        return sum(p.stat().st_size for p in (tmp_path / d / "backbone").glob("*.safetensors"))

    assert size(f"q-{dtype}") < 0.6 * size(f"f-{dtype}")
    assert full


def test_parity_report(model, tmp_path):
    out = export_model(model, str(tmp_path / "q"), "tiny", quantize=True)
    q, tok = load_exported(out, dtype=torch.float32)
    items = [
        it
        for cell in synthetic_pools(3).values()
        for s in cell
        for it in sample_to_items(s, TOK, model.cfg)
    ]
    rep = parity(model, q, tok, items, "cpu")
    assert rep["n"] == len(items)
    assert 0.0 <= rep["argmax_agreement"] <= 1.0 and rep["mean_kl"] >= 0.0


# -------------------------------------------------------------- server


def test_parse_typesafe_questions():
    spec, labels = parse_question(
        {"type": "choice", "instructions": "i", "criteria": {"a": "alpha", "b": ""}}
    )
    assert labels == ["a", "b"] and spec.candidates == ["alpha", "b"]
    spec, labels = parse_question(
        {"type": "score", "instructions": "i", "criteria": ["lo", "mid", "hi"]}
    )
    assert labels == ["0", "1", "2"] and spec.ordinals == [0, 1, 2]
    spec, labels = parse_question({"type": "noul", "instructions": "ok?"})
    assert spec.candidates == ["false", "true"]
    with pytest.raises(BadRequest):
        parse_question({"type": "choice", "instructions": "i", "criteria": {"only": "one"}})
    with pytest.raises(BadRequest):
        parse_question({"type": "rank", "instructions": "i"})


def test_service_answers_match_decision(model):
    svc = DecisionService(Decision(model, TOK), "electra-tiny")
    body = {
        "state": STATE,
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "What does the user want?",
                "criteria": {"track": "track order", "refund": "refund", "cancel": "cancel"},
            },
            "premium": {
                "type": "noul",
                "instructions": "Is the customer premium?",
                "criteria": {"false": "not premium", "true": "premium"},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["low", "medium", "high"],
            },
        },
    }
    out = svc.handle(body)
    ref = Decision(model, TOK).decide(STATE, QUESTIONS)
    a = out["answers"]
    assert a["intent"]["probabilities"]["refund"] == pytest.approx(ref[0].probs[1], abs=1e-5)
    assert a["intent"]["choice"] == max(
        a["intent"]["probabilities"], key=a["intent"]["probabilities"].get
    )
    assert a["premium"]["noul"] == pytest.approx(ref[1].probs[1], abs=1e-5)
    assert a["urgency"]["score"] == pytest.approx(ref[2].expected, abs=1e-5)
    assert set(a["urgency"]["probabilities"]) == {"0", "1", "2"}
    assert out["usage"]["input_tokens"] > 0 and out["usage"]["output_tokens"] == 0


def test_http_server_roundtrip(model):
    httpd = serve(Decision(model, TOK), "electra-tiny", host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        body = {
            "state": "a short note",
            "questions": {"d": {"type": "noul", "instructions": "Is it short?"}},
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/systemone",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
        assert out["answers"]["d"]["type"] == "noul"
        assert 0.0 <= out["answers"]["d"]["noul"] <= 1.0
        bad = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/systemone",
            data=b'{"questions": {}}',
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad, timeout=60)
        assert e.value.code == 400
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
            assert json.loads(r.read())["status"] == "ok"
    finally:
        httpd.shutdown()
