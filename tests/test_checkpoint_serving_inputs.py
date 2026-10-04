"""Actual offline checkpoint -> typed service parity, including native LoRA."""

import copy
import json
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, replace

import pytest
import torch
from test_evidence_swift_bridge import native_model, tokenizer

from ayaka.backbone import detach_text_backbone
from ayaka.checkpoint import apply_lora, load_checkpoint, save_checkpoint
from ayaka.config import tiny_config
from ayaka.eval.read_artifact import fingerprint
from ayaka.export import export_model, load_exported
from ayaka.http_transport import request_bytes
from ayaka.input_contract import metadata_contract
from ayaka.model.electra import ElectraDecisionModel
from ayaka.primitives import Decision, QuestionSpec
from ayaka.reasoning import ReasoningSettings
from ayaka.reasoning_pipeline import controlled_decision
from ayaka.serve import DecisionService, parse_question, serve
from ayaka.tokenization import HFTokenizer
from ayaka.training.batching import collate_items
from ayaka.training.swift_direct import input_serving_recipe, swift_sample_to_items

ENCODING = {
    "encoder": "swift_canonical",
    "prompt_variant": "labeled",
    "state_format": "compact",
    "chat_template_kwargs": {"enable_thinking": False},
}
STATE = {"facts": "Approved. Rule 3 applies.\n승인はい."}
QUESTIONS = {
    "pick": {
        "type": "choice",
        "instructions": "Which rule?",
        "criteria": {"third": "Rule 3", "second": "Rule 2"},
    },
    "judge": {
        "type": "noul",
        "instructions": "Approved?",
        "criteria": {"false": "No approval", "true": "Approval granted"},
    },
    "rate": {
        "type": "score",
        "instructions": "Which level?",
        "criteria": {"10": "High", "-2": "Low", "3": "Mid"},
    },
}


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def setup_checkpoint(tmp_path, family="granite", *, variant="labeled"):
    lm, text = native_model(family)
    # Write the real native LM layout, not the detached text stack's extra alias.
    text._modules.pop("_ayaka_lm_head", None)
    if family == "granite":
        lm.lm_head.bias = None  # the stock Granite config uses an unbiased output head
    native_dir = tmp_path / "native"
    lm.save_pretrained(native_dir)
    native_tok = tokenizer()
    native_tok.save_pretrained(native_dir)
    tok = HFTokenizer(native_tok, str(native_dir))
    cfg = tiny_config(
        version=2,
        backbone=str(native_dir),
        readout="lm",
        max_seq_len=2048,
        serve_max_seq_len=2048,
        long_prompt_tokens=180,
        lora_dropout=0,
    )
    model = apply_lora(ElectraDecisionModel(cfg, detach_text_backbone(lm), text.config))
    generator = torch.Generator().manual_seed(117)
    with torch.no_grad():
        for name, weight in model.backbone.named_parameters():
            if "lora_B" in name:
                weight.copy_(torch.randn(weight.shape, generator=generator) * 0.08)
        model.temperature.copy_(torch.tensor([[0.6, 1.7], [0.8, 2.0], [1.2, 0.7]]))
    encoding = {**ENCODING, "prompt_variant": variant}
    recipe = input_serving_recipe(tok, encoding)
    meta = {
        "input_encoding": encoding,
        "input_recipe": recipe,
        "input_recipe_sha256": fingerprint(recipe),
    }
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(model, str(checkpoint), meta)
    loaded = load_checkpoint(
        str(checkpoint),
        dtype=torch.float32,
        merge=False,
        local_files_only=True,
        strict_loading=True,
    ).eval()
    return model.eval(), loaded, tok, checkpoint, meta


def expected(model, tok, state, questions, encoding):
    from ayaka.data.schema import Sample
    from ayaka.input_contract import serving_question

    sample = Sample(state, [serving_question(q.view(), i) for i, q in enumerate(questions)])
    items = swift_sample_to_items(
        sample, tok, model.cfg, **{k: v for k, v in encoding.items() if k != "encoder"}
    )
    with torch.inference_mode():
        batch = collate_items(items, tok.pad_id)
        out = model(batch.batch, apply_temperature=True)
        probabilities = out.probs().tolist()
        cu = out.cand_cu.tolist()
    return items, [probabilities[cu[i] : cu[i + 1]] for i in range(len(items))]


@pytest.mark.parametrize("family", ["granite", "gemma"])
@pytest.mark.parametrize("variant", ["min", "labeled"])
def test_real_checkpoint_service_matches_full_training_rows(tmp_path, family, variant):
    source, model, tok, checkpoint, meta = setup_checkpoint(tmp_path, family, variant=variant)
    questions = [parse_question(q)[0] for q in QUESTIONS.values()]
    direct = Decision(model, tok)
    for state in ("Approved.", STATE, {"facts": "Evidence only. " * 30}):
        items, probabilities = expected(source, tok, state, questions, meta["input_encoding"])
        prefix, encoded = direct.encode(state, [q.view() for q in questions])
        assert [asdict(row) for row in encoded] == [asdict(item.enc) for item in items]
        assert prefix == items[0].enc.prefix_ids
        actual = direct.decide(state, questions)
        for result, p in zip(actual, probabilities, strict=True):
            assert result.probs == pytest.approx(p, abs=3e-6)
        assert direct._head is None
        # Isolated requests, a warmed instance, and reordered questions all agree.
        alone = [direct.decide(state, [q])[0].probs for q in questions]
        reverse = [r.probs for r in direct.decide(state, list(reversed(questions)))][::-1]
        for p, single, rev in zip(probabilities, alone, reverse, strict=True):
            assert single == pytest.approx(p, abs=3e-6)
            assert rev == pytest.approx(p, abs=3e-6)
        body = {"state": state, "questions": QUESTIONS, "options": {"reasoning": {"mode": "off"}}}
        for decision in (direct, controlled_decision(model, tok)):
            response = DecisionService(decision, "fixture").handle(copy.deepcopy(body))
            assert response["usage"]["reasoning_tokens"] == 0
            assert response["usage"]["input_tokens"] == sum(direct.input_counts(state, questions))
            assert response["answers"]["judge"]["noul"] == pytest.approx(
                probabilities[1][1], abs=3e-6
            )
            assert list(response["answers"]["pick"]["probabilities"]) == ["third", "second"]
            assert list(response["answers"]["rate"]["probabilities"]) == ["10", "-2", "3"]
            assert list(response["answers"]["rate"]["probabilities"].values()) == pytest.approx(
                probabilities[2], abs=3e-6
            )
    copy_path = tmp_path / "resaved"
    save_checkpoint(model, str(copy_path), {"source": "resaved"})
    assert json.loads((copy_path / "meta.json").read_text())["input_recipe"] == meta["input_recipe"]
    with pytest.raises(ValueError, match="replace"):
        save_checkpoint(model, str(tmp_path / "bad"), {"input_recipe": {}})


@pytest.mark.parametrize(
    "change", ["missing", "partial", "digest", "encoding", "version", "non_lm"]
)
def test_invalid_contract_rejected_before_weight_loading(tmp_path, monkeypatch, change):
    _, _, tok, checkpoint, meta = setup_checkpoint(tmp_path)
    if change == "missing":
        meta = {}
    elif change == "partial":
        meta.pop("input_recipe")
    elif change == "digest":
        meta["input_recipe_sha256"] = "0" * 64
    elif change == "encoding":
        meta["input_encoding"]["prompt_variant"] = "min"
    elif change == "version":
        meta["input_recipe"]["version"] = "unknown"
        meta["input_recipe_sha256"] = fingerprint(meta["input_recipe"])
    else:
        config = json.loads((checkpoint / "ayaka_config.json").read_text())
        config["readout"] = "hybrid"
        (checkpoint / "ayaka_config.json").write_text(json.dumps(config))
    (checkpoint / "meta.json").write_text(json.dumps(meta))

    def forbid(*args, **kwargs):
        pytest.fail("invalid metadata reached weight loader")

    monkeypatch.setattr(ElectraDecisionModel, "from_config", forbid)
    with pytest.raises(ValueError):
        load_checkpoint(str(checkpoint), local_files_only=True)


def test_mutation_unsupported_inputs_and_empty_questions(tmp_path, monkeypatch):
    _, model, tok, _, _ = setup_checkpoint(tmp_path)
    decision = Decision(model, tok)
    q = parse_question(QUESTIONS["pick"])[0]
    assert decision.decide(STATE, []) == []

    def forbid(*args, **kwargs):
        pytest.fail("invalid input reached forward")

    monkeypatch.setattr(model, "encode_prefix", forbid)
    with pytest.raises(ValueError, match="complete|fit|context"):
        decision.decide("Evidence" * 2000, [q])
    with pytest.raises(ValueError):
        decision.decide(STATE, [QuestionSpec("choice", "Pick", [str(i) for i in range(27)])])
    with pytest.raises(ValueError, match="integers"):
        decision.decide(STATE, [QuestionSpec("score", "Level", ["lo", "hi"], [0.5, 1.5])])
    tok.hf.chat_template += "!"
    with pytest.raises(ValueError, match="tokenizer|template"):
        decision.decide(STATE, [q])
    with pytest.raises(ValueError, match="tokenizer|template"):
        Decision(model, tok)


def test_swift_never_uses_legacy_shortlist_for_small_config_alphabet(tmp_path):
    source, model, tok, _, meta = setup_checkpoint(tmp_path)
    model.cfg = replace(model.cfg, max_label_candidates=1)
    spec = parse_question(QUESTIONS["pick"])[0]
    _, probabilities = expected(source, tok, STATE, [spec], meta["input_encoding"])
    assert Decision(model, tok).decide(STATE, [spec])[0].probs == pytest.approx(
        probabilities[0], abs=3e-6
    )


def test_http_context_limit_is_422_and_tokenizer_mismatch_is_backend_error(tmp_path, monkeypatch):
    _, model, tok, _, _ = setup_checkpoint(tmp_path)
    decision = controlled_decision(model, tok)

    def forbid(*args, **kwargs):
        pytest.fail("invalid input reached native forward")

    monkeypatch.setattr(model, "encode_prefix", forbid)
    httpd = serve(decision, "fixture", host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        for index, (state, code) in enumerate((("Evidence" * 2000, 422), (STATE, 502))):
            if code == 502:
                tok.hf.chat_template += "!"
            body = {
                "state": state,
                "questions": QUESTIONS,
                "options": {"reasoning": {"mode": "off"}},
            }
            request = urllib.request.Request(
                f"http://127.0.0.1:{httpd.server_address[1]}/v1/systemone",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Idempotency-Key": f"invalid-{index}"},
            )
            with pytest.raises(urllib.error.HTTPError) as exc:
                request_bytes(request, timeout=20)
            assert exc.value.code == code
            response = json.loads(exc.value.read())
            if code == 422:
                assert response["field"] == "questions"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_native_merged_export_retains_recipe_and_calibrated_service(tmp_path):
    source, model, tok, _, meta = setup_checkpoint(tmp_path, "gemma")
    model.backbone = model.backbone.merge_and_unload()
    out = export_model(
        model, str(tmp_path / "export"), model.cfg.backbone, meta={"source": "fixture"}
    )
    loaded, exported_tok = load_exported(out, dtype=torch.float32)
    assert loaded.input_contract == metadata_contract(meta, source.cfg)
    questions = [parse_question(q)[0] for q in QUESTIONS.values()]
    _, probabilities = expected(source, tok, STATE, questions, meta["input_encoding"])
    actual = Decision(loaded, exported_tok).decide(STATE, questions)
    for result, p in zip(actual, probabilities, strict=True):
        assert result.probs == pytest.approx(p, abs=3e-6)
    (tmp_path / "export" / "export_meta.json").write_text('{"input_encoding": {}}')
    with pytest.raises(ValueError, match="incomplete"):
        load_exported(out, dtype=torch.float32)


def test_legacy_checkpoint_still_uses_segmented_inputs(tmp_path):
    model = ElectraDecisionModel.from_config(tiny_config(), dtype=torch.float32)
    path = tmp_path / "legacy"
    save_checkpoint(model, str(path))
    loaded = load_checkpoint(str(path), dtype=torch.float32)
    assert loaded.input_contract is None


def favor_trace_token(model, tok):
    # A deterministic native LM head makes real greedy/cache execution readable.
    # No generation or loader is mocked; this is random-model mechanics only.
    head = model.text_model()._ayaka_lm_head
    bias = torch.zeros(head.weight.shape[0])
    bias[tok.hf.encode("x", add_special_tokens=False)[0]] = 100
    head.bias = torch.nn.Parameter(bias, requires_grad=False)


def test_real_swift_trace_continues_same_cache_and_matches_full_native_read(tmp_path):
    from ayaka.collate import full_rows
    from ayaka.model.electra import PRIMITIVE_INDEX
    from ayaka.prompt import RenderedQuestion

    _, model, tok, _, _ = setup_checkpoint(tmp_path)
    favor_trace_token(model, tok)
    decision = controlled_decision(model, tok)
    for spec in [parse_question(q)[0] for q in QUESTIONS.values()]:
        messages = decision.generator.messages_for(STATE, spec)
        reserve = decision.generator.reserve_tokens(messages, spec)
        trace = decision.generator.generate_trace(messages, 3, reserve)
        cache = trace.cache
        assert trace.text == "xxx" and trace.generated_tokens == 3
        probabilities = decision.generator.readout(trace, spec)
        assert trace.cache is cache
        rendered = RenderedQuestion(
            trace.readout_input_ids,
            [(0, 1)] * len(spec.candidates),
            trace.readout_label_ids,
            list(range(len(spec.candidates))),
        )
        from ayaka.collate import EncodedQuestion

        batch = full_rows([EncodedQuestion([], rendered, PRIMITIVE_INDEX[spec.type])], tok.pad_id)
        with torch.inference_mode():
            full = model(batch, apply_temperature=True).probs().tolist()
        assert probabilities == pytest.approx(full, abs=3e-6)
        assert trace.readout_tokens == len(trace.readout_input_ids) - len(trace.input_ids) - 3


def test_forced_high_real_native_generation_has_no_baseline_or_router(tmp_path, monkeypatch):
    _, model, tok, _, _ = setup_checkpoint(tmp_path)
    favor_trace_token(model, tok)
    decision = controlled_decision(model, tok)

    def forbid(*args, **kwargs):
        pytest.fail("forced on+high reached baseline/router")

    monkeypatch.setattr(decision.original, "decide", forbid)

    class Router:
        should_reason = staticmethod(forbid)

    decision.router = Router()
    q = QuestionSpec("noul", "Is the greeting polite?", ["no", "yes"])
    result = decision.decide("Hello", [q], reasoning=[ReasoningSettings(mode="on", effort="high")])[
        0
    ]
    diagnostics = result.extras["reasoning"]
    assert diagnostics["route"] == "reasoned"
    assert diagnostics["budget"] == 1024
    assert diagnostics["generated_tokens"] == 1024
    assert diagnostics["finish_reason"] == "length"


def test_empty_eos_and_context_failure_preserve_budget_and_account_fallback(tmp_path):
    _, model, tok, _, _ = setup_checkpoint(tmp_path)
    favor_trace_token(model, tok)
    decision = controlled_decision(model, tok)
    token = tok.hf.encode("x", add_special_tokens=False)[0]
    decision.generator.eos = {token}
    spec = parse_question(QUESTIONS["pick"])[0]
    high = [ReasoningSettings(mode="on", effort="high")]
    expected = decision.original.decide(STATE, [spec])[0].probs
    result = decision.decide(STATE, [spec], reasoning=high)[0]
    diagnostics = result.extras["reasoning"]
    assert result.probs == pytest.approx(expected, abs=3e-6)
    assert diagnostics["route"] == "fallback" and diagnostics["finish_reason"] == "empty_trace"
    assert diagnostics["generated_tokens"] == 1 and diagnostics["budget"] == 1024
    assert diagnostics["input_tokens"] > sum(decision.original.input_counts(STATE, [spec]))
    decision.generator.max_context = 700
    result = decision.decide(STATE, [spec], reasoning=high)[0]
    diagnostics = result.extras["reasoning"]
    assert diagnostics["route"] == "fallback" and diagnostics["finish_reason"] == "context_limit"
    assert diagnostics["generated_tokens"] == 0 and diagnostics["budget"] == 1024


@pytest.mark.parametrize(
    "token",
    [
        "bos_token",
        "eos_token",
        "split_special_tokens",
        "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output",
    ],
)
def test_existing_vocabulary_special_token_mutation_rejected(tmp_path, token):
    _, model, tok, _, meta = setup_checkpoint(tmp_path)
    before = tok.hf.backend_tokenizer.to_str()
    setattr(tok.hf, token, "x" if token.endswith("_token") else not getattr(tok.hf, token, False))
    assert tok.hf.backend_tokenizer.to_str() == before
    assert input_serving_recipe(tok, meta["input_encoding"]) != meta["input_recipe"]
    with pytest.raises(ValueError, match="tokenizer"):
        Decision(model, tok)


def test_post_generation_decode_failure_keeps_real_generated_usage(tmp_path, monkeypatch):
    _, model, tok, _, _ = setup_checkpoint(tmp_path)
    favor_trace_token(model, tok)
    decision = controlled_decision(model, tok)
    decode = tok.hf.decode

    def fail_special_decode(*args, **kwargs):
        if kwargs.get("skip_special_tokens"):
            raise ValueError("injected post-generation decoder failure")
        return decode(*args, **kwargs)

    monkeypatch.setattr(tok.hf, "decode", fail_special_decode)
    spec = parse_question(QUESTIONS["pick"])[0]
    result = decision.decide(STATE, [spec], reasoning=[ReasoningSettings(mode="on", max_tokens=3)])[
        0
    ]
    diagnostic = result.extras["reasoning"]
    assert diagnostic["route"] == "fallback" and diagnostic["finish_reason"] == "generation_error"
    assert diagnostic["generated_tokens"] == 3
    assert diagnostic["input_tokens"] > sum(decision.original.input_counts(STATE, [spec]))


def test_native_service_preserves_signed_levels_and_rejects_fractional_aliases(
    tmp_path, monkeypatch
):
    _, model, tok, _, meta = setup_checkpoint(tmp_path)
    raw = {"type": "score", "instructions": "Level?", "criteria": {"+2": "lo", "+4": "hi"}}
    spec = parse_question(raw, native_score=True)[0]
    assert spec.ordinals == [2, 4]
    _, probabilities = expected(model, tok, STATE, [spec], meta["input_encoding"])
    service = DecisionService(controlled_decision(model, tok), "fixture")
    body = {"state": STATE, "questions": {"s": raw}, "options": {"reasoning": {"mode": "off"}}}
    answer = service.handle(body)["answers"]["s"]
    assert list(answer["probabilities"]) == ["+2", "+4"]
    assert answer["score"] == pytest.approx(
        2 * probabilities[0][0] + 4 * probabilities[0][1], abs=3e-6
    )

    def forbid(*args, **kwargs):
        pytest.fail("invalid Score levels reached native forward")

    monkeypatch.setattr(model, "encode_prefix", forbid)
    from ayaka.serve import BadRequest

    for criteria in ({"0.5": "lo", "1.5": "hi"}, {"01": "lo", "1": "hi"}):
        body["questions"]["s"]["criteria"] = criteria
        with pytest.raises(BadRequest) as exc:
            service.handle(body)
        assert exc.value.field == "criteria"
    assert parse_question(raw)[0].ordinals == [1, 1]  # historical v1 parse remains unchanged


def test_real_gemma_sliding_cache_keeps_eos_outside_trace_then_matches_full_read(tmp_path):
    from ayaka.collate import EncodedQuestion, full_rows
    from ayaka.model.electra import PRIMITIVE_INDEX
    from ayaka.model.fastpath import prefill_last
    from ayaka.prompt import RenderedQuestion

    _, model, tok, _, _ = setup_checkpoint(tmp_path, "gemma")
    decision = controlled_decision(model, tok)
    generator = decision.generator
    spec = parse_question(QUESTIONS["pick"])[0]
    messages = generator.messages_for(STATE, spec)
    ids, _ = generator.prepare(messages)
    trace_token = tok.hf.encode("~", add_special_tokens=False)[0]
    eos_token = tok.hf.encode("@", add_special_tokens=False)[0]
    assert trace_token not in ids and eos_token not in ids
    text = model.text_model()
    embedding = text.get_input_embeddings().weight
    # Schedule a native head EOS after the sliding window has been crossed.
    # Every greedy selection and cache update still runs the actual tiny model.
    with torch.inference_mode():
        hidden, _ = prefill_last(text, torch.tensor([ids]))
        embedding[trace_token].copy_(hidden[0] * 100)
    generator.eos = {eos_token}
    calls = 0

    def finish_after_window(module, args, output):
        nonlocal calls
        calls += 1
        if calls == 32:
            with torch.inference_mode():
                embedding[eos_token].copy_(output.last_hidden_state[0, -1] * 10000)

    hook = text.register_forward_hook(finish_after_window)
    try:
        trace = generator.generate_trace(messages, 40, generator.reserve_tokens(messages, spec))
    finally:
        hook.remove()
    assert trace.finish_reason == "eos" and trace.token_ids[-1] == eos_token
    assert trace.generated_tokens == 32 and trace.text == "~" * 31
    assert trace.cache.get_seq_length() == len(trace.input_ids) + 31
    cache = trace.cache
    probabilities = generator.readout(trace, spec)
    assert trace.cache is cache
    rendered = RenderedQuestion(
        trace.readout_input_ids, [(0, 1)] * 2, trace.readout_label_ids, [0, 1]
    )
    batch = full_rows([EncodedQuestion([], rendered, PRIMITIVE_INDEX[spec.type])], tok.pad_id)
    with torch.inference_mode():
        reference = model(batch, apply_temperature=True).probs().tolist()
    assert probabilities == pytest.approx(reference, abs=3e-6)
