"""Actual tiny HF backbones on CPU; no checkpoint download or accuracy claim."""

import inspect
from dataclasses import replace

import pytest
import torch

from ayaka.backbone import detach_text_backbone, output_rows, tiny_text_config
from ayaka.model.evidence import EvidenceResidualHead
from ayaka.prompt import QuestionView
from ayaka.tokenization import ToyTokenizer
from ayaka.training.evidence_features import (
    FeatureExtractionError,
    extract_evidence_features,
    prepare_evidence_inputs,
)
from ayaka.training.evidence_objective import EvidenceLossWeights, evidence_loss


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def questions():
    return [
        QuestionView("noul", "Approved?", ["no", "yes"]),
        QuestionView("choice", "Department?", ["sales", "finance", "support"]),
        QuestionView("score", "Urgency?", ["low", "medium", "high"], [0, 1, 2]),
    ]


@pytest.fixture
def inputs(questions):
    return prepare_evidence_inputs(
        "The request was approved by finance.", questions, ToyTokenizer(), context_limit=768
    )


def backbone(family, monkeypatch):
    from transformers import (
        Gemma4ForCausalLM,
        GraniteConfig,
        GraniteForCausalLM,
        Qwen3_5ForCausalLM,
        Qwen3_5TextConfig,
    )

    torch.manual_seed(19)
    if family == "gemma":
        lm = Gemma4ForCausalLM(tiny_text_config()).eval()
    elif family.startswith("granite"):
        config = GraniteConfig(
            vocab_size=512,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            logits_scaling=2.5,
            tie_word_embeddings=False,
            max_position_embeddings=1024,
        )
        lm = GraniteForCausalLM(config).eval()
        # Exercise actual output-head bias as well as an untied, scaled readout.
        lm.lm_head.bias = torch.nn.Parameter(torch.linspace(-0.2, 0.2, 512))
    else:
        from transformers.models.qwen3_5 import modeling_qwen3_5

        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
            monkeypatch.setattr(
                modeling_qwen3_5, name, inspect.unwrap(getattr(modeling_qwen3_5, name))
            )
        config = Qwen3_5TextConfig(
            vocab_size=512,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            tie_word_embeddings=False,
            max_position_embeddings=1024,
        )
        lm = Qwen3_5ForCausalLM(config).eval()
    if family == "granite_double":
        lm.double()
    return lm, detach_text_backbone(lm).eval()


@pytest.mark.parametrize("family", ["gemma", "granite", "qwen_hybrid", "granite_double"])
def test_native_logits_state_and_features_match_full_lm_and_cache(inputs, family, monkeypatch):
    lm, text = backbone(family, monkeypatch)
    full = extract_evidence_features(text, inputs, mode="full_rows", lexical=True)
    cached = extract_evidence_features(text, inputs, mode="prefix_cache", lexical=True)
    if family == "granite_double":
        assert cached.tensors["native_logits"].dtype == torch.float64
    for key in ("native_logits", "queries", "candidates", "memory", "lexical"):
        assert torch.allclose(full.tensors[key], cached.tensors[key], atol=2e-5, rtol=2e-5), key
    with torch.no_grad():
        for qi, question in enumerate(inputs.questions):
            ids = torch.tensor([inputs.prefix_ids + question.suffix_ids])
            expected = lm(ids, use_cache=False).logits[0, -1, list(question.label_ids)]
            actual = cached.tensors["native_logits"][0, qi, : len(question.label_ids)]
            assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert cached.metadata["forward_calls"] == 1 + len(inputs.questions)
    assert full.metadata["forward_calls"] == len(inputs.questions)
    assert full.metadata["forward_tokens"] - cached.metadata["forward_tokens"] == (
        len(inputs.questions) - 1
    ) * len(inputs.prefix_ids)
    assert cached.metadata["forward_tokens"] == cached.metadata["planned_forward_tokens"]
    assert cached.metadata["promotable"] is False


def test_prefix_is_forwarded_once_and_only_suffixes_branch(inputs, monkeypatch):
    _, text = backbone("granite", monkeypatch)
    calls = []
    hook = text.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(kwargs["input_ids"].shape[1]), with_kwargs=True
    )
    try:
        bundle = extract_evidence_features(text, inputs)
    finally:
        hook.remove()
    assert calls == [len(inputs.prefix_ids), *[len(q.suffix_ids) for q in inputs.questions]]
    assert bundle.metadata["forward_tokens"] == sum(calls)
    assert bundle.tensors["memory"].shape[1] == len(inputs.prefix_ids)
    assert (
        bundle.metadata["stored_feature_bytes"]
        <= bundle.metadata["planned_feature_bytes_upper_bound"]
    )


def test_frozen_features_can_train_head_even_inside_outer_inference_mode(inputs, monkeypatch):
    _, text = backbone("granite", monkeypatch)
    with torch.inference_mode():
        features = extract_evidence_features(text, inputs)
    assert all(not torch.is_inference(t) and not t.requires_grad for t in features.tensors.values())
    head = EvidenceResidualHead(32, dim=16, heads=4)
    out = head(**features.head_inputs())
    assert torch.equal(out.logits, features.tensors["native_logits"])
    targets = torch.zeros_like(out.logits)
    targets[..., 0] = 1
    levels = torch.zeros_like(out.logits)
    levels[0, 2] = torch.tensor([0, 1, 2])
    loss = evidence_loss(
        out,
        targets,
        features.tensors["primitive"],
        ordinals=levels,
        reference_logits=features.tensors["native_logits"],
        weights=EvidenceLossWeights(reference=0.05),
    )
    loss["total"].backward()
    assert head.correction[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in text.parameters())


@pytest.mark.parametrize("mode", ["prefix_cache", "full_rows"])
def test_unrelated_question_and_input_question_order_do_not_change_native_features(
    questions, mode, monkeypatch
):
    _, text = backbone("granite", monkeypatch)
    tok = ToyTokenizer()
    one = prepare_evidence_inputs("Approved by finance.", [questions[1]], tok)
    many = prepare_evidence_inputs(
        "Approved by finance.", [questions[2], questions[1], questions[0]], tok
    )
    a = extract_evidence_features(text, one, mode=mode)
    b = extract_evidence_features(text, many, mode=mode)
    for key in ("native_logits", "candidates", "queries"):
        assert torch.equal(a.tensors[key][0, 0], b.tensors[key][0, 1])
    assert torch.allclose(a.tensors["memory"], b.tensors["memory"], atol=1e-6, rtol=1e-6)


def test_actual_prompt_candidate_permutation_reorders_features(monkeypatch):
    _, text = backbone("granite", monkeypatch)
    tok = ToyTokenizer()
    descriptions = ["sales", "finance", "support"]
    perm = [2, 0, 1]
    a = prepare_evidence_inputs(
        "A finance question.", [QuestionView("choice", "Department?", descriptions)], tok
    )
    b = prepare_evidence_inputs(
        "A finance question.",
        [QuestionView("choice", "Department?", [descriptions[i] for i in perm])],
        tok,
    )
    assert a.questions[0].suffix_ids == b.questions[0].suffix_ids
    af = extract_evidence_features(text, a, lexical=True)
    bf = extract_evidence_features(text, b, lexical=True)
    for key in ("native_logits", "candidates", "lexical"):
        assert torch.equal(af.tensors[key][0, 0, perm], bf.tensors[key][0, 0])


def test_lexical_features_use_untied_output_rows(inputs, monkeypatch):
    _, text = backbone("granite", monkeypatch)
    bundle = extract_evidence_features(text, inputs, lexical=True)
    question = inputs.questions[1]
    s, e = question.option_spans[0]
    ids = torch.tensor(question.suffix_ids[s:e])
    expected = output_rows(text, ids).detach().float().mean(0)
    embedding_mean = text.get_input_embeddings().weight[ids].detach().float().mean(0)
    assert not torch.allclose(expected, embedding_mean)
    assert torch.equal(bundle.tensors["lexical"][0, 1, 0], expected)


def test_memory_and_all_features_are_independent_of_later_model_calls(inputs, monkeypatch):
    _, text = backbone("qwen_hybrid", monkeypatch)
    a = extract_evidence_features(text, inputs)
    frozen = {key: tensor.clone() for key, tensor in a.tensors.items()}
    changed = prepare_evidence_inputs(
        "Rejected by sales.", [QuestionView("noul", "Accepted?", ["no", "yes"])], ToyTokenizer()
    )
    extract_evidence_features(text, changed)
    for key in frozen:
        assert torch.equal(frozen[key], a.tensors[key])


@pytest.mark.parametrize(
    "cap", ["tokens", "bytes", "model_context", "vocabulary", "training", "span", "label", "mode"]
)
def test_preflight_errors_have_zero_backbone_calls(inputs, cap, monkeypatch):
    _, text = backbone("granite", monkeypatch)
    calls = []
    hook = text.register_forward_pre_hook(lambda *args: calls.append(1))
    kwargs = {}
    if cap == "tokens":
        kwargs["max_forward_tokens"] = 1
    elif cap == "bytes":
        kwargs["max_feature_bytes"] = 1
    elif cap == "model_context":
        text.config.max_position_embeddings = 10
    elif cap == "vocabulary":
        inputs = replace(inputs, prefix_ids=(512,))
    elif cap == "training":
        text.train()
    elif cap == "span":
        question = replace(inputs.questions[0], option_spans=((0, 0), (1, 2)))
        inputs = replace(inputs, questions=(question,))
    elif cap == "label":
        question = replace(inputs.questions[0], label_ids=(10, 10))
        inputs = replace(inputs, questions=(question,))
    else:
        kwargs["mode"] = "silent_fallback"
    try:
        with pytest.raises(ValueError):
            extract_evidence_features(text, inputs, **kwargs)
    finally:
        hook.remove()
    assert calls == []


@pytest.mark.parametrize(
    "bad", ["too_long", "too_many", "wrong_type", "noul_count", "score_duplicate", "score_nan"]
)
def test_render_preflight_refuses_unsupported_input(questions, bad):
    state = "A request."
    cap = 8192
    if bad == "too_long":
        state = "Never discard this fact. " * 50
        cap = 512
    elif bad == "too_many":
        questions = [QuestionView("choice", "Pick?", [str(i) for i in range(40)])]
    elif bad == "wrong_type":
        questions = [QuestionView("unknown", "Pick?", ["a", "b"])]
    elif bad == "noul_count":
        questions = [QuestionView("noul", "Pick?", ["a", "b", "c"])]
    elif bad == "score_duplicate":
        questions = [QuestionView("score", "Rate?", ["a", "b"], [1, 1])]
    else:
        questions = [QuestionView("score", "Rate?", ["a", "b"], [1, float("nan")])]
    with pytest.raises(ValueError):
        prepare_evidence_inputs(state, questions, ToyTokenizer(), context_limit=cap)


def test_full_state_tokenization_is_preserved_and_fingerprints_change(questions):
    tok = ToyTokenizer()
    state = "Initial evidence. " * 8 + "CRITICAL EXCEPTION. " + "Latest evidence. " * 8
    inputs = prepare_evidence_inputs(state, questions, tok)
    from ayaka.prompt import render_prefix

    assert inputs.prefix_ids == tuple(render_prefix(state, tok))
    changed = prepare_evidence_inputs(state + "changed", questions, tok)
    assert inputs.input_sha256 != changed.input_sha256
    assert inputs.state_sha256 != changed.state_sha256


def test_missing_cache_is_explicit_without_full_row_retry(inputs, monkeypatch):
    _, text = backbone("granite", monkeypatch)
    original = text.forward
    calls = []

    def no_cache(**kwargs):
        calls.append(kwargs["input_ids"].shape[1])
        result = original(**kwargs)
        result.past_key_values = None
        return result

    monkeypatch.setattr(text, "forward", no_cache)
    with pytest.raises(FeatureExtractionError, match="no native prefix cache") as failure:
        extract_evidence_features(text, inputs)
    assert calls == [len(inputs.prefix_ids)]
    assert failure.value.progress["attempted_forward_calls"] == 1
    assert failure.value.progress["attempted_forward_tokens"] == len(inputs.prefix_ids)


def test_failed_suffix_reports_attempted_work_without_retry(inputs, monkeypatch):
    _, text = backbone("granite", monkeypatch)
    original = text.forward
    calls = []

    def fail_second(**kwargs):
        calls.append(kwargs["input_ids"].shape[1])
        if len(calls) == 2:
            raise RuntimeError("synthetic kernel failure")
        return original(**kwargs)

    monkeypatch.setattr(text, "forward", fail_second)
    with pytest.raises(FeatureExtractionError, match="synthetic kernel failure") as failure:
        extract_evidence_features(text, inputs)
    assert len(calls) == 2
    assert failure.value.progress["attempted_forward_calls"] == 2
    assert failure.value.progress["attempted_forward_tokens"] == sum(calls)
    assert failure.value.progress["complete"] is False


def test_nonfinite_native_head_is_rejected_without_fake_prior(inputs, monkeypatch):
    _, text = backbone("granite", monkeypatch)
    with torch.no_grad():
        text._ayaka_lm_head.bias.fill_(torch.nan)
    with pytest.raises(FeatureExtractionError, match="output-head logits") as failure:
        extract_evidence_features(text, inputs)
    assert failure.value.progress["attempted_forward_calls"] == 2
