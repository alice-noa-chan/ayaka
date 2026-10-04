"""Real offline HF tokenizer/model parity with Swift's serving reader."""

import copy
from dataclasses import replace
from string import printable

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    Gemma4ForCausalLM,
    GraniteConfig,
    GraniteForCausalLM,
    PreTrainedTokenizerFast,
)

from ayaka.backbone import detach_text_backbone, tiny_text_config
from ayaka.eval.read_artifact import fingerprint
from ayaka.swift.prompt import render_question
from ayaka.swift.readers import READOUT, HFReader
from ayaka.training.swift_evidence import (
    extract_swift_evidence_features,
    prepare_swift_evidence_inputs,
)


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tokenizer(trim=False, merged=False):
    chars = dict.fromkeys(printable + "승인はい営業")
    vocab = {"[UNK]": 0, **{char: i for i, char in enumerate(chars, 1)}}
    if merged:
        # A real greedy BPE merges across the state/question boundary (x + \n).
        vocab["x\n"] = len(vocab)
        model = models.BPE(vocab=vocab, merges=[("x", "\n")], unk_token="[UNK]")
    else:
        model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    backend = Tokenizer(model)
    if not merged:
        backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    backend.decoder = decoders.Fuse()
    result = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    content = "message['content'].strip()" if trim else "message['content']"
    result.chat_template = (
        "{% for message in messages %}[{{message['role']}}]\n{{"
        + content
        + "}}\n{% endfor %}{% if add_generation_prompt %}[assistant]\n{% endif %}"
    )
    return result


def native_model(family):
    torch.manual_seed(73)
    if family == "gemma":
        lm = Gemma4ForCausalLM(tiny_text_config()).eval()
    else:
        lm = GraniteForCausalLM(
            GraniteConfig(
                vocab_size=512,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                logits_scaling=2.5,
                tie_word_embeddings=False,
                max_position_embeddings=2048,
            )
        ).eval()
        lm.lm_head.bias = torch.nn.Parameter(torch.linspace(-0.2, 0.2, 512))
    return lm, detach_text_backbone(lm).eval()


QUESTIONS = [
    {"type": "noul", "instructions": "Approved?", "criteria": {"true": "yes\n승인"}},
    {"type": "choice", "instructions": "Where?", "criteria": {"sales": "営業", "it": "help\nはい"}},
    {
        "type": "score",
        "instructions": "Level?",
        "criteria": {"10": "high", "-2": "low", "3": "mid"},
    },
]


@pytest.mark.parametrize("family", ["gemma", "granite"])
@pytest.mark.parametrize("variant", ["min", "cygnet", "rules"])
@pytest.mark.parametrize("state_format", ["pretty", "compact"])
@pytest.mark.parametrize("mode", ["full_rows", "prefix_cache", "copy_on_write"])
def test_actual_swift_reader_and_feature_prior_are_identical(family, variant, state_format, mode):
    tok = tokenizer()
    lm, text = native_model(family)
    state = {"policy": "Approved by finance.\nUse the updated rules.", "flag": True}
    prepared = prepare_swift_evidence_inputs(
        state, QUESTIONS, tok, prompt_variant=variant, state_format=state_format, context_limit=1536
    )
    feature_options = (
        {"mode": "prefix_cache", "cache_strategy": "copy_on_write"}
        if mode == "copy_on_write"
        else {"mode": mode}
    )
    features = extract_swift_evidence_features(text, prepared, **feature_options)
    reader = HFReader("offline-random-fixture", device="cpu", dtype="float32", readout=READOUT)
    reader.model, reader.tokenizer = lm, tok  # No load, download, or pretrained accuracy claim.
    for qi, question in enumerate(QUESTIONS):
        messages, mapping = render_question(
            state, question, prompt_variant=variant, state_format=state_format
        )
        retained = prepared.recipe["questions"][qi]
        assert retained["labels"] == list(mapping.values())
        result = reader.read(messages, list(mapping))
        assert retained["input_token_ids"] == result.input_token_ids
        assert retained["canonical_token_ids"] == result.canonical_token_ids
        logits = features.tensors["native_logits"][0, qi, : len(mapping)].float()
        actual = logits.softmax(-1)
        expected = torch.tensor(list(result.letter_probs.values()))
        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
        assert torch.allclose(
            logits - logits.mean(),
            torch.tensor(list(result.letter_log_masses.values())) - logits.mean(),
            atol=2e-5,
            rtol=2e-5,
        )
    assert prepared.recipe["questions"][2]["labels"] == ["-2", "3", "10"]
    assert features.metadata["prior_recipe_sha256"] == fingerprint(prepared.recipe)
    assert features.metadata["serving_readout"] == READOUT
    assert features.metadata["promotable"] is False


@pytest.mark.parametrize("trim", [False, True])
@pytest.mark.parametrize("variant", ["min", "cygnet", "rules"])
def test_whitespace_multiline_options_and_question_isolation(trim, variant):
    tok = tokenizer(trim=trim)
    _, text = native_model("granite")
    questions = copy.deepcopy(QUESTIONS)
    questions[1]["criteria"]["it"] = "help\nはい  \n"
    state = "  Finance reviewed the document.  \n"
    a = prepare_swift_evidence_inputs(state, questions, tok, prompt_variant=variant)
    b = prepare_swift_evidence_inputs(state, list(reversed(questions)), tok, prompt_variant=variant)
    af = extract_swift_evidence_features(text, a)
    bf = extract_swift_evidence_features(text, b)
    assert torch.equal(af.tensors["memory"], bf.tensors["memory"])
    for key in ("native_logits", "queries", "candidates"):
        assert torch.equal(af.tensors[key].flip(1), bf.tensors[key])
    changed = prepare_swift_evidence_inputs(
        state,
        [{"type": "choice", "instructions": "Different?", "criteria": ["a", "b"]}],
        tok,
        prompt_variant=variant,
    )
    cf = extract_swift_evidence_features(text, changed)
    assert torch.allclose(af.tensors["memory"], cf.tensors["memory"], atol=1e-6, rtol=1e-6)
    q = a.inputs.questions[1]
    ids = a.inputs.prefix_ids + q.suffix_ids
    prompt = tok.decode(ids)
    last_span = [len(a.inputs.prefix_ids) + i for i in range(*q.option_spans[-1])]
    pooled = tok.decode([ids[i] for i in last_span])
    assert "[assistant]" not in pooled
    assert "help\nはい" in pooled
    assert "[assistant]" in prompt


def test_boundary_straddling_bpe_token_stays_in_suffix_with_reader_parity():
    tok = tokenizer(merged=True)
    lm, text = native_model("granite")
    state = "State ends x"
    prepared = prepare_swift_evidence_inputs(state, QUESTIONS, tok)
    merged_id = tok.convert_tokens_to_ids("x\n")
    assert merged_id not in prepared.inputs.prefix_ids
    assert all(q.suffix_ids[0] == merged_id for q in prepared.inputs.questions)
    features = extract_swift_evidence_features(text, prepared)
    for qi, q in enumerate(prepared.inputs.questions):
        ids = torch.tensor([prepared.inputs.prefix_ids + q.suffix_ids])
        with torch.no_grad():
            expected = lm(ids).logits[0, -1, list(q.label_ids)].softmax(-1)
        assert torch.allclose(
            features.tensors["native_logits"][0, qi, : len(q.label_ids)].softmax(-1),
            expected,
            atol=2e-6,
            rtol=2e-6,
        )


@pytest.mark.parametrize("mutation", ["input", "labels", "span", "primitive", "recipe", "template"])
def test_mutated_binding_rejected_before_model_forward(mutation):
    prepared = prepare_swift_evidence_inputs("state", QUESTIONS, tokenizer())
    if mutation in ("input", "labels", "span", "primitive"):
        q = prepared.inputs.questions[0]
        if mutation == "input":
            q = replace(q, suffix_ids=q.suffix_ids[:-1] + (1,))
        elif mutation == "labels":
            q = replace(q, label_ids=tuple(reversed(q.label_ids)))
        elif mutation == "span":
            q = replace(q, option_spans=tuple(reversed(q.option_spans)))
        else:
            q = replace(q, primitive=1)
        prepared = replace(
            prepared, inputs=replace(prepared.inputs, questions=(q, *prepared.inputs.questions[1:]))
        )
    elif mutation == "recipe":
        prepared.recipe["questions"][0]["canonical_token_ids"]["A"][0] = 1
    else:
        prepared.recipe["chat_template_sha256"] = "f" * 64
    # None cannot be forwarded: all binding checks precede feature/model access.
    with pytest.raises(ValueError, match="binding changed"):
        extract_swift_evidence_features(None, prepared)


@pytest.mark.parametrize("case", ["slow", "thinking", "too_many", "context", "blank", "rewritten"])
def test_unsupported_inputs_fail_explicitly(case):
    tok = tokenizer()
    questions, options = QUESTIONS, {}
    if case == "slow":
        tok = type("Slow", (), {"is_fast": False})()
    elif case == "thinking":
        options["chat_template_kwargs"] = {"enable_thinking": True}
    elif case == "too_many":
        questions = [{"type": "choice", "criteria": [str(i) for i in range(27)]}]
    elif case == "context":
        options["context_limit"] = 10
    elif case == "blank":
        questions = [{"type": "choice", "criteria": {"a": " ", "b": "ok"}}]
    else:
        tok.chat_template = "[assistant]\n"  # Discards the evidence entirely.
    with pytest.raises(ValueError):
        prepare_swift_evidence_inputs("state", questions, tok, **options)


def test_recipe_metadata_is_a_snapshot_not_a_mutable_alias():
    prepared = prepare_swift_evidence_inputs("state", QUESTIONS, tokenizer())
    _, text = native_model("granite")
    features = extract_swift_evidence_features(text, prepared)
    prepared.recipe["questions"][0]["labels"][0] = "changed"
    assert features.metadata["prior_recipe"]["questions"][0]["labels"][0] == "false"
    assert (
        fingerprint(features.metadata["prior_recipe"]) == features.metadata["prior_recipe_sha256"]
    )
