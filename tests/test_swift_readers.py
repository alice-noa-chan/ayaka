import json
import math
from types import SimpleNamespace

import pytest

from ayaka.swift.readers import (
    HFReader,
    VLLMChatReader,
    aggregate_letter_logits,
    cached_tokenizer,
    canonical_letter_ids,
    gather_token_logits,
    letter_token_ids,
    token_input,
    top_letter_probs,
)


def test_canonical_letters_reject_aliases_and_missing_mass():
    entries = [{"token": "A", "logprob": math.log(0.4)}, {"token": "B", "logprob": math.log(0.6)}]
    assert top_letter_probs(entries, ["A", "B"]) == pytest.approx({"A": 0.4, "B": 0.6})
    with pytest.raises(ValueError, match="missing canonical"):
        top_letter_probs(entries, ["A", "B", "C"])
    with pytest.raises(ValueError, match="missing canonical"):
        top_letter_probs([], ["A", "B"])
    with pytest.raises(ValueError, match="noncanonical.* A"):
        top_letter_probs([*entries, {"token": " A", "logprob": -1}], ["A", "B"])


@pytest.mark.parametrize("kwargs", [None, {"thinking": False}, {}])
def test_vllm_request_and_usage(monkeypatch, kwargs):
    payload = {
        "prompt_token_ids": [7, 8],
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "token_id:65",
                            "top_logprobs": [
                                {"token": "token_id:65", "logprob": math.log(0.6)},
                                {"token": "token_id:66", "logprob": math.log(0.4)},
                            ],
                        }
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
    }
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(payload).encode()

    def urlopen(request, timeout):
        seen.update(url=request.full_url, body=json.loads(request.data), timeout=timeout)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    result = VLLMChatReader(
        "http://localhost:9999/v1",
        "tiny",
        5,
        chat_template_kwargs=kwargs,
        revision="a" * 40,
        tokenizer=StubTokenizer(),
    ).read([{"role": "user", "content": "Q"}], ["A", "B"])
    assert seen["url"] == "http://localhost:9999/v1/chat/completions"
    assert seen["timeout"] == 5
    assert seen["body"] == {
        "model": "tiny",
        "messages": [{"role": "user", "content": "Q"}],
        "max_tokens": 1,
        "logprobs": True,
        "logprob_token_ids": [65, 66],
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False} if kwargs is None else kwargs,
    }
    assert result.letter_probs == pytest.approx({"A": 0.6, "B": 0.4})
    assert (result.input_tokens, result.output_tokens) == (2, 1)
    assert result.latency_s >= 0


def test_hf_mapping_all_ids_and_stable_aggregation():
    tokens = ["<unk>", "A", " A", "A\n", "B", " B ", "AB", "a"]
    tokenizer = SimpleNamespace(decode=lambda ids: tokens[ids[0]])

    class Tokenizer:
        def __len__(self):
            return len(tokens)

        def decode(self, ids):
            return tokenizer.decode(ids)

    mapping = letter_token_ids(Tokenizer())
    assert mapping["A"] == [1, 2, 3]
    assert mapping["B"] == [4, 5]
    probs = aggregate_letter_logits(
        [9999, 1000, 1000, 1000, 1000, 1000, 9999, 9999], mapping, ["A", "B"]
    )
    assert probs == pytest.approx({"A": 0.6, "B": 0.4})
    with pytest.raises(ValueError, match="no token"):
        aggregate_letter_logits([0] * len(tokens), mapping, ["C"])


def test_hf_tiny_model_constructed_offline(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("transformers.models.gpt2.modeling_gpt2", exc_type=ImportError)
    tokenizer_impl = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"<unk>": 0, "A": 1, "B": 2, "Q": 3}, unk_token="<unk>")
    )
    tokenizer_impl.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl, unk_token="<unk>"
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}{% if add_generation_prompt %}Q {% endif %}"
    config = transformers.GPT2Config(
        vocab_size=4, n_positions=32, n_embd=8, n_layer=1, n_head=1, bos_token_id=0, eos_token_id=0
    )
    torch.manual_seed(15)
    model = transformers.GPT2LMHeadModel(config).eval()
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    reader = HFReader(str(tmp_path))
    messages = [{"role": "user", "content": "Q"}]
    result = reader.read(messages, ["A", "B"])
    encoded = reader.tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        enable_thinking=False,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    with torch.inference_mode():
        logits = reader.model(**encoded).logits[0, -1]
        expected = torch.softmax(logits[[1, 2]], dim=0).tolist()
    assert result.letter_probs == pytest.approx(
        dict(zip(["A", "B"], expected, strict=True)), abs=1e-6
    )
    assert result.input_tokens == 2
    assert result.output_tokens == 1
    assert reader._letter_ids is None  # No vocabulary scan in canonical mode.
    reader.read(messages, ["A", "B"])
    diagnostic = HFReader(str(tmp_path), readout="alias_sum")
    assert diagnostic.read(messages, ["A", "B"]).letter_probs == pytest.approx(result.letter_probs)
    assert diagnostic._letter_ids["A"] == [1]


def test_hf_forward_options_do_not_guess_support_from_arbitrary_kwargs():
    from ayaka.swift.readers import _read_forward_kwargs

    class Model:
        def forward(self, input_ids, **kwargs):
            pass

    assert _read_forward_kwargs(Model()) == {}
    assert _read_forward_kwargs(SimpleNamespace()) == {}


def test_hf_forward_failure_propagates_without_retrying_inference():
    import torch

    class Tokenizer(StubTokenizer):
        def apply_chat_template(self, messages, tokenize, **kwargs):
            if kwargs.get("return_tensors"):
                return {"input_ids": torch.tensor([[7, 8]])}
            return super().apply_chat_template(messages, tokenize, **kwargs)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, input_ids, logits_to_keep=0, use_cache=True):
            self.calls += 1
            assert logits_to_keep == 1 and use_cache is False
            raise TypeError("backend execution failed")

    reader = HFReader("offline-failure-fixture")
    reader.tokenizer, reader.model = Tokenizer(), Model()
    with pytest.raises(TypeError, match="backend execution failed"):
        reader.read([], ["A", "B"])
    assert reader.model.calls == 1


def test_vllm_rejects_decoded_strings(monkeypatch):
    import io

    payload = {"choices": [{"logprobs": {"content": [{"token": " A", "top_logprobs": []}]}}]}
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **kw: io.BytesIO(json.dumps(payload).encode())
    )
    reader = VLLMChatReader("http://fixture", "fixture", tokenizer=StubTokenizer())
    with pytest.raises(ValueError, match="token_id:N"):
        reader.read([], ["A", "B"])


class StubTokenizer:
    def apply_chat_template(self, messages, tokenize, **kwargs):
        return [7, 8] if tokenize else "assistant:"

    def encode(self, text, add_special_tokens):
        return [7, 8] + ([] if text == "assistant:" else [ord(text[-1])])


class BatchEncodingStubTokenizer(StubTokenizer):
    """transformers 5 returns a dict-like BatchEncoding from tokenize=True by default."""

    def apply_chat_template(self, messages, tokenize, **kwargs):
        return {"input_ids": [7, 8], "attention_mask": [1, 1]} if tokenize else "assistant:"


@pytest.mark.parametrize("tokenizer", [StubTokenizer(), BatchEncodingStubTokenizer()])
def test_token_input_accepts_list_and_batch_encoding(tokenizer):
    from ayaka.swift.readers import token_input

    result = token_input(tokenizer, [], ["A", "B"], {})
    assert result["input_token_ids"] == [7, 8]


class FastBatchTokenizer(StubTokenizer):
    is_fast = True

    def __init__(self):
        self.encodes = []
        self.batches = []

    def encode(self, text, add_special_tokens):
        self.encodes.append(text)
        return super().encode(text, add_special_tokens)

    def __call__(self, texts, *, add_special_tokens):
        self.batches.append(texts)
        return {
            "input_ids": [StubTokenizer.encode(self, text, add_special_tokens) for text in texts]
        }


def test_token_input_batches_full_continuations_and_reuses_the_verified_prefix():
    tokenizer = FastBatchTokenizer()
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    actual = token_input(tokenizer, [], letters, {})
    assert actual == token_input(StubTokenizer(), [], letters, {})
    assert tokenizer.encodes == ["assistant:"]
    assert tokenizer.batches == [["assistant:" + letter for letter in letters]]


@pytest.mark.parametrize(
    "rows,error",
    [
        ([[7, 8, 65]], "one row per letter"),
        ([[7, 9, 65], [7, 8, 66]], "answer position"),
        ([[7, 8, 65, 99], [7, 8, 66]], "answer position"),
        ([[7, 8, 65], []], "answer position"),
        ([[7, 8, 65], [7, 8, 65]], "distinct token ids"),
        ([[7, 8, 65], [7, 8, -1]], "vocabulary integers"),
    ],
)
def test_batched_canonical_ids_reject_missing_rows_and_invalid_continuations(rows, error):
    class Tokenizer(FastBatchTokenizer):
        def __call__(self, texts, *, add_special_tokens):
            return {"input_ids": rows}

    with pytest.raises(ValueError, match=error):
        canonical_letter_ids(Tokenizer(), "assistant:", ["A", "B"])


def test_fast_bpe_rejects_a_letter_that_merges_into_the_answer_prefix():
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        models.BPE(
            vocab={"[UNK]": 0, "x": 1, "A": 2, "B": 3, "xA": 4},
            merges=[("x", "A")],
            unk_token="[UNK]",
        )
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    assert tokenizer.encode("A", add_special_tokens=False) == [2]
    with pytest.raises(ValueError, match="letter A.*answer position"):
        canonical_letter_ids(tokenizer, "x", ["A", "B"])


@pytest.mark.parametrize(
    "entries,error",
    [
        ([{"token": "token_id:65", "logprob": 0}], "missing"),
        ([{"token": "token_id:65", "logprob": 0}] * 2, "duplicate"),
        ([{"token": "token_id:65", "logprob": -9999}], "clipped"),
        ([{"token": "token_id:65", "logprob": -10000}], "clipped"),
        ([{"token": "token_id:65", "logprob": float("nan")}], "finite"),
        ([{"token": "token_id:65", "logprob": float("inf")}], "finite"),
        ([{"token": "token_id:99", "logprob": 0}], "unexpected"),
    ],
)
def test_exact_gather_rejects_loss(entries, error):
    with pytest.raises(ValueError, match=error):
        gather_token_logits(entries, [65, 66], "token_id:0")


def test_exact_gather_ignores_only_extra_sampled_token():
    entries = [
        {"token": f"token_id:{i}", "logprob": v} for i, v in [(0, 1000), (65, 100), (66, 98)]
    ]
    assert gather_token_logits(entries, [65, 66], "token_id:0") == {65: 100, 66: 98}
    assert gather_token_logits(entries[1:], [65, 66], "token_id:65") == {65: 100, 66: 98}


def test_vllm_26_letters_one_exact_request(monkeypatch):
    import io

    calls = []

    def urlopen(request, timeout):
        body = json.loads(request.data)
        calls.append(body)
        ids = body["logprob_token_ids"]
        payload = {
            "prompt_token_ids": [7, 8],
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": "token_id:0",
                                "top_logprobs": [
                                    {
                                        "token": f"token_id:{i}",
                                        "logprob": 100.0 if i == 65 else 98.0,
                                    }
                                    for i in [0, *ids]
                                ],
                            }
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    result = VLLMChatReader("http://fixture", "fixture", tokenizer=StubTokenizer()).read(
        [], list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )
    assert len(calls) == 1 and len(calls[0]["logprob_token_ids"]) == 26
    assert "top_logprobs" not in calls[0] and "structured_outputs" not in calls[0]
    assert result.letter_probs["A"] / result.letter_probs["Z"] == pytest.approx(math.exp(2))
    assert result.letter_log_masses["A"] == 100
    assert result.input_token_ids == [7, 8]


def test_tokenizer_only_lazy_cache_per_pinned_revision(monkeypatch):
    import sys

    from ayaka.swift import readers

    calls = []
    readers._TOKENIZERS.clear()

    def load(model, **kwargs):
        calls.append((model, kwargs))
        return StubTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)),
    )
    reader = VLLMChatReader("http://fixture", "fixture", revision="a" * 40)
    assert not calls
    reader.describe([], ["A", "B"])
    assert cached_tokenizer("fixture", "a" * 40) is reader.tokenizer
    cached_tokenizer("fixture", "b" * 40)
    assert len(calls) == 2 and all(kw["local_files_only"] for _, kw in calls)
    with pytest.raises(ValueError, match="pinned"):
        cached_tokenizer("fixture", "main")
    readers._TOKENIZERS.clear()


def test_vllm_exact_field_rejection_is_a_loud_preflight_error(monkeypatch):
    import urllib.error

    def rejected(*args, **kwargs):
        raise urllib.error.HTTPError("http://fixture", 400, "unknown logprob_token_ids", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", rejected)
    reader = VLLMChatReader("http://fixture", "fixture", tokenizer=StubTokenizer())
    with pytest.raises(ValueError, match="exact token gather rejected.*raw_logits"):
        reader.read([], ["A", "B"])


def test_vllm_rejects_server_prompt_token_mismatch(monkeypatch):
    import io

    payload = {
        "prompt_token_ids": [8, 7],
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "token_id:65",
                            "top_logprobs": [
                                {"token": "token_id:65", "logprob": 0},
                                {"token": "token_id:66", "logprob": 0},
                            ],
                        }
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
    }
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **kw: io.BytesIO(json.dumps(payload).encode())
    )
    with pytest.raises(ValueError, match="server prompt token ids"):
        VLLMChatReader("http://fixture", "fixture", tokenizer=StubTokenizer()).read([], ["A", "B"])
