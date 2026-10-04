import json
import math
from types import SimpleNamespace

import pytest

from ayaka.swift.readers import (
    HFReader,
    VLLMChatReader,
    aggregate_letter_logits,
    letter_token_ids,
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
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "A", "logprob": math.log(0.6)},
                                {"token": "B", "logprob": math.log(0.4)},
                            ]
                        }
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 17, "completion_tokens": 1},
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
        "http://localhost:9999/v1", "tiny", 5, chat_template_kwargs=kwargs
    ).read([{"role": "user", "content": "Q"}], ["A", "B"])
    assert seen["url"] == "http://localhost:9999/v1/chat/completions"
    assert seen["timeout"] == 5
    assert seen["body"] == {
        "model": "tiny",
        "messages": [{"role": "user", "content": "Q"}],
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 2,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False} if kwargs is None else kwargs,
        "structured_outputs": {"choice": ["A", "B"]},
    }
    assert result.letter_probs == pytest.approx({"A": 0.6, "B": 0.4})
    assert (result.input_tokens, result.output_tokens) == (17, 1)
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


def test_vllm_records_and_rejects_aliases(monkeypatch):
    import io

    payload = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": " A", "logprob": 0},
                                {"token": "B", "logprob": 0},
                            ]
                        }
                    ]
                }
            }
        ]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode())
    )
    reader = VLLMChatReader("http://fixture", "fixture")
    with pytest.raises(ValueError, match="rejected noncanonical"):
        reader.read([], ["A", "B"])
    assert reader.rejected_aliases == [" A"]
