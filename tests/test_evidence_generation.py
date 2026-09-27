from contextlib import contextmanager

import pytest
import torch

from ayaka.config import tiny_config
from ayaka.evidence import EvidenceError
from ayaka.evidence_generation import PlanGenerator, chat_ids, plan_complete
from ayaka.model.electra import ElectraDecisionModel
from ayaka.tokenization import ToyTokenizer


class Tok(ToyTokenizer):
    def decode(self, ids):
        return " ".join(map(str, ids))


def _model():
    torch.manual_seed(7)
    model = ElectraDecisionModel.from_config(tiny_config(), dtype=torch.float32, device="cpu")
    return model.eval()


def _naive_greedy(model, ids, steps):
    text, embed = model.text_model(), model.embed_weight()
    ids = list(ids)
    out = []
    with torch.no_grad():
        for _ in range(steps):
            hidden = text(input_ids=torch.tensor([ids])).last_hidden_state[:, -1, :]
            token = int(torch.nn.functional.linear(hidden, embed).argmax(-1))
            out.append(token)
            ids.append(token)
    return out


def test_batched_cached_generation_matches_naive_greedy_per_prompt():
    model, tok = _model(), Tok()
    messages = [
        [{"role": "user", "content": "short"}],
        [{"role": "user", "content": "a noticeably longer prompt for left padding"}],
    ]
    gen = PlanGenerator(model, tok, max_new_tokens=6, stop_when=None)
    gen.eos = set()  # random weights: compare the full greedy continuation
    got = gen.generate(messages)
    for m, text in zip(messages, got, strict=True):
        expected = _naive_greedy(model, chat_ids(tok, m), 6)
        assert text == " ".join(map(str, expected))


def test_generation_runs_with_the_adapter_disabled_and_rejects_overlong_context():
    model, tok = _model(), Tok()
    entered = []

    @contextmanager
    def disable_adapter():
        entered.append(True)
        yield

    object.__setattr__(model.backbone, "disable_adapter", disable_adapter)
    gen = PlanGenerator(model, tok, max_new_tokens=2, stop_when=None)
    gen([{"role": "user", "content": "x"}])
    assert entered == [True]
    with pytest.raises(EvidenceError):
        PlanGenerator(model, tok, max_new_tokens=2, max_context=8)(
            [{"role": "user", "content": "a prompt that is far too long"}]
        )


def test_plan_complete_detects_a_parseable_id_plan():
    assert plan_complete('{"e":[0],"c":{}} trailing')
    assert not plan_complete('{"e":[0],"c":{')
