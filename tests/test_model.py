"""Structural invariants of the Electra decision model (docs.md section 48).

Runs a random tiny Gemma 4 text stack on CPU in fp32.
"""

import random

import pytest
import torch

from ayaka.collate import encode_decision, full_rows
from ayaka.config import tiny_config
from ayaka.model.electra import ElectraDecisionModel
from ayaka.model.ragged import ragged_softmax
from ayaka.primitives import Decision, QuestionSpec
from ayaka.prompt import QuestionView
from ayaka.tokenization import ToyTokenizer

TOK = ToyTokenizer()
STATE = {
    "order": {"id": 42, "status": "shipped", "carrier": "DHL"},
    "message": "where is my parcel?",
}


@pytest.fixture(scope="module")
def model():
    m = ElectraDecisionModel.from_config(tiny_config(), dtype=torch.float32)
    with torch.no_grad():
        m.gate.fill_(0.7)  # exercise the pointer path too
    return m.eval()


def test_probabilities_are_distributions(model):
    d = Decision(model, TOK)
    res = d.decide(
        STATE,
        [
            QuestionSpec("choice", "What does the user want?", ["track order", "refund", "cancel"]),
            QuestionSpec("noul", "Has the order shipped?", ["not shipped", "shipped"]),
            QuestionSpec("score", "How urgent?", ["low", "mid", "high"], ordinals=[0, 1, 2]),
        ],
    )
    for r in res:
        assert sum(r.probs) == pytest.approx(1.0, abs=1e-5)
        assert all(p >= 0 for p in r.probs)
    assert 0 <= res[1].extras["p_true"] <= 1
    assert 0 <= res[2].expected <= 2


def test_candidate_permutation_equivariance(model):
    """Invariant 1: permuting candidates permutes the distribution exactly."""
    d = Decision(model, TOK)
    cands = ["track order", "refund", "cancel", "change address", "billing"]
    base = d.decide(STATE, [QuestionSpec("choice", "Intent?", cands)])[0].distribution
    for seed in range(3):
        perm = cands[:]
        random.Random(seed).shuffle(perm)
        got = d.decide(STATE, [QuestionSpec("choice", "Intent?", perm)])[0].distribution
        for c in cands:
            assert got[c] == pytest.approx(base[c], abs=1e-6)


def test_sibling_questions_do_not_leak(model):
    """Invariant 2 + 5: adding siblings never changes a question's result,
    and batched multi-question equals single-question runs."""
    d = Decision(model, TOK)
    q = QuestionSpec("choice", "Which carrier?", ["DHL", "UPS", "FedEx"])
    alone = d.decide(STATE, [q])[0].probs
    siblings = [
        QuestionSpec("noul", "Is it urgent?", ["no", "yes"]),
        q,
        QuestionSpec("choice", "Tone?", ["angry", "calm", "neutral", "happy"]),
    ]
    together = d.decide(STATE, siblings)[1].probs
    assert together == pytest.approx(alone, abs=1e-4)


def test_cache_path_matches_full_rows(model):
    """Serving (prefix KV cache) and training (full rows) agree."""
    views = [
        QuestionView("choice", "Which carrier?", ["DHL", "UPS", "FedEx"]),
        QuestionView("noul", "Shipped?", ["no", "yes"]),
    ]
    _, items = encode_decision(STATE, views, TOK, max_seq_len=512)
    with torch.no_grad():
        out = model(full_rows(items, TOK.pad_id), apply_temperature=True)
    p_full = ragged_softmax(out.logits, out.cand_cu).tolist()
    d = Decision(model, TOK, max_seq_len=512)
    p_cache = [
        x
        for r in d.decide(
            STATE, [QuestionSpec(v.type, v.instruction, v.descriptions) for v in views]
        )
        for x in r.probs
    ]
    assert p_cache == pytest.approx(p_full, abs=1e-4)


def test_zero_gate_reproduces_backbone_readout(model):
    """g = 0 => logits are exactly the backbone's restricted label logits."""
    views = [QuestionView("choice", "Which carrier?", ["DHL", "UPS", "FedEx"])]
    _, items = encode_decision(STATE, views, TOK, max_seq_len=512)
    batch = full_rows(items, TOK.pad_id)
    with torch.no_grad():
        saved = model.gate.clone()
        model.gate.zero_()
        out = model(batch)
        model.gate.copy_(saved)
    assert torch.allclose(out.logits, out.label_logits)
    # and those are the tied-embedding LM-head logits of the label tokens
    with torch.no_grad():
        full_h = model.backbone(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        h = full_h.last_hidden_state[0, batch.answer_pos[0]]
        full = torch.tanh((model.embed_weight() @ h) / model.softcap) * model.softcap
    assert torch.allclose(out.label_logits, full[batch.label_ids], atol=1e-5)


def test_large_candidate_set_shortlist(model):
    d = Decision(model, TOK)
    cands = [f"intent {i:02d}" for i in range(40)]
    r = d.decide(STATE, [QuestionSpec("choice", "Which intent?", cands)])[0]
    assert len(r.probs) == 40
    assert sum(r.probs) == pytest.approx(1.0, abs=1e-4)


def test_temperature_applies_per_primitive(model):
    d = Decision(model, TOK)
    q = QuestionSpec("choice", "Which carrier?", ["DHL", "UPS", "FedEx"])
    p1 = d.decide(STATE, [q])[0].probs
    with torch.no_grad():
        model.temperature[1] = 4.0
    try:
        p4 = d.decide(STATE, [q])[0].probs
    finally:
        with torch.no_grad():
            model.temperature.fill_(1.0)
    assert max(p4) - min(p4) < max(p1) - min(p1)  # flatter
    assert max(range(3), key=p4.__getitem__) == max(range(3), key=p1.__getitem__)


def _both_paths(model, fn):
    """Run fn with KV-shared position pruning on and off."""
    saved = model.prune_shared_positions
    try:
        model.prune_shared_positions = True
        on = fn()
        model.prune_shared_positions = False
        off = fn()
    finally:
        model.prune_shared_positions = saved
    return on, off


def test_shared_layer_pruning_is_exact_full_rows(model):
    """Dropping unread positions in KV-shared layers changes nothing."""
    from ayaka.model.fastpath import kv_shared_start

    assert kv_shared_start(model.text_config) is not None  # tiny config shares KV
    views = [
        QuestionView("choice", "Which carrier?", ["DHL", "UPS", "FedEx"]),
        QuestionView("noul", "Shipped?", ["no", "yes"]),
        QuestionView("score", "Urgency?", ["low", "mid", "high"], ordinals=[0, 1, 2]),
    ]
    _, items = encode_decision(STATE, views, TOK, max_seq_len=512)
    batch = full_rows(items, TOK.pad_id)
    with torch.no_grad():
        on, off = _both_paths(model, lambda: model(batch))
    assert torch.allclose(on.logits, off.logits, atol=1e-5)
    assert torch.allclose(on.pointer_logits, off.pointer_logits, atol=1e-5)


def test_shared_layer_pruning_is_exact_cache_path(model):
    d = Decision(model, TOK)
    qs = [
        QuestionSpec("choice", "What does the user want?", ["track order", "refund", "cancel"]),
        QuestionSpec("noul", "Has the order shipped?", ["not shipped", "shipped"]),
    ]
    on, off = _both_paths(model, lambda: d.decide(STATE, qs))
    for a, b in zip(on, off, strict=True):
        assert a.probs == pytest.approx(b.probs, abs=1e-5)


def test_shared_layer_pruning_same_gradients():
    """Training through the pruned path yields the same gradients."""
    m = ElectraDecisionModel.from_config(tiny_config(), dtype=torch.float32)
    views = [QuestionView("choice", "Which carrier?", ["DHL", "UPS", "FedEx"])]
    _, items = encode_decision(STATE, views, TOK, max_seq_len=512)
    batch = full_rows(items, TOK.pad_id)
    grads = []
    for flag in (True, False):
        m.zero_grad()
        m.prune_shared_positions = flag
        m(batch).logits.logsumexp(0).backward()
        grads.append(m.text_model().layers[0].mlp.down_proj.weight.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_head_cache_reuse_is_exact(model):
    """Encoding the constant prompt head once and reusing it is exact,
    including across requests with different states."""
    d = Decision(model, TOK)
    qs = [
        QuestionSpec("choice", "Which carrier?", ["DHL", "UPS", "FedEx"]),
        QuestionSpec("noul", "Shipped?", ["no", "yes"]),
    ]
    other = {"order": {"id": 9, "status": "pending"}, "message": "cancel it please " * 20}
    for state in (STATE, other, STATE):
        d.reuse_head = True
        a = d.decide(state, qs)
        d.reuse_head = False
        b = d.decide(state, qs)
        for ra, rb in zip(a, b, strict=True):
            assert ra.probs == pytest.approx(rb.probs, abs=1e-5)
