import pytest
import torch

from ayaka.collate import build_decision_inputs
from ayaka.config import tiny_config
from ayaka.model.model import CHOICE, ElectraDecisionModel
from ayaka.primitives import Decision, QuestionSpec
from ayaka.tokenizer import HashTokenizer


def _model(seed=0):
    torch.manual_seed(seed)
    return ElectraDecisionModel(tiny_config()).eval()


def _forward(model, samples):
    tok = HashTokenizer(vocab_size=512)
    inp = build_decision_inputs(samples, tok)
    with torch.no_grad():
        return model(
            state_ids=inp.state_ids,
            state_cu=inp.state_cu,
            question_ids=inp.question_ids,
            question_cu=inp.question_cu,
            question_state_index=inp.question_state_index,
            candidate_ids=inp.candidate_ids,
            candidate_cu=inp.candidate_cu,
            candidate_question_index=inp.candidate_question_index,
            primitive_index=inp.primitive_index,
        )


def test_forward_shapes_and_normalization():
    model = _model()
    samples = [
        {
            "state": {"msg": "card charged twice"},
            "questions": [
                {"type": "choice", "instruction": "route?", "candidates": ["a", "b", "c"]},
                {"type": "noul", "instruction": "is angry?", "candidates": ["f", "t"]},
            ],
        }
    ]
    out = _forward(model, samples)
    assert out.logits.shape == (5,)
    assert out.cand_cu.tolist() == [0, 3, 5]
    probs = out.probs()
    assert torch.allclose(probs[:3].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(probs[3:].sum(), torch.tensor(1.0), atol=1e-5)


def test_candidate_permutation_equivariance():
    # F(state, q, pi(C)) = pi(F(state, q, C)) (sec 13.2)
    model = _model()
    base = {
        "state": "evidence text",
        "questions": [
            {"type": "choice", "instruction": "pick", "candidates": ["x", "y", "z", "w"]}
        ],
    }
    perm = {
        "state": "evidence text",
        "questions": [
            {"type": "choice", "instruction": "pick", "candidates": ["w", "x", "z", "y"]}
        ],
    }
    p1 = _forward(model, [base]).probs()
    p2 = _forward(model, [perm]).probs()
    order = [3, 0, 2, 1]  # [x,y,z,w] -> [w,x,z,y]
    assert torch.allclose(p1[order], p2, atol=1e-5)


def test_question_isolation_and_batch_equivalence():
    # sec 48 invariant 5: same-state multi-question == individual runs
    model = _model()
    q1 = {"type": "choice", "instruction": "q1?", "candidates": ["a", "b"]}
    q2 = {"type": "choice", "instruction": "q2?", "candidates": ["p", "q", "r"]}
    both = _forward(model, [{"state": "s", "questions": [q1, q2]}])
    only1 = _forward(model, [{"state": "s", "questions": [q1]}])
    only2 = _forward(model, [{"state": "s", "questions": [q2]}])
    assert torch.allclose(both.probs()[:2], only1.probs(), atol=1e-5)
    assert torch.allclose(both.probs()[2:], only2.probs(), atol=1e-5)


def test_state_isolation_across_samples():
    model = _model()
    q = {"type": "choice", "instruction": "q?", "candidates": ["a", "b"]}
    s1 = [{"state": "state one", "questions": [q]}]
    s2 = [
        {"state": "state one", "questions": [q]},
        {"state": "completely different", "questions": [q]},
    ]
    p1 = _forward(model, s1).probs()
    p2 = _forward(model, s2).probs()
    assert torch.allclose(p1, p2[:2], atol=1e-5)


def test_variable_candidate_counts():
    model = _model()
    for k in (2, 5, 17):
        cands = [f"cand{i}" for i in range(k)]
        out = _forward(
            model,
            [
                {
                    "state": "s",
                    "questions": [{"type": "choice", "instruction": "q", "candidates": cands}],
                }
            ],
        )
        assert out.probs().shape == (k,)
        assert torch.allclose(out.probs().sum(), torch.tensor(1.0), atol=1e-4)


def test_long_context_routed_path():
    cfg = tiny_config(block_size=8, short_context_threshold=16)
    torch.manual_seed(0)
    model = ElectraDecisionModel(cfg).eval()
    long_state = "word " * 40  # > threshold tokens
    out = _forward(
        model,
        [
            {
                "state": long_state.strip(),
                "questions": [
                    {"type": "choice", "instruction": "q", "candidates": ["a", "b", "c"]}
                ],
            }
        ],
    )
    assert out.memory.route_tokens
    assert torch.allclose(out.probs().sum(), torch.tensor(1.0), atol=1e-5)


def test_primitives_api():
    model = _model()
    dec = Decision(model, HashTokenizer(vocab_size=512))
    dist = dec.choice("customer charged twice", "route?", ["billing", "tech", "sales"])
    assert set(dist) == {"billing", "tech", "sales"}
    assert abs(sum(dist.values()) - 1.0) < 1e-4
    p_true = dec.noul("sky is blue", "the sky is blue")
    assert 0.0 <= p_true <= 1.0
    expected, sdist = dec.score("review text", "severity?", ["low", "mid", "high"])
    assert set(sdist) == {"low", "mid", "high"}
    assert 0.0 <= expected <= 2.0


def test_multi_question_decide_matches_singles():
    model = _model()
    dec = Decision(model, HashTokenizer(vocab_size=512))
    state = "shared state text"
    qs = [
        QuestionSpec("choice", "q1?", ["a", "b"]),
        QuestionSpec("score", "q2?", ["l1", "l2", "l3"]),
    ]
    both = dec.decide(state, qs)
    single1 = dec.decide(state, [qs[0]])
    single2 = dec.decide(state, [qs[1]])
    for k in both[0].distribution:
        assert both[0].distribution[k] == pytest.approx(single1[0].distribution[k], abs=1e-5)
    for k in both[1].distribution:
        assert both[1].distribution[k] == pytest.approx(single2[0].distribution[k], abs=1e-5)
    assert both[1].expected == pytest.approx(single2[0].expected, abs=1e-5)


def test_temperature_scales_logits():
    model = _model()
    model.temperature[CHOICE] = 2.0
    samples = [
        {
            "state": "s",
            "questions": [{"type": "choice", "instruction": "q", "candidates": ["a", "b"]}],
        }
    ]
    tok = HashTokenizer(vocab_size=512)
    inp = build_decision_inputs(samples, tok)
    with torch.no_grad():
        raw = model(
            state_ids=inp.state_ids,
            state_cu=inp.state_cu,
            question_ids=inp.question_ids,
            question_cu=inp.question_cu,
            question_state_index=inp.question_state_index,
            candidate_ids=inp.candidate_ids,
            candidate_cu=inp.candidate_cu,
            candidate_question_index=inp.candidate_question_index,
            primitive_index=inp.primitive_index,
            apply_temperature=False,
        )
        scaled = model(
            state_ids=inp.state_ids,
            state_cu=inp.state_cu,
            question_ids=inp.question_ids,
            question_cu=inp.question_cu,
            question_state_index=inp.question_state_index,
            candidate_ids=inp.candidate_ids,
            candidate_cu=inp.candidate_cu,
            candidate_question_index=inp.candidate_question_index,
            primitive_index=inp.primitive_index,
            apply_temperature=True,
        )
    assert torch.allclose(scaled.logits, raw.logits / 2.0, atol=1e-5)
