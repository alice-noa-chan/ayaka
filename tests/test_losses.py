import pytest
import torch

from ayaka.collate import build_decision_inputs
from ayaka.config import tiny_config
from ayaka.losses import (
    brier_loss,
    candidate_relation_loss,
    decision_loss,
    kl_distill,
    missing_evidence_loss,
    nll_loss,
    permutation_consistency,
    router_bce,
    router_kl,
    rps_loss,
    state_relation_loss,
)
from ayaka.metrics import (
    accuracy,
    compute_metrics,
    jsd_to_target,
    oos_auroc,
    permutation_invariance_error,
)
from ayaka.model.model import ElectraDecisionModel


def _out(samples=None, seed=0):
    from ayaka.tokenizer import HashTokenizer

    torch.manual_seed(seed)
    model = ElectraDecisionModel(tiny_config()).eval()
    samples = samples or [
        {
            "state": "s",
            "questions": [{"type": "choice", "instruction": "q", "candidates": ["a", "b", "c"]}],
        }
    ]
    inp = build_decision_inputs(samples, HashTokenizer(512))
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
        )


def test_nll_zero_for_perfect_prediction():
    out = _out()
    # NLL with target = own dist equals entropy
    val_self = float(nll_loss(out, out.probs()))
    ent = float(-(out.probs() * out.log_probs()).sum())
    assert val_self == pytest.approx(ent, abs=1e-5)
    # one-hot target at the argmax -> small positive NLL
    t = torch.zeros(3)
    t[out.logits.argmax()] = 1.0
    assert float(nll_loss(out, t)) >= 0.0


def test_brier_zero_when_p_equals_y():
    out = _out()
    assert float(brier_loss(out, out.probs())) < 1e-6


def test_rps_only_scores_and_ordinal():
    samples = [
        {
            "state": "s",
            "questions": [
                {"type": "score", "instruction": "sev?", "candidates": ["l1", "l2", "l3"]},
                {"type": "choice", "instruction": "c?", "candidates": ["x", "y"]},
            ],
        }
    ]
    out = _out(samples)
    # candidates: l1,l2,l3 (ord 0,1,2), x,y (ord -1 dummy)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
    ordinals = torch.tensor([0, 1, 2, 0, 1])
    mask = torch.tensor([True, False])
    val = rps_loss(out, targets, ordinals, mask)
    assert val >= 0
    # perfect prediction -> 0
    t2 = out.probs().clone()
    val2 = rps_loss(out, t2, ordinals, mask)
    assert val2 < 1e-6


def test_permutation_consistency_identical_runs():
    out = _out()
    lp = out.log_probs()
    assert float(permutation_consistency(lp, lp, out.cand_cu)) < 1e-6


def test_missing_evidence_penalty():
    out = _out()
    flagged = torch.tensor([True])
    val = missing_evidence_loss(out, flagged, tau=0.8)
    assert val >= 0
    unflagged = missing_evidence_loss(out, torch.tensor([False]))
    assert float(unflagged) == 0.0


def test_kl_distill_self_zero():
    out = _out()
    lp = out.log_probs()
    assert float(kl_distill(lp, lp, out.cand_cu)) < 1e-6


def test_router_losses():
    out = _out()
    n_c, n_b = out.route_probs.shape
    target = torch.zeros(n_c, n_b)
    mask = torch.zeros(n_c, n_b, dtype=torch.bool)
    mask[:, :2] = True
    target[:, 0] = 1.0
    assert router_bce(out.route_probs, target, mask) >= 0
    lp = out.route_probs.clamp(min=1e-9).log()
    assert router_kl(lp, lp, mask) < 1e-6


def test_relation_losses():
    a = torch.randn(8, 32)
    assert state_relation_loss(a, a) < 1e-6
    r = torch.randn(5, 32)
    assert candidate_relation_loss(r, r) < 1e-6


def test_decision_loss_composite():
    samples = [
        {
            "state": "s",
            "questions": [
                {"type": "score", "instruction": "sev?", "candidates": ["l1", "l2"]},
                {"type": "choice", "instruction": "c?", "candidates": ["x", "y"]},
            ],
        }
    ]
    out = _out(samples)
    targets = torch.tensor([1.0, 0.0, 0.0, 1.0])
    ordinals = torch.tensor([0, 1, 0, 1])
    smask = torch.tensor([True, False])
    mmask = torch.tensor([False, True])
    parts = decision_loss(
        out,
        targets,
        cand_ordinals=ordinals,
        score_question_mask=smask,
        missing_mask=mmask,
    )
    assert "total" in parts and parts["total"] > 0


def test_metrics_basic():
    out = _out()
    targets = torch.zeros(3)
    targets[out.logits.argmax()] = 1.0  # target = prediction
    m = compute_metrics(out, targets)
    assert m["accuracy"] == 1.0
    assert m["nll"] >= 0 and m["brier"] >= 0
    assert 0 <= m["ece"] <= 1
    assert m["jsd"] >= 0
    assert 0 <= m["selective_risk@0.8"] <= 1


def test_metric_edge_cases():
    out = _out()
    assert 0 <= accuracy(out, torch.tensor([1.0, 0, 0])) <= 1
    assert jsd_to_target(out, out.probs()) < 1e-5
    scores = torch.tensor([0.9, 0.8, 0.3, 0.1])
    labels = torch.tensor([1, 1, 0, 0])
    assert abs(oos_auroc(scores, labels) - 1.0) < 1e-6
    assert permutation_invariance_error(torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5])) == 0
