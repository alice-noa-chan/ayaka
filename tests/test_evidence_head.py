"""Feature-level invariants, not evidence of real-model accuracy improvement."""

import pytest
import torch

from ayaka.model.evidence import EvidenceResidualHead


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def sample():
    torch.manual_seed(71)
    return {
        "native_logits": torch.randn(2, 3, 4),
        "candidates": torch.randn(2, 3, 4, 8),
        "queries": torch.randn(2, 3, 8),
        "memory": torch.randn(2, 5, 8),
        "memory_mask": torch.ones(2, 5, dtype=torch.bool),
        "candidate_mask": torch.ones(2, 3, 4, dtype=torch.bool),
        "question_mask": torch.ones(2, 3, dtype=torch.bool),
        "primitive": torch.tensor([[0, 1, 2], [2, 0, 1]]),
    }


def make_head(**kwargs):
    return EvidenceResidualHead(8, dim=8, heads=2, **kwargs).eval()


def activate(head):
    # Simulate a learned final projection, so isolation tests cannot pass solely
    # because the initial correction is zero.
    with torch.no_grad():
        head.correction[-1].weight.copy_(torch.linspace(-0.3, 0.5, 8)[None])
    return head


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("primitive", [0, 1, 2])
def test_initial_native_logits_and_distribution_preserved(sample, dtype, primitive):
    sample["native_logits"] = sample["native_logits"].to(dtype)
    sample["primitive"].fill_(primitive)
    out = make_head()(**sample)
    assert out.logits.dtype == dtype
    assert torch.equal(out.logits, sample["native_logits"])
    assert torch.count_nonzero(out.correction) == 0
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert torch.equal(out.log_probs(), sample["native_logits"].to(work_dtype).log_softmax(-1))


def test_padding_nan_and_empty_record_are_safe(sample):
    sample["candidate_mask"][0, 1, 2:] = False
    sample["candidate_mask"][1] = False
    sample["question_mask"][1] = False
    sample["memory_mask"][1] = False
    sample["candidates"][~sample["candidate_mask"]] = torch.nan
    sample["native_logits"][~sample["candidate_mask"]] = torch.nan
    sample["queries"][1] = torch.nan
    sample["primitive"][1] = -99
    sample["memory"][1] = torch.nan
    out = activate(make_head())(**sample)
    assert torch.isfinite(out.logits[sample["candidate_mask"]]).all()
    assert torch.isneginf(out.logits[~sample["candidate_mask"]]).all()
    assert (out.probs()[~sample["candidate_mask"]] == 0).all()
    assert torch.allclose(out.probs().sum(-1), sample["question_mask"].float())
    assert torch.isfinite(out.correction).all()


def test_memory_changes_learned_evidence_readout(sample):
    head = activate(make_head())
    before = head(**sample).correction.clone()
    sample["memory"] *= -2
    assert not torch.allclose(head(**sample).correction, before)


def test_scoped_memory_blocks_other_trace_bank(sample):
    sample["memory_scope"] = torch.ones(2, 3, 5, dtype=torch.bool)
    sample["memory_scope"][:, 0, 2:] = False
    head = activate(make_head())
    before = head(**sample).logits.clone()
    sample["memory"][:, 2:] = torch.randn(2, 3, 8) * 7
    after = head(**sample).logits
    assert torch.equal(after[:, 0], before[:, 0])
    assert not torch.allclose(after[:, 1:], before[:, 1:])


def test_masked_memory_values_do_not_affect_output(sample):
    sample["memory_mask"][:, -2:] = False
    head = activate(make_head())
    before = head(**sample).logits.clone()
    sample["memory"][:, -2:] = torch.nan
    assert torch.equal(before, head(**sample).logits)


def test_default_questions_are_isolated(sample):
    head = activate(make_head(field_layers=2))
    before = head(**sample).logits.clone()
    sample["candidates"][:, 1:] *= -3
    sample["queries"][:, 1:] *= 5
    sample["primitive"][:, 1:] = 0
    assert torch.equal(before[:, 0], head(**sample).logits[:, 0])


def test_explicit_group_mixes_only_related_questions(sample):
    sample["group_ids"] = torch.tensor([[0, 0, 1], [0, 0, 1]])
    head = activate(make_head(field_layers=2))
    before = head(**sample).logits.clone()
    sample["queries"][:, 1] *= -4
    after = head(**sample).logits
    assert not torch.allclose(after[:, 0], before[:, 0])
    assert torch.equal(after[:, 2], before[:, 2])


def test_records_never_mix_even_with_same_group_ids(sample):
    sample["group_ids"] = torch.zeros(2, 3, dtype=torch.long)
    head = activate(make_head())
    before = head(**sample).logits[0].clone()
    for name in ("candidates", "queries", "memory", "native_logits"):
        sample[name][1] *= -8
    assert torch.equal(head(**sample).logits[0], before)


def test_candidate_permutation_equivariance_with_padding(sample):
    sample["candidate_mask"][:, :, -1] = False
    head = activate(make_head())
    before = head(**sample).logits
    perm = torch.tensor([3, 1, 0, 2])
    for name in ("native_logits", "candidates", "candidate_mask"):
        sample[name] = sample[name][:, :, perm]
    assert torch.allclose(head(**sample).logits, before[:, :, perm], atol=1e-6, rtol=1e-6)


def test_question_permutation_equivariance(sample):
    sample["group_ids"] = torch.tensor([[0, 0, 1], [0, 0, 1]])
    head = activate(make_head())
    before = head(**sample).logits
    perm = torch.tensor([2, 0, 1])
    for name in (
        "native_logits",
        "candidates",
        "queries",
        "candidate_mask",
        "question_mask",
        "primitive",
        "group_ids",
    ):
        sample[name] = sample[name][:, perm]
    assert torch.allclose(head(**sample).logits, before[:, perm], atol=1e-6, rtol=1e-6)


def test_first_step_learns_final_projection_second_step_reaches_evidence(sample):
    head = make_head()
    prior = sample["native_logits"].requires_grad_()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.5)
    for step in range(2):
        optimizer.zero_grad()
        loss = -head(**sample).log_probs()[..., 0].mean()
        loss.backward()
        assert head.correction[-1].weight.grad.abs().sum() > 0
        if step == 1:
            assert head.memory_projection.weight.grad.abs().sum() > 0
        assert prior.grad is None
        optimizer.step()
    assert not torch.equal(head(**sample).logits, prior)


def test_explicit_prior_gradient_opt_in(sample):
    sample["native_logits"].requires_grad_()
    out = make_head(detach_prior=False)(**sample)
    loss = -out.log_probs()[..., 0].mean()
    loss.backward()
    assert sample["native_logits"].grad.abs().sum() > 0


def test_lexical_features_require_explicit_output_head_input(sample):
    head = activate(make_head(lexical=True))
    with pytest.raises(ValueError, match="actual output-head"):
        head(**sample)
    sample["lexical"] = torch.randn_like(sample["candidates"])
    before = head(**sample).logits
    sample["lexical"] *= -5
    assert not torch.allclose(head(**sample).logits, before)
    with pytest.raises(ValueError, match="lexical=True"):
        make_head()(**sample)


def test_more_than_26_candidates_without_group_approximation(sample):
    sample["native_logits"] = torch.randn(2, 3, 40)
    sample["candidates"] = torch.randn(2, 3, 40, 8)
    sample["candidate_mask"] = torch.ones(2, 3, 40, dtype=torch.bool)
    out = make_head()(**sample)
    assert torch.equal(out.logits, sample["native_logits"])
    assert torch.allclose(out.probs().sum(-1), torch.ones(2, 3))


@pytest.mark.parametrize(
    "caps",
    [
        {"max_memory_tokens": 4},
        {"max_questions": 2},
        {"max_candidates": 3},
        {"max_total_candidates": 23},
    ],
)
def test_caps_fail_explicitly(sample, caps):
    with pytest.raises(ValueError, match="cap"):
        make_head(**caps)(**sample)


@pytest.mark.parametrize("name", ["native_logits", "candidates", "queries", "memory"])
def test_nonfinite_active_features_rejected(sample, name):
    sample[name].flatten()[0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        make_head()(**sample)


@pytest.mark.parametrize(
    "bad", ["mask_dtype", "mask_consistency", "primitive", "group", "scope", "shape"]
)
def test_invalid_contracts_rejected(sample, bad):
    if bad == "mask_dtype":
        sample["memory_mask"] = sample["memory_mask"].float()
    elif bad == "mask_consistency":
        sample["candidate_mask"][0, 0] = False
    elif bad == "primitive":
        sample["primitive"][0, 0] = 3
    elif bad == "group":
        sample["group_ids"] = torch.full((2, 3), -1)
    elif bad == "scope":
        sample["memory_scope"] = torch.zeros(2, 3, 5, dtype=torch.bool)
    else:
        sample["queries"] = sample["queries"][:, :2]
    with pytest.raises(ValueError):
        make_head()(**sample)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 7},
        {"heads": 0},
        {"routing_layers": -1},
        {"dropout": 1},
        {"lexical": 1},
        {"max_questions": True},
    ],
)
def test_invalid_head_configuration_rejected(kwargs):
    params = {"hidden": 8, "dim": 8, "heads": 2}
    params.update(kwargs)
    with pytest.raises(ValueError):
        EvidenceResidualHead(**params)


def test_routing_and_field_ablation(sample):
    out = make_head(routing_layers=0, field_layers=0)(**sample)
    assert torch.equal(out.logits, sample["native_logits"])


def test_feature_overflow_fails_instead_of_losing_native_prior(sample):
    sample["candidates"].fill_(1e10)
    with pytest.raises(ValueError, match="overflow head dtype"):
        make_head().half()(**sample)
