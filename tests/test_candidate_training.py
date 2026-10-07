from dataclasses import replace

import pytest
import torch
from test_candidates import request, service

from ayaka.checkpoint import apply_lora
from ayaka.config import tiny_config
from ayaka.data.candidate_v2 import (
    candidate_curriculum,
    finite_partition_audit,
    partition_diagnostics,
)
from ayaka.model.decision import AyakaDecisionModel
from ayaka.tokenization import ToyTokenizer
from ayaka.training.candidates import proposal_items
from ayaka.training.trainer import TrainConfig, Trainer


def test_exact_audit_detects_synonyms_overlap_parent_escape_and_residual():
    result = finite_partition_audit(
        ["a", "b", "c"], {"x": ["a"], "y": ["a", "b"], "synonym": ["a"]}, ["a", "c"]
    )
    assert result["equivalent_pairs"] == [["x", "synonym"]]
    assert len(result["overlap_pairs"]) == 3
    assert result["outside_parent"] == ["b"] and result["residual"] == ["c"]
    assert result["explicit_coverage"] == 0.5
    with pytest.raises(ValueError, match="unknown"):
        finite_partition_audit(["a"], {"x": ["outside"]})


def test_proposal_ce_uses_separate_context_and_reaches_lora_without_weight_updates():
    torch.set_num_threads(1)
    cfg, tok = tiny_config(version=2, max_seq_len=4096), ToyTokenizer()
    sample = candidate_curriculum("train", 1)[0]
    item = proposal_items(sample, tok, cfg)[0]
    assert item.proposal_positions[0] > 0
    assert item.proposal_input_ids != item.enc.prefix_ids + item.enc.rendered.suffix_ids
    model = AyakaDecisionModel.from_config(cfg, dtype=torch.float32)
    model.backbone.requires_grad_(False)
    apply_lora(model)
    trainer = Trainer(model, tok, TrainConfig(bf16=False), "cpu")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    out, tensors = trainer._forward("rows", [item])
    assert tensors.proposal_ce > 0 and torch.isfinite(out.logits).all()
    tensors.proposal_ce.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if "lora_B" in n
    )
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    with pytest.raises(ValueError, match="do not truncate"):
        proposal_items(sample, tok, replace(cfg, max_seq_len=32))


def test_training_rejects_invalid_semantic_partition():
    sample = candidate_curriculum("train", 1)[0]
    sample.metadata["proposal_supervision"]["memberships"]["b"] = sample.metadata[
        "proposal_supervision"
    ]["memberships"]["a"]
    with pytest.raises(ValueError, match="invalid partition"):
        proposal_items(sample, ToyTokenizer(), tiny_config(max_seq_len=4096))


def test_held_out_finite_vocab_and_diagnostics_targets():
    from ayaka.data.reasoning_v2 import SPLITS

    previous = set()
    for split in SPLITS:
        samples = candidate_curriculum(split, 16)
        vocab = {v for s in samples for v in s.metadata["proposal_supervision"]["universe"]}
        assert not previous & vocab
        previous |= vocab
        diagnostics = [partition_diagnostics(s) for s in samples]
        for name in ("overlap", "coverage", "sufficiency"):
            assert {
                q.target_distribution["true"]
                for s in diagnostics
                for q in s.questions
                if q.id == name
            } == {0, 1}


def test_empty_task_fails_before_scoring_or_proposing():
    from ayaka.serve import BadRequest

    server, original = service()
    body = request("open")
    body["questions"]["intent"]["instructions"] = " "
    with pytest.raises(BadRequest, match="explicit instructions"):
        server.handle(body)
    assert not original.calls and not server.decision.generator.budgets


@pytest.mark.parametrize("media", [None, []])
def test_unsupported_explicit_media_never_spends_proposal_tokens(media):
    from ayaka.serve import BadRequest

    server, original = service()
    body = request("open")
    body["media"] = media
    with pytest.raises(BadRequest, match="text states only"):
        server.handle(body)
    assert not original.calls and not server.decision.generator.budgets
