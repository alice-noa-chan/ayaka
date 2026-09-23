import torch

from ayaka.config import tiny_config
from ayaka.model.evidence import EvidenceCrossAttention
from ayaka.model.router import BlockRouter, flatten_selection


def _setup(n_blocks=6, d=64):
    cfg = tiny_config(hidden=d, block_topk=3)
    router = BlockRouter(cfg)
    bs = torch.randn(n_blocks, d)
    block_state_index = torch.tensor([0, 0, 0, 1, 1, 1])
    return cfg, router, bs, block_state_index


def test_router_probs_restricted_to_own_state():
    cfg, router, bs, bidx = _setup()
    q = torch.randn(2, bs.shape[1])
    item_state = torch.tensor([0, 1])
    out = router(q, bs, item_state, bidx)
    probs = out["route_probs"]
    assert probs.shape == (2, 6)
    assert torch.allclose(probs[0, :3].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.equal(probs[0, 3:], torch.zeros(3))
    assert torch.allclose(probs[1, 3:].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.equal(probs[1, :3], torch.zeros(3))


def test_router_topk_selection_and_validity():
    cfg, router, bs, bidx = _setup(n_blocks=4)
    bidx = torch.tensor([0, 0, 1, 1])  # 2 blocks per state
    q = torch.randn(2, bs.shape[1])
    item_state = torch.tensor([0, 1])
    out = router(q, bs, item_state, bidx)
    sel, valid = out["selected"], out["valid"]
    assert sel.shape == (2, 3) and valid.shape == (2, 3)
    # only 2 real blocks per state -> 2 valid picks each
    assert valid.sum(dim=1).tolist() == [2, 2]
    flat, counts = flatten_selection(sel, valid)
    assert counts.tolist() == [2, 2]
    assert flat.numel() == 4
    for i in range(2):
        blocks = set(sel[i][valid[i]].tolist())
        assert blocks <= {0, 1} if i == 0 else blocks <= {2, 3}


def test_evidence_cross_attention_shapes_and_isolation():
    torch.manual_seed(0)
    cfg = tiny_config(cross_attn_blocks=2)
    eca = EvidenceCrossAttention(cfg).eval()
    m = cfg.candidate_latents
    latents = torch.randn(3, m, cfg.hidden)  # 3 candidates
    memory = torch.randn(20, cfg.hidden)
    mem_cu = torch.tensor([0, 8, 14, 20])  # per-candidate memory
    out = eca(latents, memory, mem_cu)
    assert out.shape == (3, m, cfg.hidden)
    # candidate 0 unaffected by other segments' memory
    mem2 = memory.clone()
    mem2[8:] = 100.0
    out2 = eca(latents, mem2, mem_cu)
    assert torch.allclose(out[0], out2[0], atol=1e-5)
