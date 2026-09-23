import torch

from ayaka.config import tiny_config
from ayaka.model.ragged import (
    block_cu_seqlens,
    chunk_cu_seqlens,
    gather_block_tokens,
    segment_mean,
    state_block_ranges,
)
from ayaka.model.state_encoder import StateEncoder


def test_chunk_short_segments_stay_whole():
    cu = torch.tensor([0, 10, 90, 300])
    out = chunk_cu_seqlens(cu, chunk_size=64, short_threshold=200)
    # seg0 (10) and seg1 (80) whole; seg2 (210) split into 4 chunks
    assert out.tolist() == [0, 10, 90, 154, 218, 282, 300]


def test_block_cu_seqlens_fixed_blocks():
    cu = torch.tensor([0, 70, 200])
    out = block_cu_seqlens(cu, block_size=64)
    assert out.tolist() == [0, 64, 70, 134, 198, 200]


def test_state_block_ranges():
    cu = torch.tensor([0, 70, 200])
    blk = block_cu_seqlens(cu, 64)  # 5 blocks
    sbc, bidx = state_block_ranges(cu, blk)
    assert sbc.tolist() == [0, 2, 5]
    assert bidx.tolist() == [0, 0, 1, 1, 1]


def test_segment_mean():
    x = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    cu = torch.tensor([0, 2, 6])
    m = segment_mean(x, cu)
    assert torch.allclose(m[0], x[:2].mean(0))
    assert torch.allclose(m[1], x[2:].mean(0))


def test_gather_block_tokens():
    hs = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    blk_cu = torch.tensor([0, 4, 7, 10])
    sel = torch.tensor([[0, 2], [1, 2]])
    toks, cu = gather_block_tokens(hs, blk_cu, sel)
    assert cu.tolist() == [0, 7, 13]
    assert torch.equal(toks[:4], hs[0:4])
    assert torch.equal(toks[4:7], hs[7:10])
    assert torch.equal(toks[7:10], hs[4:7])
    assert torch.equal(toks[10:13], hs[7:10])


def test_state_encoder_short_path():
    torch.manual_seed(0)
    cfg = tiny_config()
    enc = StateEncoder(cfg)
    ids = torch.randint(0, cfg.vocab_size, (30,))
    cu = torch.tensor([0, 12, 30])
    mem = enc(ids, cu)
    assert mem.hs.shape == (30, cfg.hidden)
    assert mem.blk_cu.tolist() == [0, 12, 30] or True  # blocks of 64 -> per-state
    assert mem.state_blk_cu[-1].item() == mem.bs.shape[0]
    assert not mem.route_tokens


def test_state_encoder_long_path_and_shapes():
    torch.manual_seed(0)
    cfg = tiny_config(block_size=8, short_context_threshold=16)
    enc = StateEncoder(cfg)
    # one long state (40 tokens > 16 threshold) + one short (10)
    ids = torch.randint(0, cfg.vocab_size, (50,))
    cu = torch.tensor([0, 40, 50])
    mem = enc(ids, cu)
    assert mem.route_tokens
    assert mem.hs.shape == (50, cfg.hidden)
    # blocks: state0 -> 5 blocks of 8; state1 -> 2 blocks (8, 2)
    assert mem.blk_cu.tolist() == [0, 8, 16, 24, 32, 40, 48, 50]
    assert mem.bs.shape == (7, cfg.hidden)
    assert mem.state_blk_cu.tolist() == [0, 5, 7]
    assert mem.block_state_index.tolist() == [0] * 5 + [1] * 2


def test_state_encoder_isolation_between_states():
    torch.manual_seed(0)
    cfg = tiny_config()
    enc = StateEncoder(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (20,))
    cu = torch.tensor([0, 10, 20])
    with torch.no_grad():
        mem1 = enc(ids, cu)
        ids2 = ids.clone()
        ids2[10:] = torch.randint(0, cfg.vocab_size, (10,))  # corrupt state 1
        mem2 = enc(ids2, cu)
    # state 0 tokens' memory must be unchanged
    assert torch.allclose(mem1.hs[:10], mem2.hs[:10], atol=1e-5)
