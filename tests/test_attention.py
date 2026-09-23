import torch
import torch.nn.functional as F

from ayaka.model.attention import varlen_attention
from ayaka.model.blocks import SelfAttention, TransformerBlock
from ayaka.model.rope import RotaryEmbedding, rope_positions


def _ref_sdpa(q, k, v, cu_q, cu_k):
    """Per-segment reference implementation."""
    out = torch.empty_like(q)
    for i in range(cu_q.numel() - 1):
        qs, qe = int(cu_q[i]), int(cu_q[i + 1])
        ks, ke = int(cu_k[i]), int(cu_k[i + 1])
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)
        ki = k[ks:ke].transpose(0, 1).unsqueeze(0)
        vi = v[ks:ke].transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=False)
        out[qs:qe] = oi.transpose(1, 2).squeeze(0)
    return out


def test_varlen_attention_matches_segmentwise_sdpa():
    torch.manual_seed(0)
    t, h, dh = 11, 2, 8
    q, k, v = torch.randn(t, h, dh), torch.randn(t, h, dh), torch.randn(t, h, dh)
    cu = torch.tensor([0, 3, 11])
    out = varlen_attention(q, k, v, cu, cu)
    ref = _ref_sdpa(q, k, v, cu, cu)
    assert torch.allclose(out, ref, atol=1e-5)


def test_varlen_attention_no_cross_segment_leak():
    torch.manual_seed(0)
    t, h, dh = 8, 2, 4
    q = torch.randn(t, h, dh)
    k, v = torch.randn(t, h, dh), torch.randn(t, h, dh)
    cu = torch.tensor([0, 3, 8])
    out = varlen_attention(q, k, v, cu, cu)
    # corrupt segment 1's keys/values entirely; segment 0 output must not move
    k2, v2 = k.clone(), v.clone()
    k2[3:], v2[3:] = 100.0, -100.0
    out2 = varlen_attention(q, k2, v2, cu, cu)
    assert torch.allclose(out[:3], out2[:3], atol=1e-5)


def test_cross_attention_varlen():
    torch.manual_seed(0)
    q = torch.randn(6, 2, 4)  # two ragged query segments
    kv = torch.randn(9, 2, 4)
    cu_q = torch.tensor([0, 2, 6])
    cu_k = torch.tensor([0, 5, 9])
    out = varlen_attention(q, kv, kv, cu_q, cu_k)
    assert out.shape == (6, 2, 4)
    ref = _ref_sdpa(q, kv, kv, cu_q, cu_k)
    assert torch.allclose(out, ref, atol=1e-5)


def test_rope_positions_reset_per_segment():
    cu = torch.tensor([0, 3, 7])
    pos = rope_positions(cu)
    assert pos.tolist() == [0, 1, 2, 0, 1, 2, 3]


def test_self_attention_and_block_shapes():
    torch.manual_seed(0)
    d, heads = 32, 4
    attn = SelfAttention(d, heads)
    block = TransformerBlock(d, heads, ffn=64, resid_dropout=0.0)
    rope = RotaryEmbedding(d // heads)
    x = torch.randn(9, d)
    cu = torch.tensor([0, 4, 9])
    assert attn(x, cu, rope).shape == (9, d)
    assert block(x, cu, rope).shape == (9, d)
