import pytest
import torch

from ayaka.config import tiny_config
from ayaka.model.decision import AyakaDecisionModel
from ayaka.model.fastpath import prefill_last


@pytest.mark.parametrize("padded", [False, True])
def test_native_prefill_matches_full_forward_and_next_cached_step(padded):
    torch.manual_seed(119)
    model = AyakaDecisionModel.from_config(tiny_config(), dtype=torch.float32, device="cpu")
    model.eval()
    text = model.text_model()
    ids = torch.randint(15, 100, (2, 17))
    mask = torch.ones_like(ids)
    if padded:
        ids[1, :6] = 0
        mask[1, :6] = 0
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        full = text(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True)
        hidden, cache = prefill_last(text, ids, mask, positions)
        torch.testing.assert_close(hidden, full.last_hidden_state[:, -1, :], atol=1e-5, rtol=1e-5)
        new = torch.tensor([[22], [35]])
        nextmask = torch.cat([mask, torch.ones((2, 1), dtype=mask.dtype)], dim=1)
        nextpos = (nextmask.sum(-1) - 1)[:, None]
        expected = text(
            input_ids=new,
            attention_mask=nextmask,
            position_ids=nextpos,
            past_key_values=full.past_key_values,
            use_cache=True,
        )
        actual = text(
            input_ids=new,
            attention_mask=nextmask,
            position_ids=nextpos,
            past_key_values=cache,
            use_cache=True,
        )
        torch.testing.assert_close(
            actual.last_hidden_state, expected.last_hidden_state, atol=1e-5, rtol=1e-5
        )
