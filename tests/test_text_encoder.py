import torch

from ayaka.config import tiny_config
from ayaka.model.state_encoder import StateEncoder
from ayaka.model.text_encoder import TextEncoder


def test_text_encoder_question_and_candidate_latents():
    torch.manual_seed(0)
    cfg = tiny_config()
    state_enc = StateEncoder(cfg)
    enc = TextEncoder(cfg, state_enc.embed)

    q_ids = torch.randint(0, cfg.vocab_size, (10,))
    q_cu = torch.tensor([0, 4, 10])  # 2 questions
    rq = enc(q_ids, q_cu, "q")
    assert rq.shape == (2, cfg.question_latents, cfg.hidden)

    c_ids = torch.randint(0, cfg.vocab_size, (15,))
    c_cu = torch.tensor([0, 5, 9, 15])  # 3 candidates
    rc = enc(c_ids, c_cu, "c")
    assert rc.shape == (3, cfg.candidate_latents, cfg.hidden)


def test_text_encoder_item_isolation():
    torch.manual_seed(0)
    cfg = tiny_config()
    enc = TextEncoder(cfg, StateEncoder(cfg).embed).eval()
    ids = torch.randint(0, cfg.vocab_size, (12,))
    cu = torch.tensor([0, 5, 12])
    with torch.no_grad():
        r1 = enc(ids, cu, "q")
        ids2 = ids.clone()
        ids2[5:] = torch.randint(0, cfg.vocab_size, (7,))
        r2 = enc(ids2, cu, "q")
    assert torch.allclose(r1[0], r2[0], atol=1e-5)


def test_shared_embedding_table():
    cfg = tiny_config()
    state_enc = StateEncoder(cfg)
    enc = TextEncoder(cfg, state_enc.embed)
    assert enc.embed is state_enc.embed
