"""Probability-only prediction should not copy unused candidate logits to the host."""

from dataclasses import replace

import pytest
import torch
from test_training_reductions import _trainer

from ayaka.config import tiny_config
from ayaka.data.schema import Candidate, Question, Sample
from ayaka.losses import LossWeights
from ayaka.tokenization import ToyTokenizer
from ayaka.training.batching import sample_to_items


@pytest.mark.parametrize("return_logits", [False, True])
def test_predict_only_transfers_logits_when_requested(monkeypatch, return_logits):
    tok = ToyTokenizer()
    samples = [
        Sample(
            "Evidence",
            [
                Question(
                    "q", "choice", "Choose", [Candidate("a", "A"), Candidate("b", "B")], {"a": 1.0}
                )
            ],
        )
    ]
    items = [item for sample in samples for item in sample_to_items(sample, tok, tiny_config())]
    trainer = _trainer(tok, items, LossWeights())
    trainer.model.logits.data = torch.tensor([-2.0, 3.0], dtype=torch.float64)
    original_forward = trainer._forward
    outputs, transferred = [], []

    def forward(*args, **kwargs):
        # This fixture isolates predict from transformer execution.
        result = original_forward(args[0], args[1])
        outputs.append(result[0].logits.float())
        return replace(result[0], logits=outputs[-1]), result[1]

    original_tolist = torch.Tensor.tolist

    def record_tolist(tensor):
        if any(tensor is logits for logits in outputs):
            transferred.append(tensor)
        return original_tolist(tensor)

    monkeypatch.setattr(trainer, "_forward", forward)
    monkeypatch.setattr(torch.Tensor, "tolist", record_tolist)
    result = trainer.predict(items, return_logits=return_logits)
    assert len(transferred) == int(return_logits)
    probabilities = result[0] if return_logits else result
    assert probabilities[0] == pytest.approx(torch.softmax(torch.tensor([-2.0, 3.0]), 0).tolist())
    if return_logits:
        assert result[1] == [[-2.0, 3.0]]
