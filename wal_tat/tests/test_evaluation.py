from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from wal_tat import evaluate_causal_lm


class FixedLossModel(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.loss = loss

    def forward(self, **_batch):
        return SimpleNamespace(loss=self.anchor * 0 + self.loss)


def test_evaluator_reports_nll_ppl_and_token_count():
    model = FixedLossModel(2.0)
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    metrics = evaluate_causal_lm(model, [batch])
    assert metrics.nll == 2.0
    assert metrics.perplexity == pytest.approx(7.389056)
    assert metrics.predicted_tokens == 3
