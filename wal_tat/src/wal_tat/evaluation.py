"""Small deterministic before/after evaluator for causal language models."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import torch


@dataclass(frozen=True)
class LossMetrics:
    nll: float
    perplexity: float
    predicted_tokens: int


@torch.inference_mode()
def evaluate_causal_lm(
    model,
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    device: Optional[torch.device | str] = None,
) -> LossMetrics:
    """Evaluate an HF-style causal LM on already-tokenized frozen batches."""
    total_nll = 0.0
    total_tokens = 0
    was_training = model.training
    model.eval()
    try:
        for original in batches:
            batch = {
                key: value.to(device) if device is not None else value
                for key, value in original.items()
            }
            labels = batch.get("labels")
            if labels is None:
                labels = batch["input_ids"].clone()
                batch["labels"] = labels
            predicted = labels[..., 1:]
            token_count = int((predicted != -100).sum().item())
            if token_count == 0:
                continue
            output = model(**batch)
            loss = output.loss if hasattr(output, "loss") else output[0]
            total_nll += float(loss.detach().float().item()) * token_count
            total_tokens += token_count
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("evaluation batches contain no predicted tokens")
    mean_nll = total_nll / total_tokens
    return LossMetrics(mean_nll, math.exp(mean_nll), total_tokens)
