"""Paired uncertainty estimates for frozen WAL-TAT audit windows."""
from __future__ import annotations

from collections.abc import Sequence

import torch


def paired_block_bootstrap_nll(
    baseline_nll_sums: Sequence[float],
    candidate_nll_sums: Sequence[float],
    token_counts: Sequence[int],
    *,
    gate_ratio: float,
    samples: int = 4096,
    block_size: int = 8,
    confidence: float = 0.95,
    seed: int = 109,
) -> dict:
    """Estimate paired NLL uncertainty with a circular moving-block bootstrap.

    Adjacent audit windows are deliberately resampled together because the
    frozen suites are contiguous token windows rather than IID documents.
    Every replicate uses the same indices for BF16 and candidate losses.
    """

    baseline = torch.as_tensor(baseline_nll_sums, dtype=torch.float64)
    candidate = torch.as_tensor(candidate_nll_sums, dtype=torch.float64)
    tokens = torch.as_tensor(token_counts, dtype=torch.float64)
    if baseline.ndim != 1 or candidate.shape != baseline.shape or tokens.shape != baseline.shape:
        raise ValueError("baseline, candidate, and token counts must be matching vectors")
    if baseline.numel() < 2:
        raise ValueError("at least two paired windows are required")
    if not torch.isfinite(baseline).all() or not torch.isfinite(candidate).all():
        raise ValueError("NLL sums must be finite")
    if (baseline <= 0).any() or (candidate <= 0).any() or (tokens <= 0).any():
        raise ValueError("NLL sums and token counts must be positive")
    if gate_ratio <= 0:
        raise ValueError("gate_ratio must be positive")
    if samples < 2:
        raise ValueError("samples must be at least two")
    if not 1 <= block_size <= baseline.numel():
        raise ValueError("block_size must be within the paired window count")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    window_count = int(baseline.numel())
    blocks_per_sample = (window_count + block_size - 1) // block_size
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    starts = torch.randint(
        window_count,
        (samples, blocks_per_sample),
        generator=generator,
    )
    offsets = torch.arange(block_size)
    indices = (starts.unsqueeze(-1) + offsets) % window_count
    indices = indices.reshape(samples, -1)[:, :window_count]

    baseline_boot = baseline[indices].sum(dim=1)
    candidate_boot = candidate[indices].sum(dim=1)
    tokens_boot = tokens[indices].sum(dim=1)
    ratios = candidate_boot / baseline_boot
    deltas = (candidate_boot - baseline_boot) / tokens_boot
    alpha = (1.0 - confidence) / 2.0
    quantiles = torch.tensor([alpha, 0.5, 1.0 - alpha], dtype=torch.float64)
    ratio_ci = torch.quantile(ratios, quantiles)
    delta_ci = torch.quantile(deltas, quantiles)

    baseline_total = baseline.sum()
    candidate_total = candidate.sum()
    token_total = tokens.sum()
    observed_ratio = candidate_total / baseline_total
    observed_delta = (candidate_total - baseline_total) / token_total
    per_window_delta = (candidate - baseline) / tokens
    upper_ratio = float(ratio_ci[2].item())
    return {
        "windows": window_count,
        "tokens": int(token_total.item()),
        "bootstrap_samples": int(samples),
        "block_size_windows": int(block_size),
        "confidence": float(confidence),
        "seed": int(seed),
        "observed_baseline_nll": float((baseline_total / token_total).item()),
        "observed_candidate_nll": float((candidate_total / token_total).item()),
        "observed_delta_nll": float(observed_delta.item()),
        "observed_ratio": float(observed_ratio.item()),
        "paired_window_win_rate": float((per_window_delta <= 0).double().mean().item()),
        "delta_nll_ci": {
            "lower": float(delta_ci[0].item()),
            "median": float(delta_ci[1].item()),
            "upper": float(delta_ci[2].item()),
        },
        "ratio_ci": {
            "lower": float(ratio_ci[0].item()),
            "median": float(ratio_ci[1].item()),
            "upper": upper_ratio,
        },
        "gate_ratio": float(gate_ratio),
        "point_passed": bool(observed_ratio.item() <= gate_ratio),
        "confidence_passed": bool(upper_ratio <= gate_ratio),
        "upper_ratio_margin": float(gate_ratio - upper_ratio),
    }
