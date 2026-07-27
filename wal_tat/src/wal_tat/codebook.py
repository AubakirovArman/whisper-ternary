"""Non-uniform column codebooks for ultra-low-bit reference experiments.

These primitives deliberately optimize representation quality, not runtime.
They mirror the column-wise K-means baseline used by recent low-bit Whisper
work and account for codebooks and the mixed-format mask in physical-bpw
figures.  A production kernel may choose a different layout after the quality
frontier is established.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def column_codebook_payload_bits(
    *,
    out_features: int,
    in_features: int,
    code_bits: int,
    centroid_bits: int = 16,
) -> int:
    """Return codes plus one dense codebook per input column."""
    if out_features < 1 or in_features < 1:
        raise ValueError("matrix dimensions must be positive")
    if code_bits < 1 or code_bits > 8:
        raise ValueError("code_bits must be in [1, 8]")
    if centroid_bits < 1:
        raise ValueError("centroid_bits must be positive")
    levels = 1 << code_bits
    return (
        out_features * in_features * code_bits
        + in_features * levels * centroid_bits
    )


def column_codebook_physical_bpw(
    *,
    out_features: int,
    code_bits: int,
    centroid_bits: int = 16,
) -> float:
    """Physical bpw for column codes and their dense centroid tables."""
    return code_bits + (1 << code_bits) * centroid_bits / out_features


def mixed_column_codebook_payload_bits(
    q4_mask: torch.Tensor,
    *,
    out_features: int,
    q2_bits: int = 2,
    q4_bits: int = 4,
    centroid_bits: int = 16,
    include_mask: bool = True,
) -> int:
    """Return exact payload bits for a column-wise Q2/Q4 representation."""
    if q4_mask.ndim != 1 or q4_mask.dtype != torch.bool:
        raise ValueError("q4_mask must be a one-dimensional bool tensor")
    if out_features < 1:
        raise ValueError("out_features must be positive")
    if not 0 < q2_bits < q4_bits <= 8:
        raise ValueError("expected 0 < q2_bits < q4_bits <= 8")
    if centroid_bits < 1:
        raise ValueError("centroid_bits must be positive")
    q4_columns = int(q4_mask.sum().item())
    q2_columns = q4_mask.numel() - q4_columns
    code_bits = out_features * (
        q2_columns * q2_bits + q4_columns * q4_bits
    )
    codebook_bits = centroid_bits * (
        q2_columns * (1 << q2_bits) + q4_columns * (1 << q4_bits)
    )
    mask_bits = q4_mask.numel() if include_mask else 0
    return code_bits + codebook_bits + mask_bits


def _weighted_1d_kmeans(
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    levels: int,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run deterministic batched weighted one-dimensional K-means.

    ``values`` and ``weights`` have shape ``[units, samples]``.  Quantile
    initialization avoids random-seed dependence, while empty clusters retain
    their previous centroid.
    """
    if values.ndim != 2 or weights.shape != values.shape:
        raise ValueError("values and weights must be matching matrices")
    if levels < 2 or levels > 256:
        raise ValueError("levels must be in [2, 256]")
    if values.shape[1] < levels:
        raise ValueError("the number of samples must cover every codebook level")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if torch.any(weights < 0):
        raise ValueError("weights must be non-negative")

    sorted_values = values.sort(dim=1).values
    positions = (
        (torch.arange(levels, device=values.device, dtype=torch.float32) + 0.5)
        * values.shape[1]
        / levels
    ).floor().long().clamp_max(values.shape[1] - 1)
    centroids = sorted_values.index_select(1, positions)

    for _ in range(iterations):
        distance = (values.unsqueeze(-1) - centroids.unsqueeze(1)).square()
        codes = distance.argmin(-1)
        weighted_values = values * weights
        numerator = torch.zeros_like(centroids)
        denominator = torch.zeros_like(centroids)
        numerator.scatter_add_(1, codes, weighted_values)
        denominator.scatter_add_(1, codes, weights)
        updated = numerator / denominator.clamp_min(1e-20)
        centroids = torch.where(denominator > 0, updated, centroids)
        centroids = centroids.sort(dim=1).values

    distance = (values.unsqueeze(-1) - centroids.unsqueeze(1)).square()
    codes = distance.argmin(-1)
    reconstructed = centroids.gather(1, codes)
    error = (weights * (values - reconstructed).square()).sum(1)
    return codes.to(torch.int16), centroids, error


@torch.no_grad()
def weighted_column_codebook_project(
    weight: torch.Tensor,
    input_second_moment: torch.Tensor,
    *,
    bits: int = 2,
    output_importance: Optional[torch.Tensor] = None,
    iterations: int = 12,
    chunk_columns: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project each input column into an independent learned codebook.

    The returned codes retain the original ``[out, in]`` matrix layout,
    centroids have shape ``[in, 2**bits]``, and error is reported per input
    column under the diagonal activation/output-importance metric.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    if bits not in {2, 4}:
        raise ValueError("only Q2 and Q4 column codebooks are supported")
    if input_second_moment.ndim != 1 or input_second_moment.numel() != weight.shape[1]:
        raise ValueError("input_second_moment must match input features")
    if chunk_columns < 1:
        raise ValueError("chunk_columns must be positive")
    if output_importance is None:
        output_weight = torch.ones(
            weight.shape[0], device=weight.device, dtype=torch.float32
        )
    else:
        if output_importance.ndim != 1 or output_importance.numel() != weight.shape[0]:
            raise ValueError("output_importance must match output features")
        output_weight = output_importance.detach().to(
            device=weight.device, dtype=torch.float32
        ).clamp_min(0)

    values = weight.detach().float().transpose(0, 1).contiguous()
    input_weight = input_second_moment.detach().to(
        device=weight.device, dtype=torch.float32
    ).clamp_min(0)
    code_chunks = []
    centroid_chunks = []
    error_chunks = []
    levels = 1 << bits
    for start in range(0, values.shape[0], chunk_columns):
        chunk = values[start : start + chunk_columns]
        weights = (
            input_weight[start : start + chunk_columns].unsqueeze(1)
            * output_weight.unsqueeze(0)
        ).expand_as(chunk)
        codes, centroids, error = _weighted_1d_kmeans(
            chunk,
            weights,
            levels=levels,
            iterations=iterations,
        )
        code_chunks.append(codes)
        centroid_chunks.append(centroids)
        error_chunks.append(error)

    column_codes = torch.cat(code_chunks, dim=0)
    code_dtype = torch.int8 if bits <= 4 else torch.int16
    return (
        column_codes.transpose(0, 1).contiguous().to(code_dtype),
        torch.cat(centroid_chunks, dim=0),
        torch.cat(error_chunks, dim=0),
    )


def reconstruct_column_codebook(
    codes: torch.Tensor, centroids: torch.Tensor
) -> torch.Tensor:
    """Materialize a reference weight matrix from column codebooks."""
    if codes.ndim != 2 or centroids.ndim != 2:
        raise ValueError("codes and centroids must be matrices")
    if codes.shape[1] != centroids.shape[0]:
        raise ValueError("one codebook is required per input column")
    column_codes = codes.long().transpose(0, 1)
    if column_codes.numel() and (
        int(column_codes.min()) < 0
        or int(column_codes.max()) >= centroids.shape[1]
    ):
        raise ValueError("code is outside the centroid table")
    return centroids.gather(1, column_codes).transpose(0, 1).contiguous()


def column_outlier_density(
    weight: torch.Tensor, *, threshold_multiplier: float = 13.0
) -> torch.Tensor:
    """Measure the fraction of large-magnitude weights in each column."""
    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    if threshold_multiplier <= 0:
        raise ValueError("threshold_multiplier must be positive")
    value = weight.detach().float().abs()
    threshold = (
        value.mean(0).clamp_min(torch.finfo(value.dtype).tiny)
        * float(threshold_multiplier)
    )
    return (value > threshold.unsqueeze(0)).float().mean(0)


@dataclass(frozen=True)
class MixedColumnCodebookProjection:
    codes: torch.Tensor
    q2_centroids: torch.Tensor
    q4_centroids: torch.Tensor
    q4_mask: torch.Tensor
    column_error: torch.Tensor
    payload_bits: int
    physical_bpw: float
    selection: str

    def effective_weight(self) -> torch.Tensor:
        q2 = reconstruct_column_codebook(self.codes.clamp_max(3), self.q2_centroids)
        q4 = reconstruct_column_codebook(self.codes, self.q4_centroids)
        return torch.where(self.q4_mask.unsqueeze(0), q4, q2)


@torch.no_grad()
def mixed_column_codebook_project(
    weight: torch.Tensor,
    input_second_moment: torch.Tensor,
    *,
    q4_fraction: float = 0.05,
    output_importance: Optional[torch.Tensor] = None,
    outlier_threshold_multiplier: float = 13.0,
    iterations: int = 12,
    chunk_columns: int = 128,
    centroid_bits: int = 16,
    selection: str = "outlier",
) -> MixedColumnCodebookProjection:
    """Use Q4 on selected columns and learned Q2 elsewhere.

    ``outlier`` reproduces the inexpensive column-density heuristic from
    ultra-low-bit Whisper PTQ.  ``error_gain`` is a calibration-aware
    rate--distortion oracle: it promotes columns with the largest measured
    Q2-to-Q4 reduction under the declared diagonal metric.
    """
    if not 0.0 <= q4_fraction <= 1.0:
        raise ValueError("q4_fraction must be in [0, 1]")
    if selection not in {"outlier", "error_gain"}:
        raise ValueError("selection must be outlier or error_gain")
    q2_codes, q2_centroids, q2_error = weighted_column_codebook_project(
        weight,
        input_second_moment,
        bits=2,
        output_importance=output_importance,
        iterations=iterations,
        chunk_columns=chunk_columns,
    )
    q4_codes, q4_centroids, q4_error = weighted_column_codebook_project(
        weight,
        input_second_moment,
        bits=4,
        output_importance=output_importance,
        iterations=iterations,
        chunk_columns=chunk_columns,
    )
    count = int(math.floor(weight.shape[1] * q4_fraction + 0.5))
    q4_mask = torch.zeros(
        weight.shape[1], device=weight.device, dtype=torch.bool
    )
    if count:
        if selection == "outlier":
            score = column_outlier_density(
                weight, threshold_multiplier=outlier_threshold_multiplier
            )
        else:
            score = q2_error - q4_error
        # Stable index tie-break keeps all artifacts reproducible.
        order = torch.argsort(score, descending=True, stable=True)
        q4_mask[order[:count]] = True
    codes = torch.where(q4_mask.unsqueeze(0), q4_codes, q2_codes)
    error = torch.where(q4_mask, q4_error, q2_error)
    payload = mixed_column_codebook_payload_bits(
        q4_mask,
        out_features=weight.shape[0],
        centroid_bits=centroid_bits,
    )
    return MixedColumnCodebookProjection(
        codes=codes,
        q2_centroids=q2_centroids,
        q4_centroids=q4_centroids,
        q4_mask=q4_mask,
        column_error=error,
        payload_bits=payload,
        physical_bpw=payload / weight.numel(),
        selection=selection,
    )


class FixedColumnCodebookLinear(nn.Module):
    """Reference evaluation layer backed by materialized column codebooks."""

    def __init__(
        self,
        codes: torch.Tensor,
        centroids: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
        compute_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        weight = reconstruct_column_codebook(codes, centroids)
        self.register_buffer("_evaluation_weight", weight.to(compute_dtype))
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().to(compute_dtype).clone())
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.compute_dtype = compute_dtype

    def effective_weight(self) -> torch.Tensor:
        return self._evaluation_weight

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(value.dtype)
        return F.linear(value, self._evaluation_weight.to(value.dtype), bias)


class FixedMixedColumnCodebookLinear(nn.Module):
    """Reference evaluation layer for outlier-selected Q2/Q4 columns."""

    def __init__(
        self,
        projection: MixedColumnCodebookProjection,
        *,
        bias: Optional[torch.Tensor] = None,
        compute_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        weight = projection.effective_weight()
        self.register_buffer("_evaluation_weight", weight.to(compute_dtype))
        self.register_buffer("q4_mask", projection.q4_mask.detach().clone())
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().to(compute_dtype).clone())
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.compute_dtype = compute_dtype
        self.payload_bits = int(projection.payload_bits)
        self.physical_bpw = float(projection.physical_bpw)

    def effective_weight(self) -> torch.Tensor:
        return self._evaluation_weight

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(value.dtype)
        return F.linear(value, self._evaluation_weight.to(value.dtype), bias)
