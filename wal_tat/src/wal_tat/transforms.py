"""Fixed orthogonal transforms for packed low-bit linear layers.

The randomized Hadamard transform is block diagonal over the input feature
dimension. Both stored weight rows and row-major activations use the same
forward transform ``v -> (v * D) H``. Consequently, for orthogonal ``R = H D``

``linear(x, W) == linear(transform(x), transform(W))``.

Only the transformed weight is ternary and stored. The activation transform
must remain part of the deployed operator; materializing the inverse transform
into the weight would destroy the ternary representation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantization import hard_codes_scales


def _power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def normalized_hadamard(value: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal Walsh-Hadamard transform on the last dimension."""
    features = int(value.shape[-1])
    if not _power_of_two(features):
        raise ValueError("Hadamard dimension must be a positive power of two")
    result = value
    width = 1
    prefix = value.shape[:-1]
    while width < features:
        paired = result.reshape(*prefix, -1, 2, width)
        left = paired[..., 0, :]
        right = paired[..., 1, :]
        result = torch.cat((left + right, left - right), dim=-1).reshape(
            *prefix, features
        )
        width *= 2
    return result / math.sqrt(features)


def rademacher_signs(
    groups: int,
    group_size: int,
    *,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate device-independent deterministic Rademacher signs."""
    if groups < 1:
        raise ValueError("groups must be positive")
    if not _power_of_two(group_size):
        raise ValueError("group_size must be a positive power of two")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    bits = torch.randint(
        0,
        2,
        (groups, group_size),
        generator=generator,
        dtype=torch.int8,
        device="cpu",
    )
    return bits.to(device=device, dtype=dtype).mul(2).sub(1)


def blockwise_randomized_hadamard(
    value: torch.Tensor,
    *,
    group_size: int = 128,
    seed: int = 109,
) -> torch.Tensor:
    """Map row vectors to the randomized-Hadamard basis.

    The last dimension must be exactly divisible by ``group_size``. Padding is
    deliberately rejected because padding before an orthogonal transform would
    change the represented linear operator.
    """
    features = int(value.shape[-1])
    if features % group_size:
        raise ValueError("input features must be divisible by group_size")
    groups = features // group_size
    grouped = value.reshape(*value.shape[:-1], groups, group_size)
    signs = rademacher_signs(
        groups,
        group_size,
        seed=seed,
        device=value.device,
        dtype=value.dtype,
    )
    return normalized_hadamard(grouped * signs).reshape_as(value)


def inverse_blockwise_randomized_hadamard(
    value: torch.Tensor,
    *,
    group_size: int = 128,
    seed: int = 109,
) -> torch.Tensor:
    """Map transformed weight rows back to the original dense basis."""
    features = int(value.shape[-1])
    if features % group_size:
        raise ValueError("input features must be divisible by group_size")
    groups = features // group_size
    grouped = value.reshape(*value.shape[:-1], groups, group_size)
    signs = rademacher_signs(
        groups,
        group_size,
        seed=seed,
        device=value.device,
        dtype=value.dtype,
    )
    return (normalized_hadamard(grouped) * signs).reshape_as(value)


class FixedTernaryLinear(nn.Module):
    """Inference-only ternary linear with identity or randomized-Hadamard input."""

    def __init__(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        *,
        compute_dtype: torch.dtype,
        transform: str = "identity",
        transform_seed: int = 109,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        if codes.ndim != 3 or scales.shape != codes.shape[:2]:
            raise ValueError("codes must be [out, groups, group_size]")
        if not torch.all((codes >= -1) & (codes <= 1)):
            raise ValueError("codes must be ternary")
        if transform not in {"identity", "rht"}:
            raise ValueError("transform must be 'identity' or 'rht'")
        self.register_buffer("ternary_codes", codes.detach().to(torch.int8))
        self.register_buffer("group_scales", scales.detach().float())
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None
        self.compute_dtype = compute_dtype
        self.transform = transform
        self.transform_seed = int(transform_seed)
        self.in_features = int(codes.shape[1] * codes.shape[2])
        self.out_features = int(codes.shape[0])
        self.group_size = int(codes.shape[2])
        self.register_buffer(
            "_evaluation_weight",
            (
                self.ternary_codes.float()
                * self.group_scales.unsqueeze(-1)
            )
            .reshape(self.out_features, self.in_features)
            .to(compute_dtype),
            persistent=False,
        )

    @classmethod
    @torch.no_grad()
    def from_weight(
        cls,
        weight: torch.Tensor,
        *,
        group_size: int = 128,
        transform: str = "identity",
        transform_seed: int = 109,
        bias: torch.Tensor | None = None,
    ) -> "FixedTernaryLinear":
        if weight.ndim != 2:
            raise ValueError("weight must be a matrix")
        if weight.shape[1] % group_size:
            raise ValueError("weight input features must be divisible by group_size")
        transformed = weight.detach().float()
        if transform == "rht":
            transformed = blockwise_randomized_hadamard(
                transformed, group_size=group_size, seed=transform_seed
            )
        elif transform != "identity":
            raise ValueError("transform must be 'identity' or 'rht'")
        flat_codes, scales = hard_codes_scales(transformed, group_size)
        codes = flat_codes.reshape(weight.shape[0], -1, group_size)
        return cls(
            codes,
            scales,
            compute_dtype=weight.dtype,
            transform=transform,
            transform_seed=transform_seed,
            bias=bias,
        )

    def transformed_weight(self) -> torch.Tensor:
        return self._evaluation_weight

    def effective_weight(self) -> torch.Tensor:
        value = self.transformed_weight()
        if self.transform == "rht":
            value = inverse_blockwise_randomized_hadamard(
                value,
                group_size=self.group_size,
                seed=self.transform_seed,
            )
        return value.to(self.compute_dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.transform == "rht":
            value = blockwise_randomized_hadamard(
                value,
                group_size=self.group_size,
                seed=self.transform_seed,
            )
        weight = self.transformed_weight().to(value.dtype)
        bias = None if self.bias is None else self.bias.to(value.dtype)
        return F.linear(value, weight, bias)

    def code_histogram(self) -> dict[int, int]:
        values, counts = torch.unique(self.ternary_codes.cpu(), return_counts=True)
        result = {-1: 0, 0: 0, 1: 0}
        result.update(
            {int(value): int(count) for value, count in zip(values, counts)}
        )
        return result


class TransformedProxyTernaryLinear(nn.Module):
    """Training wrapper for hard-forward proxy codes in a transformed basis.

    ``matrix.effective_weight()`` is expected to return the deployed weight in
    transform space.  The input transform stays explicit so no dense inverse
    weight is materialized during forward.
    """

    def __init__(
        self,
        matrix: nn.Module,
        *,
        transform: str = "rht",
        transform_seed: int = 109,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        if transform not in {"identity", "rht"}:
            raise ValueError("transform must be 'identity' or 'rht'")
        if not hasattr(matrix, "effective_weight"):
            raise TypeError("matrix must provide effective_weight()")
        self.matrix = matrix
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None
        self.transform = transform
        self.transform_seed = int(transform_seed)
        self.in_features = int(matrix.in_features)
        self.out_features = int(matrix.out_features)
        self.group_size = int(matrix.group_size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.transform == "rht":
            value = blockwise_randomized_hadamard(
                value,
                group_size=self.group_size,
                seed=self.transform_seed,
            )
        weight = self.matrix.effective_weight().to(value.dtype)
        bias = None if self.bias is None else self.bias.to(value.dtype)
        return F.linear(value, weight, bias)
