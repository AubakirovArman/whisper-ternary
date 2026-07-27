"""True binary quantization primitives for selective WAL-TAT conversion."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantization import padded_grouped


def q1_g128_physical_bpw(group_size: int = 128, scale_bits: int = 16) -> float:
    """Physical bpw for one sign bit plus one group scale."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return 1.0 + scale_bits / group_size


@torch.no_grad()
def weighted_binary_project(
    weight: torch.Tensor,
    input_second_moment: torch.Tensor,
    *,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagonal activation-weighted projection to ``{-scale, +scale}``."""
    if input_second_moment.ndim != 1 or input_second_moment.numel() != weight.shape[1]:
        raise ValueError("input_second_moment must match weight input features")
    grouped, padding, size = padded_grouped(weight.detach(), group_size)
    moment = input_second_moment.detach().float().clamp_min(0)
    if padding:
        moment = F.pad(moment, (0, padding))
    moment = moment.view(1, -1, size).expand_as(grouped)
    codes = torch.where(grouped < 0, -1, 1).to(torch.int8)
    denominator = moment.sum(-1)
    numerator = (moment * grouped.abs()).sum(-1)
    fallback = grouped.abs().mean(-1)
    scales = torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1e-12),
        fallback,
    ).clamp_min(1e-5)
    error = (
        moment * (grouped - codes.float() * scales.unsqueeze(-1)).square()
    ).sum(-1)
    return codes, scales, error


def soft_binary_proxy(proxy: torch.Tensor, temperature: float) -> torch.Tensor:
    tau = max(float(temperature), 1e-4)
    return torch.tanh(proxy / tau)


class ProxyBinaryMatrix(nn.Module):
    """Hard-forward binary codes with a smooth training-only gradient path."""

    def __init__(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        *,
        compute_dtype: torch.dtype,
        temperature: float = 0.35,
        initial_proxy_magnitude: float = 0.25,
        fake_fp16_scale: bool = False,
    ):
        super().__init__()
        if codes.ndim != 3 or scales.shape != codes.shape[:2]:
            raise ValueError("codes must be [out, groups, group_size] with matching scales")
        if not torch.all((codes == -1) | (codes == 1)):
            raise ValueError("binary codes must be in {-1, +1}")
        if initial_proxy_magnitude <= 0:
            raise ValueError("initial_proxy_magnitude must be positive")
        self.proxy_code = nn.Parameter(
            codes.detach().float().clone() * float(initial_proxy_magnitude)
        )
        self.group_scale = nn.Parameter(scales.detach().float().clone())
        self.register_buffer("initial_codes", codes.detach().to(torch.int8).clone())
        self.compute_dtype = compute_dtype
        self.temperature = float(temperature)
        self.fake_fp16_scale = bool(fake_fp16_scale)

    @property
    def out_features(self) -> int:
        return self.proxy_code.shape[0]

    @property
    def in_features(self) -> int:
        return self.proxy_code.shape[1] * self.proxy_code.shape[2]

    @property
    def group_size(self) -> int:
        return self.proxy_code.shape[2]

    def hard_codes(self) -> torch.Tensor:
        return torch.where(self.proxy_code.detach() < 0, -1, 1).to(torch.int8)

    def effective_weight(self) -> torch.Tensor:
        soft = soft_binary_proxy(self.proxy_code, self.temperature)
        hard = torch.where(self.proxy_code < 0, -1.0, 1.0)
        code = hard.detach() + soft - soft.detach()
        scale = self.group_scale.abs().clamp_min(1e-5)
        if self.fake_fp16_scale:
            rounded = scale.half().float()
            scale = scale + (rounded - scale).detach()
        return (code * scale.unsqueeze(-1)).reshape(
            self.out_features, self.in_features
        ).to(self.compute_dtype)

    def code_churn(self) -> float:
        return float((self.hard_codes() != self.initial_codes).float().mean().item())

    @torch.no_grad()
    def constrain_(self) -> None:
        self.proxy_code.clamp_(-1.5, 1.5)
        self.group_scale.clamp_(min=1e-5)


class ProxyBinaryLinear(nn.Module):
    def __init__(self, matrix: ProxyBinaryMatrix, bias=None):
        super().__init__()
        self.matrix = matrix
        self.bias = None if bias is None else nn.Parameter(
            bias.detach().clone(), requires_grad=False
        )
        self.in_features = matrix.in_features
        self.out_features = matrix.out_features

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.matrix.effective_weight().to(value.dtype), self.bias)


def pack_binary_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack eight binary signs per byte, little-endian within each byte."""
    values = codes.detach().to(torch.int8).contiguous().cpu().reshape(-1)
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)
    if not torch.all((values == -1) | (values == 1)):
        raise ValueError("binary codes must be in {-1, +1}")
    bits = (values > 0).to(torch.uint8)
    padding = (-bits.numel()) % 8
    if padding:
        bits = F.pad(bits, (0, padding), value=0)
    lanes = bits.view(-1, 8)
    packed = torch.zeros(lanes.shape[0], dtype=torch.uint8)
    for bit in range(8):
        packed |= lanes[:, bit] << bit
    return packed


def unpack_binary_codes(packed: torch.Tensor, count: int) -> torch.Tensor:
    if count < 0:
        raise ValueError("count must be non-negative")
    if packed.dtype != torch.uint8:
        raise TypeError("packed must use torch.uint8")
    value = packed.detach().contiguous().cpu().reshape(-1)
    required = (count + 7) // 8
    if value.numel() != required:
        raise ValueError(f"packed code length is {value.numel()}, expected {required}")
    if count == 0:
        return torch.empty(0, dtype=torch.int8)
    lanes = torch.stack(tuple((value >> bit) & 1 for bit in range(8)), dim=1)
    return lanes.reshape(-1)[:count].to(torch.int8).mul(2).sub(1).contiguous()
