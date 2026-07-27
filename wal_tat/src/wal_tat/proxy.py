"""Training-only proxy codes for hard-forward ternary optimization."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_ternary_proxy(proxy: torch.Tensor, temperature: float) -> torch.Tensor:
    """Smooth three-level staircase with transitions at -0.5 and +0.5."""
    tau = max(float(temperature), 1e-4)
    return torch.sigmoid((proxy - 0.5) / tau) - torch.sigmoid(
        (-proxy - 0.5) / tau
    )


def soft_ternary_proxy_derivative(
    proxy: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Analytic derivative of :func:`soft_ternary_proxy` with respect to proxy."""
    tau = max(float(temperature), 1e-4)
    positive = torch.sigmoid((proxy - 0.5) / tau)
    negative = torch.sigmoid((-proxy - 0.5) / tau)
    return (
        positive * (1.0 - positive) + negative * (1.0 - negative)
    ) / tau


class ProxyTernaryMatrix(nn.Module):
    """Optimize a scalar proxy per weight while always executing hard codes."""

    def __init__(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        *,
        compute_dtype: torch.dtype,
        temperature: float = 0.35,
        committed_mask: torch.Tensor | None = None,
        master_weight: torch.Tensor | None = None,
        fake_fp16_scale: bool = False,
        initial_proxy_magnitude: float = 1.0,
        initial_zero_proxy_boundary: float | None = None,
    ):
        super().__init__()
        if codes.ndim != 3 or scales.shape != codes.shape[:2]:
            raise ValueError("codes must be [out, groups, group_size] with matching scales")
        if not torch.all((codes >= -1) & (codes <= 1)):
            raise ValueError("codes must be ternary")
        if committed_mask is None:
            committed_mask = torch.ones(codes.shape[:2], dtype=torch.bool, device=codes.device)
        if committed_mask.shape != codes.shape[:2] or committed_mask.dtype != torch.bool:
            raise ValueError("committed_mask must match the code group shape")
        if not 0.5 <= initial_proxy_magnitude <= 1.5:
            raise ValueError("initial_proxy_magnitude must be in [0.5, 1.5]")
        if initial_zero_proxy_boundary is not None:
            if not 0.0 < initial_zero_proxy_boundary < 0.5:
                raise ValueError(
                    "initial_zero_proxy_boundary must be strictly inside (0, 0.5)"
                )
            if master_weight is None:
                raise ValueError(
                    "initial_zero_proxy_boundary requires master_weight for "
                    "deterministic activation directions"
                )
        full_in_features = codes.shape[1] * codes.shape[2]
        if master_weight is None:
            if not committed_mask.all():
                raise ValueError("partial proxy matrices require master_weight")
            base_weight = torch.zeros_like(codes, dtype=torch.float32)
            in_features = full_in_features
        else:
            if master_weight.ndim != 2 or master_weight.shape[0] != codes.shape[0]:
                raise ValueError("master_weight must match the code output dimension")
            if not 0 < master_weight.shape[1] <= full_in_features:
                raise ValueError("master_weight has an invalid input dimension")
            if full_in_features - master_weight.shape[1] >= codes.shape[2]:
                raise ValueError("master_weight padding must be smaller than one group")
            in_features = int(master_weight.shape[1])
            padded = F.pad(
                master_weight.detach().float(), (0, full_in_features - in_features)
            )
            base_weight = padded.view_as(codes)
        initial_proxy = codes.detach().float().clone() * float(
            initial_proxy_magnitude
        )
        if initial_zero_proxy_boundary is not None:
            # Keep the deployed code exactly zero while placing its continuous
            # training proxy close to the nearest hard boundary.  The sign of
            # the original high-precision weight supplies a deterministic
            # direction for possible 0 -> +/-1 transitions.  Exact zeros use
            # +1 so that initialization never introduces an ambiguous sign.
            source_direction = torch.where(
                base_weight >= 0,
                torch.ones_like(base_weight),
                -torch.ones_like(base_weight),
            )
            eligible_zero = (codes == 0) & committed_mask.unsqueeze(-1)
            boundary_proxy = source_direction * float(initial_zero_proxy_boundary)
            initial_proxy = torch.where(
                eligible_zero, boundary_proxy, initial_proxy
            )
        self.proxy_code = nn.Parameter(initial_proxy)
        self.group_scale = nn.Parameter(scales.detach().float().clone())
        self.register_buffer("initial_codes", codes.detach().to(torch.int8).clone())
        self.register_buffer("committed_mask", committed_mask.detach().clone())
        self.register_buffer("base_weight", base_weight.detach().clone())
        self._in_features = in_features
        self.compute_dtype = compute_dtype
        self.temperature = float(temperature)
        self.fake_fp16_scale = bool(fake_fp16_scale)

    @property
    def out_features(self) -> int:
        return self.proxy_code.shape[0]

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def group_size(self) -> int:
        return self.proxy_code.shape[2]

    def hard_codes(self) -> torch.Tensor:
        return self.proxy_code.detach().round().clamp(-1, 1).to(torch.int8)

    def code_churn(self) -> float:
        changed = self.hard_codes() != self.initial_codes
        return float(changed[self.committed_mask].float().mean().item())

    @torch.no_grad()
    def deployment_statistics(self, boundary_epsilon: float = 0.05) -> dict:
        """Summarize hard-code stability and scale health on deployed groups."""
        if boundary_epsilon < 0:
            raise ValueError("boundary_epsilon must be non-negative")
        active_proxy = self.proxy_code[self.committed_mask]
        active_codes = self.hard_codes()[self.committed_mask]
        active_initial = self.initial_codes[self.committed_mask]
        active_scales = self.group_scale.detach().abs()[self.committed_mask]
        if active_proxy.numel() == 0:
            raise ValueError("deployment statistics require committed groups")
        boundary_distance = torch.minimum(
            (active_proxy - 0.5).abs(), (active_proxy + 0.5).abs()
        )
        counts = {
            str(code): int((active_codes == code).sum().item())
            for code in (-1, 0, 1)
        }
        probabilities = torch.tensor(
            list(counts.values()), dtype=torch.float64, device=active_proxy.device
        )
        probabilities /= probabilities.sum().clamp_min(1)
        nonzero = probabilities > 0
        entropy = -(probabilities[nonzero] * probabilities[nonzero].log2()).sum()
        scale_quantiles = torch.quantile(
            active_scales.float(),
            torch.tensor([0.0, 0.5, 0.95, 1.0], device=active_scales.device),
        )
        return {
            "code_counts": counts,
            "zero_fraction": float((active_codes == 0).float().mean().item()),
            "code_entropy_bits": float(entropy.item()),
            "code_churn": float((active_codes != active_initial).float().mean().item()),
            "proxy_abs_displacement_mean": float(
                (active_proxy - active_initial.float()).abs().mean().item()
            ),
            "boundary_epsilon": float(boundary_epsilon),
            "near_boundary_fraction": float(
                (boundary_distance <= boundary_epsilon).float().mean().item()
            ),
            "boundary_distance_min": float(boundary_distance.min().item()),
            "boundary_distance_p01": float(
                torch.quantile(boundary_distance.float(), 0.01).item()
            ),
            "scale_min": float(scale_quantiles[0].item()),
            "scale_median": float(scale_quantiles[1].item()),
            "scale_p95": float(scale_quantiles[2].item()),
            "scale_max": float(scale_quantiles[3].item()),
            "scale_at_clamp_fraction": float(
                (active_scales <= 1.00001e-5).float().mean().item()
            ),
        }

    def proxy_anchor_loss(self) -> torch.Tensor:
        """Squared proxy displacement over deployed ternary groups only."""
        delta = (self.proxy_code - self.initial_codes.float()).square()
        return delta[self.committed_mask].mean()

    def effective_weight(self) -> torch.Tensor:
        soft = soft_ternary_proxy(self.proxy_code, self.temperature)
        hard = self.proxy_code.round().clamp(-1, 1)
        # Exact hard forward with the smooth staircase supplying the gradient.
        code = hard.detach() + soft - soft.detach()
        scale = self.group_scale.abs().clamp_min(1e-5)
        if self.fake_fp16_scale:
            rounded = scale.half().float()
            scale = scale + (rounded - scale).detach()
        value = code * scale.unsqueeze(-1)
        mixed = torch.where(self.committed_mask.unsqueeze(-1), value, self.base_weight)
        return mixed.reshape(self.out_features, -1)[:, : self.in_features].to(
            self.compute_dtype
        )

    @torch.no_grad()
    def constrain_(self) -> None:
        self.proxy_code.clamp_(-1.5, 1.5)
        self.group_scale.clamp_(min=1e-5)


class ProxyTernaryLinear(nn.Module):
    def __init__(self, matrix: ProxyTernaryMatrix, bias=None):
        super().__init__()
        self.matrix = matrix
        self.bias = None if bias is None else nn.Parameter(
            bias.detach().clone(), requires_grad=False
        )
        self.in_features = matrix.in_features
        self.out_features = matrix.out_features

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.matrix.effective_weight().to(value.dtype), self.bias)
