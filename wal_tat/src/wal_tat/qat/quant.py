"""Latent-weight quantizers and the ``QuantLinear`` module used for QAT.

The module implements the BitNet-style recipe that WAL-TAT uses for
*full-coverage* low-bit conversion: every target matrix keeps a
full-precision (fp32) **latent** weight, the forward pass executes the hard
quantized weight, and the backward pass uses a straight-through estimator so
that the latent weight -- and therefore the discrete code assignment -- can
move freely during training.

Two codebooks are supported, both symmetric with one positive scale per
group of ``group_size`` consecutive input features of an output row:

``t3``
    ternary ``{-1, 0, +1}``, stored in two-bit slots (see
    :mod:`wal_tat.packing`).

``b1``
    binary ``{-1, +1}``, stored in one-bit slots (see
    :mod:`wal_tat.binary_packing`).

Group scales are either *derived* from the latent weight on every step
(absmean, the BitNet rule) or *learned* as an explicit parameter.  Learned
scales initialized from the exact activation-weighted projection of
:func:`wal_tat.scoring.exact_diagonal_ternary_project` are the default and
give a markedly better starting point than absmean.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..binary import weighted_binary_project
from ..quantization import group_shape
from ..scoring import exact_diagonal_ternary_project

__all__ = [
    "BinaryQuantizer",
    "GroupQuantizer",
    "QuantLinear",
    "TernaryQuantizer",
    "PRECISIONS",
    "SCALE_EPS",
    "get_quantizer",
    "grouped_view",
]

#: Smallest representable group scale.  Chosen so that the value survives the
#: FP16 round trip used by the deployment format without flushing to zero.
SCALE_EPS = 1e-5

PRECISIONS: Tuple[str, ...] = ("t3", "b1")


def grouped_view(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Return ``weight`` as ``[out_features, groups, group_size]`` in fp32.

    The last group of a row is right-padded with zeros when ``in_features`` is
    not a multiple of ``group_size``.  The operation is differentiable and
    allocation-free when no padding is required.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    size, padding, groups = group_shape(weight.shape[1], group_size)
    value = weight.float()
    if padding:
        value = F.pad(value, (0, padding))
    return value.reshape(weight.shape[0], groups, size)


class GroupQuantizer:
    """Symmetric per-group codebook with a single positive scale."""

    name: str = ""
    levels: Tuple[int, ...] = ()
    code_bits: int = 0

    def derive_scales(
        self, grouped: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Scale implied by the weights themselves, shape ``[out, groups]``.

        Both supported codebooks use the absolute mean: it is the exact
        least-squares optimum for the sign codebook and the BitNet b1.58 rule
        for ternary.  ``lengths`` gives the number of real (non-padding)
        entries per group so that a partially padded final group is not
        biased towards a smaller scale.
        """
        total = grouped.abs().sum(-1)
        if lengths is None:
            return total / grouped.shape[-1]
        return total / lengths.to(total.dtype)

    def encode(self, grouped: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Nearest code for every weight, as float values in :attr:`levels`."""
        raise NotImplementedError

    def project(
        self,
        weight: torch.Tensor,
        input_second_moment: torch.Tensor,
        *,
        group_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Activation-weighted optimal ``(codes, scales, error)`` projection."""
        raise NotImplementedError

    @property
    def entropy_bits(self) -> float:
        """Information content of one code, ignoring the storage layout."""
        return math.log2(len(self.levels))

    def physical_bpw(self, group_size: int, scale_bits: int = 16) -> float:
        """Deployed bits per weight including the amortized group scale."""
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        return self.code_bits + scale_bits / group_size

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(levels={self.levels})"


class TernaryQuantizer(GroupQuantizer):
    """1.58-bit ternary codebook stored in two-bit slots."""

    name = "t3"
    levels = (-1, 0, 1)
    code_bits = 2

    def encode(self, grouped: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return (grouped / scales.unsqueeze(-1)).round().clamp(-1.0, 1.0)

    def project(
        self,
        weight: torch.Tensor,
        input_second_moment: torch.Tensor,
        *,
        group_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return exact_diagonal_ternary_project(
            weight, input_second_moment, group_size=group_size
        )


class BinaryQuantizer(GroupQuantizer):
    """One-bit sign codebook stored in one-bit slots."""

    name = "b1"
    levels = (-1, 1)
    code_bits = 1

    def encode(self, grouped: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        del scales  # the sign codebook is scale invariant
        return torch.where(
            grouped < 0,
            -torch.ones_like(grouped),
            torch.ones_like(grouped),
        )

    def project(
        self,
        weight: torch.Tensor,
        input_second_moment: torch.Tensor,
        *,
        group_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return weighted_binary_project(
            weight, input_second_moment, group_size=group_size
        )


class IdentityQuantizer(GroupQuantizer):
    """No codebook at all -- the full-precision control arm.

    Exists so the FP baseline can travel the *same* code path as a quantized
    run: identical parameter groups, identical loss, identical schedule,
    identical data.  A separate control script would differ from the experiment
    in more than one way, and the entire point of a control is that it differs
    in exactly one.

    Scales are pinned to one and codes are the weights themselves, so the
    forward product ``codes * scale`` reproduces the latent weight exactly and
    the straight-through estimator degenerates to the identity map.

    Deliberately absent from :data:`PRECISIONS`: this is a measurement device,
    not a deployable format, and :meth:`QuantLinear.export_codes` on it would
    be meaningless.
    """

    name = "fp"
    levels = (0,)          # log2(1) = 0 bits: there is no codebook to encode
    code_bits = 32

    def derive_scales(
        self, grouped: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        del lengths
        return torch.ones(
            grouped.shape[:-1], dtype=grouped.dtype, device=grouped.device
        )

    def encode(self, grouped: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return grouped / scales.unsqueeze(-1)

    def project(
        self,
        weight: torch.Tensor,
        input_second_moment: torch.Tensor,
        *,
        group_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del input_second_moment
        grouped = grouped_view(weight, group_size)
        scales = torch.ones(
            grouped.shape[:-1], dtype=grouped.dtype, device=grouped.device
        )
        return grouped, scales, torch.zeros((), device=weight.device)


_QUANTIZERS: Dict[str, GroupQuantizer] = {
    TernaryQuantizer.name: TernaryQuantizer(),
    BinaryQuantizer.name: BinaryQuantizer(),
    IdentityQuantizer.name: IdentityQuantizer(),
}


def get_quantizer(precision: str) -> GroupQuantizer:
    """Return the shared quantizer instance for ``'t3'`` or ``'b1'``."""
    try:
        return _QUANTIZERS[precision]
    except KeyError:
        raise ValueError(
            f"unknown precision {precision!r}, expected one of {PRECISIONS}"
        ) from None


class QuantLinear(nn.Module):
    """Linear layer with a latent fp32 weight and a hard quantized forward.

    The forward pass computes ``codes * scale`` exactly, while the backward
    pass sees the identity map from the latent weight (straight-through
    estimator).  When ``learnable_scales`` is set, the group scales are a real
    parameter and receive their exact gradient ``d(w_q)/d(scale) = codes``.

    Note that the quantized weight is cast to the activation dtype before the
    matmul, so under bf16 autocast the executed product is bf16 -- the same
    arithmetic a deployed bf16 kernel performs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        precision: str = "t3",
        group_size: int = 128,
        learnable_scales: bool = True,
        scale_eps: float = SCALE_EPS,
        fake_fp16_scale: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if scale_eps <= 0:
            raise ValueError("scale_eps must be positive")
        self.quantizer = get_quantizer(precision)
        self.precision = self.quantizer.name
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        size, padding, groups = group_shape(in_features, group_size)
        self.group_size = int(size)
        self.padding = int(padding)
        self.groups = int(groups)
        self.scale_eps = float(scale_eps)
        self.fake_fp16_scale = bool(fake_fp16_scale)
        # Number of real (non-padding) entries per group; only the last group
        # of a row can be short.  Derived from the shape, hence not persisted.
        lengths = torch.full((1, groups), float(size), device=device)
        if padding:
            lengths[0, -1] = float(size - padding)
        self.register_buffer("group_lengths", lengths, persistent=False)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float32, device=device)
        )
        if learnable_scales:
            self.group_scale = nn.Parameter(
                torch.ones(out_features, groups, dtype=torch.float32, device=device)
            )
        else:
            self.register_parameter("group_scale", None)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.float32, device=device)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    # ------------------------------------------------------------------ setup

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias.zero_()
        if self.group_scale is not None:
            self.group_scale.copy_(
                self.quantizer.derive_scales(
                    grouped_view(self.weight, self.group_size), self.group_lengths
                ).clamp_min(self.scale_eps)
            )

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        precision: str = "t3",
        group_size: int = 128,
        learnable_scales: bool = True,
        scale_eps: float = SCALE_EPS,
        fake_fp16_scale: bool = True,
        input_second_moment: Optional[torch.Tensor] = None,
    ) -> "QuantLinear":
        """Build a :class:`QuantLinear` initialized from an existing layer.

        The latent weight is a fp32 copy of ``linear.weight``.  Learned scales
        start from the exact activation-weighted projection when
        ``input_second_moment`` is supplied, and from the unweighted optimum
        otherwise; derived scales have nothing to initialize and ignore the
        moment.
        """
        if not isinstance(linear, nn.Linear):
            raise TypeError("from_linear expects an nn.Linear module")
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            precision=precision,
            group_size=group_size,
            learnable_scales=learnable_scales,
            scale_eps=scale_eps,
            fake_fp16_scale=fake_fp16_scale,
            device=linear.weight.device,
        )
        module.weight.copy_(linear.weight.detach().float())
        if linear.bias is not None:
            module.bias.copy_(linear.bias.detach().float())
        if learnable_scales:
            module.reset_scales_from_projection(input_second_moment)
        return module

    @torch.no_grad()
    def reset_scales_from_projection(
        self, input_second_moment: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Set the group scales to the optimal projection of the latent weight.

        With ``input_second_moment`` this solves the diagonal
        activation-weighted problem; without it the plain least-squares
        optimum.  The projection is *consistent* with :meth:`export_codes`:
        for a symmetric codebook the optimal support of the returned scale is
        exactly the nearest-code assignment, so re-encoding reproduces the
        projected codes.
        """
        moment = self._resolve_moment(input_second_moment)
        _, scales, error = self.quantizer.project(
            self.weight.detach(), moment, group_size=self.group_size
        )
        scales = scales.float().clamp_min(self.scale_eps)
        if self.group_scale is not None:
            self.group_scale.copy_(scales)
        grouped = grouped_view(self.weight.detach(), self.group_size)
        energy = (grouped.square() * moment_grouped(moment, grouped)).sum()
        return {
            "weighted_relative_error": float(
                (error.sum() / energy.clamp_min(1e-12)).sqrt().item()
            ),
            "scale_min": float(scales.min().item()),
            "scale_median": float(scales.median().item()),
            "scale_max": float(scales.max().item()),
        }

    def _resolve_moment(
        self, input_second_moment: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if input_second_moment is None:
            return torch.ones(
                self.in_features, dtype=torch.float32, device=self.weight.device
            )
        moment = input_second_moment.detach().to(
            device=self.weight.device, dtype=torch.float32
        )
        if moment.ndim != 1 or moment.numel() != self.in_features:
            raise ValueError(
                f"input_second_moment must have {self.in_features} entries"
            )
        return moment.clamp_min(0)

    # ------------------------------------------------------------- quantization

    def effective_scales(self) -> torch.Tensor:
        """Positive group scales used by the forward pass, ``[out, groups]``."""
        return self._scales(grouped_view(self.weight.detach(), self.group_size))

    def _scales(self, grouped_detached: torch.Tensor) -> torch.Tensor:
        if self.group_scale is None:
            scale = self.quantizer.derive_scales(grouped_detached, self.group_lengths)
            scale = scale.clamp_min(self.scale_eps)
            return scale.half().float() if self.fake_fp16_scale else scale
        scale = self.group_scale.abs().clamp_min(self.scale_eps)
        if self.fake_fp16_scale:
            # Straight-through rounding to the deployed FP16 scale grid.
            scale = scale + (scale.half().float() - scale).detach()
        return scale

    def quantized_weight(self) -> torch.Tensor:
        """Hard quantized weight with a straight-through gradient path."""
        grouped = grouped_view(self.weight, self.group_size)
        detached = grouped.detach()
        scale = self._scales(detached)
        codes = self.quantizer.encode(detached, scale.detach())
        value = codes * scale.unsqueeze(-1) + (grouped - detached)
        return value.reshape(self.out_features, -1)[:, : self.in_features]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.quantized_weight()
        if weight.dtype != value.dtype:
            weight = weight.to(value.dtype)
        bias = self.bias
        if bias is not None and bias.dtype != value.dtype:
            bias = bias.to(value.dtype)
        return F.linear(value, weight, bias)

    # ----------------------------------------------------------------- export

    @torch.no_grad()
    def export_codes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return deployable ``(codes int8 [out, groups, gs], scales fp16)``.

        The codes are re-encoded against the FP16-rounded scales, so the pair
        is always self-consistent even when ``fake_fp16_scale`` is disabled.
        Padding lanes hold code ``0`` for ternary and ``+1`` for binary; both
        are ignored by the unpackers.
        """
        grouped = grouped_view(self.weight.detach(), self.group_size)
        scales = self._scales(grouped).half()
        codes = self.quantizer.encode(grouped, scales.float())
        return codes.to(torch.int8), scales

    @torch.no_grad()
    def dequantized_weight(self) -> torch.Tensor:
        """Exact deployed matrix ``[out, in]`` implied by :meth:`export_codes`."""
        codes, scales = self.export_codes()
        value = codes.float() * scales.float().unsqueeze(-1)
        return value.reshape(self.out_features, -1)[:, : self.in_features]

    @torch.no_grad()
    def code_flips(self, reference: torch.Tensor) -> int:
        """Number of codes that differ from ``reference`` (an earlier export)."""
        codes, _ = self.export_codes()
        if codes.shape != reference.shape:
            raise ValueError("reference codes have a different shape")
        return int((codes != reference.to(codes.device)).sum().item())

    @torch.no_grad()
    def quant_error(
        self, input_second_moment: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Round-trip diagnostics of the current latent weight."""
        grouped = grouped_view(self.weight.detach(), self.group_size)
        codes, scales = self.export_codes()
        approx = codes.float() * scales.float().unsqueeze(-1)
        delta = grouped - approx
        moment = moment_grouped(self._resolve_moment(input_second_moment), grouped)
        total = grouped.square().sum().clamp_min(1e-12)
        weighted_total = (moment * grouped.square()).sum().clamp_min(1e-12)
        scale_values = scales.float()
        stats = {
            "relative_frobenius": float(
                (delta.square().sum() / total).sqrt().item()
            ),
            "weighted_relative_error": float(
                ((moment * delta.square()).sum() / weighted_total).sqrt().item()
            ),
            "scale_min": float(scale_values.min().item()),
            "scale_median": float(scale_values.median().item()),
            "scale_max": float(scale_values.max().item()),
        }
        for level in self.quantizer.levels:
            stats[f"code_fraction_{level}"] = float(
                (codes == level).float().mean().item()
            )
        return stats

    # ------------------------------------------------------------------- misc

    @torch.no_grad()
    def constrain_(self) -> None:
        """Keep learned scales strictly positive after an optimizer step."""
        if self.group_scale is not None:
            self.group_scale.abs_().clamp_(min=self.scale_eps)

    @property
    def learnable_scales(self) -> bool:
        return self.group_scale is not None

    @property
    def num_weights(self) -> int:
        return self.out_features * self.in_features

    @property
    def num_groups(self) -> int:
        return self.out_features * self.groups

    def physical_bpw(self, scale_bits: int = 16) -> float:
        """Deployed bits per weight for this matrix, padding included."""
        bits = (
            self.num_groups * self.group_size * self.quantizer.code_bits
            + self.num_groups * scale_bits
        )
        return bits / self.num_weights

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, precision={self.precision!r}, "
            f"group_size={self.group_size}, groups={self.groups}, "
            f"learnable_scales={self.learnable_scales}"
        )


def moment_grouped(moment: torch.Tensor, grouped: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-input-feature moment onto a grouped weight view."""
    groups, size = grouped.shape[1], grouped.shape[2]
    padding = groups * size - moment.numel()
    if padding < 0:
        raise ValueError("moment is longer than the grouped weight")
    value = F.pad(moment, (0, padding)) if padding else moment
    return value.view(1, groups, size)
