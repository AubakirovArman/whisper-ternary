"""Function-preserving reparameterizations for Whisper attention pairs."""
from __future__ import annotations

import torch


@torch.no_grad()
def value_output_channel_equalization(
    value_weight: torch.Tensor,
    value_bias: torch.Tensor | None,
    output_weight: torch.Tensor,
    *,
    alpha: float,
    minimum_scale: float = 0.25,
    maximum_scale: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Equalize V rows and inverse O columns without changing FP behavior.

    The scale basis is the RMS of each output-projection input column.  Alpha
    zero is the identity.  Positive alpha progressively normalizes O columns;
    the inverse transformation is absorbed into V rows and its bias.
    """
    if value_weight.ndim != 2 or output_weight.ndim != 2:
        raise ValueError("value/output weights must be matrices")
    channels = value_weight.shape[0]
    if output_weight.shape[1] != channels:
        raise ValueError("value output channels must match output input channels")
    if value_bias is not None and value_bias.shape != (channels,):
        raise ValueError("value bias shape mismatch")
    if minimum_scale <= 0 or maximum_scale < minimum_scale:
        raise ValueError("invalid equalization clamp")
    if not torch.isfinite(value_weight).all() or not torch.isfinite(output_weight).all():
        raise ValueError("weights must be finite")

    column_rms = output_weight.float().square().mean(dim=0).sqrt().clamp_min(1e-8)
    geometric_mean = column_rms.log().mean().exp()
    scale = (column_rms / geometric_mean).pow(float(alpha)).clamp(
        minimum_scale, maximum_scale
    )
    transformed_value = value_weight.float() * scale.unsqueeze(1)
    transformed_bias = (
        None if value_bias is None else value_bias.float() * scale
    )
    transformed_output = output_weight.float() / scale.unsqueeze(0)
    return transformed_value, transformed_bias, transformed_output, scale
