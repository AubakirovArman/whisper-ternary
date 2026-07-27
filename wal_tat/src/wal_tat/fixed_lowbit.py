"""Inference-only groupwise projections used for controlled PTQ baselines."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .binary import (
    ProxyBinaryLinear,
    ProxyBinaryMatrix,
    unpack_binary_codes,
    weighted_binary_project,
)
from .binary_packing import read_packed_binary_matrix
from .quantization import (
    weighted_symmetric_nz4_project,
    weighted_symmetric_q4_project,
)
from .scoring import exact_diagonal_ternary_project
from .packing import read_packed_matrix, unpack_ternary_codes
from .proxy import ProxyTernaryLinear, ProxyTernaryMatrix
from .whisper import get_module, set_module


class FixedGroupwiseLinear(nn.Module):
    """Materialized reference forward for arbitrary signed groupwise codes."""

    def __init__(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        *,
        in_features: int,
        compute_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
        allowed_codes: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if codes.ndim != 3 or scales.shape != codes.shape[:2]:
            raise ValueError("codes must be [out, groups, group_size] with matching scales")
        full_in_features = codes.shape[1] * codes.shape[2]
        if not 0 < in_features <= full_in_features:
            raise ValueError("in_features is incompatible with grouped codes")
        if full_in_features - in_features >= codes.shape[2]:
            raise ValueError("padding must be smaller than one group")
        if allowed_codes is not None:
            allowed = torch.tensor(tuple(allowed_codes), device=codes.device)
            if not torch.isin(codes, allowed).all():
                raise ValueError("codes contain a value outside the declared codebook")
        self.register_buffer("codes", codes.detach().to(torch.int8).clone())
        self.register_buffer("group_scales", scales.detach().float().clone())
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().clone())
        self.in_features = int(in_features)
        self.out_features = int(codes.shape[0])
        self.group_size = int(codes.shape[2])
        self.compute_dtype = compute_dtype
        self.register_buffer(
            "_evaluation_weight",
            (self.codes.float() * self.group_scales.unsqueeze(-1))
            .reshape(self.out_features, -1)[:, : self.in_features]
            .to(compute_dtype),
            persistent=False,
        )

    def effective_weight(self) -> torch.Tensor:
        return self._evaluation_weight.to(self.compute_dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(value.dtype)
        return F.linear(value, self._evaluation_weight.to(value.dtype), bias)


class FixedPartialGroupwiseLinear(nn.Module):
    """Reference forward with strict low-bit groups and exact BF16 fallback."""

    def __init__(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        committed_mask: torch.Tensor,
        base_weight: torch.Tensor,
        *,
        in_features: int,
        compute_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
        allowed_codes: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if codes.ndim != 3 or scales.shape != codes.shape[:2]:
            raise ValueError(
                "codes must be [out, groups, group_size] with matching scales"
            )
        if committed_mask.shape != codes.shape[:2] or committed_mask.dtype != torch.bool:
            raise ValueError("committed_mask must match the grouped code shape")
        if not committed_mask.any():
            raise ValueError("partial low-bit matrix must commit at least one group")
        full_in_features = codes.shape[1] * codes.shape[2]
        expected_base_shape = (codes.shape[0], in_features)
        if tuple(base_weight.shape) != expected_base_shape:
            raise ValueError(
                f"base_weight shape is {tuple(base_weight.shape)}, "
                f"expected {expected_base_shape}"
            )
        if not 0 < in_features <= full_in_features:
            raise ValueError("in_features is incompatible with grouped codes")
        if full_in_features - in_features >= codes.shape[2]:
            raise ValueError("padding must be smaller than one group")
        if allowed_codes is not None:
            allowed = torch.tensor(tuple(allowed_codes), device=codes.device)
            active = codes[committed_mask]
            if not torch.isin(active, allowed).all():
                raise ValueError(
                    "committed codes contain a value outside the declared codebook"
                )
        self.register_buffer("codes", codes.detach().to(torch.int8).clone())
        self.register_buffer("group_scales", scales.detach().float().clone())
        self.register_buffer("committed_mask", committed_mask.detach().clone())
        self.register_buffer(
            "base_weight", base_weight.detach().to(compute_dtype).clone()
        )
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().clone())
        self.in_features = int(in_features)
        self.out_features = int(codes.shape[0])
        self.group_size = int(codes.shape[2])
        self.compute_dtype = compute_dtype
        padded_base = F.pad(
            self.base_weight.float(), (0, full_in_features - self.in_features)
        ).view_as(self.codes)
        low_bit = self.codes.float() * self.group_scales.unsqueeze(-1)
        mixed = torch.where(
            self.committed_mask.unsqueeze(-1), low_bit, padded_base
        )
        self.register_buffer(
            "_evaluation_weight",
            mixed.reshape(self.out_features, -1)[:, : self.in_features].to(
                compute_dtype
            ),
            persistent=False,
        )

    def effective_weight(self) -> torch.Tensor:
        return self._evaluation_weight.to(self.compute_dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(value.dtype)
        return F.linear(value, self._evaluation_weight.to(value.dtype), bias)


@dataclass(frozen=True)
class ProjectionStatistics:
    name: str
    precision: str
    group_size: int
    weights: int
    weighted_error: float
    code_counts: Mapping[str, int]


@torch.no_grad()
def project_linear_module(
    linear: nn.Linear,
    *,
    precision: str,
    group_size: int,
    input_second_moment: Optional[torch.Tensor] = None,
) -> tuple[FixedGroupwiseLinear, float, Mapping[str, int]]:
    """Project one dense linear into a fixed B1/T3/NZ4/Q4 reference layer."""
    moment = (
        torch.ones(linear.in_features, device=linear.weight.device)
        if input_second_moment is None
        else input_second_moment.to(linear.weight.device)
    )
    if precision == "b1":
        codes, scales, error = weighted_binary_project(
            linear.weight, moment, group_size=group_size
        )
        allowed = (-1, 1)
    elif precision == "t3":
        codes, scales, error = exact_diagonal_ternary_project(
            linear.weight, moment, group_size=group_size
        )
        allowed = (-1, 0, 1)
    elif precision == "nz4":
        codes, scales, error = weighted_symmetric_nz4_project(
            linear.weight, moment, group_size=group_size
        )
        allowed = (-3, -1, 1, 3)
    elif precision == "q4":
        codes, scales, error = weighted_symmetric_q4_project(
            linear.weight, moment, group_size=group_size
        )
        allowed = tuple(range(-8, 8))
    else:
        raise ValueError("precision must be one of b1, t3, nz4, q4")
    values, counts = torch.unique(codes.cpu(), return_counts=True)
    histogram = {str(int(value)): int(count) for value, count in zip(values, counts)}
    fixed = FixedGroupwiseLinear(
        codes,
        scales,
        in_features=linear.in_features,
        compute_dtype=linear.weight.dtype,
        bias=linear.bias,
        allowed_codes=allowed,
    )
    return fixed, float(error.sum().item()), histogram


@torch.no_grad()
def install_fixed_projection(
    model: nn.Module,
    module_names: Sequence[str],
    *,
    precision: str,
    group_size: int,
    input_moments: Optional[Mapping[str, torch.Tensor]] = None,
) -> tuple[ProjectionStatistics, ...]:
    """Replace declared linears with deterministic projected references."""
    results = []
    for name in module_names:
        linear = get_module(model, name)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{name} is not an nn.Linear")
        fixed, error, histogram = project_linear_module(
            linear,
            precision=precision,
            group_size=group_size,
            input_second_moment=(
                None if input_moments is None else input_moments.get(name)
            ),
        )
        fixed.to(device=linear.weight.device)
        set_module(model, name, fixed)
        results.append(
            ProjectionStatistics(
                name=name,
                precision=precision,
                group_size=group_size,
                weights=linear.weight.numel(),
                weighted_error=error,
                code_counts=histogram,
            )
        )
    return tuple(results)


@torch.no_grad()
def install_fixed_checkpoint(
    model: nn.Module,
    checkpoint: Mapping,
) -> tuple[str, ...]:
    """Install a pilot/checkpoint code+scale mapping into a fresh dense model."""
    allowed_by_precision = {
        "b1": (-1, 1),
        "t3": (-1, 0, 1),
        "nz4": (-3, -1, 1, 3),
        "q4": tuple(range(-8, 8)),
    }
    checkpoint_precision = str(checkpoint["precision"])
    if checkpoint_precision != "mixed" and checkpoint_precision not in allowed_by_precision:
        raise ValueError("checkpoint precision is unsupported")
    names = []
    for name, entry in checkpoint["matrices"].items():
        precision = str(entry.get("precision", checkpoint_precision))
        if precision not in allowed_by_precision:
            raise ValueError(f"{name} checkpoint precision is unsupported")
        linear = get_module(model, name)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{name} is not an nn.Linear in the fresh model")
        codes = torch.as_tensor(entry["codes"], device=linear.weight.device)
        scales = torch.as_tensor(entry["scales"], device=linear.weight.device)
        bias = entry.get("bias")
        if bias is None:
            bias = linear.bias
        committed_mask = entry.get("committed_mask")
        if committed_mask is None:
            fixed = FixedGroupwiseLinear(
                codes,
                scales,
                in_features=linear.in_features,
                compute_dtype=linear.weight.dtype,
                bias=bias,
                allowed_codes=allowed_by_precision[precision],
            )
        else:
            if precision != "t3":
                raise ValueError(
                    f"{name} partial checkpoint currently supports T3 only"
                )
            mask = torch.as_tensor(
                committed_mask, device=linear.weight.device, dtype=torch.bool
            )
            stored_base = entry.get("base_weight")
            base_weight = (
                linear.weight
                if stored_base is None
                else torch.as_tensor(
                    stored_base,
                    device=linear.weight.device,
                    dtype=linear.weight.dtype,
                )
            )
            fixed = FixedPartialGroupwiseLinear(
                codes,
                scales,
                mask,
                base_weight,
                in_features=linear.in_features,
                compute_dtype=linear.weight.dtype,
                bias=bias,
                allowed_codes=allowed_by_precision[precision],
            )
        fixed = fixed.to(device=linear.weight.device)
        set_module(model, name, fixed)
        names.append(name)
    return tuple(names)


@torch.no_grad()
def install_trainable_lowbit_checkpoint(
    model: nn.Module,
    checkpoint: Mapping,
    *,
    temperature: float = 0.35,
    initial_proxy_magnitude: float = 0.75,
    initial_proxy_magnitudes: Optional[Mapping[str, float]] = None,
    initial_zero_proxy_boundaries: Optional[Mapping[str, float]] = None,
    fake_fp16_scale: bool = True,
) -> Mapping[str, ProxyBinaryMatrix | ProxyTernaryMatrix]:
    """Install a B1/T3 checkpoint as exact-hard-forward trainable proxies.

    This deliberately rejects Q4 and other rescue formats.  It is the bridge
    used by global recovery: the deployed codebook is already active in the
    forward pass while gradients may move proxy codes across its boundaries.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if initial_proxy_magnitude <= 0:
        raise ValueError("initial_proxy_magnitude must be positive")
    per_matrix_magnitudes = dict(initial_proxy_magnitudes or {})
    per_matrix_zero_boundaries = dict(initial_zero_proxy_boundaries or {})
    unknown_names = set(per_matrix_magnitudes) - set(checkpoint["matrices"])
    if unknown_names:
        raise ValueError(
            f"initial proxy magnitudes name unknown matrices: {sorted(unknown_names)}"
        )
    unknown_zero_names = set(per_matrix_zero_boundaries) - set(
        checkpoint["matrices"]
    )
    if unknown_zero_names:
        raise ValueError(
            "initial zero proxy boundaries name unknown matrices: "
            f"{sorted(unknown_zero_names)}"
        )
    checkpoint_precision = str(checkpoint["precision"])
    matrices: dict[str, ProxyBinaryMatrix | ProxyTernaryMatrix] = {}
    for name, entry in checkpoint["matrices"].items():
        precision = str(entry.get("precision", checkpoint_precision))
        if precision not in {"b1", "t3"}:
            raise ValueError(
                f"{name} uses {precision}; global strict-low-bit recovery "
                "accepts only b1/t3"
            )
        linear = get_module(model, name)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{name} is not an nn.Linear in the fresh model")
        codes = torch.as_tensor(entry["codes"], device=linear.weight.device)
        scales = torch.as_tensor(
            entry["scales"], device=linear.weight.device, dtype=torch.float32
        )
        committed_mask = entry.get("committed_mask")
        if committed_mask is not None:
            if precision != "t3":
                raise ValueError(
                    f"{name} partial trainable checkpoint currently supports T3 only"
                )
            committed_mask = torch.as_tensor(
                committed_mask, device=linear.weight.device, dtype=torch.bool
            )
        bias = entry.get("bias")
        if bias is None:
            bias = linear.bias
        elif not isinstance(bias, torch.Tensor):
            bias = torch.as_tensor(bias)
        if bias is not None:
            bias = bias.to(device=linear.weight.device, dtype=linear.weight.dtype)
        proxy_magnitude = float(
            per_matrix_magnitudes.get(name, initial_proxy_magnitude)
        )
        if proxy_magnitude <= 0:
            raise ValueError(f"{name} initial proxy magnitude must be positive")
        if precision == "t3" and proxy_magnitude < 0.5:
            raise ValueError(
                f"{name} initial proxy magnitude must be at least 0.5 for T3"
            )
        if precision == "b1":
            if name in per_matrix_zero_boundaries:
                raise ValueError(
                    f"{name} cannot use an initial zero proxy boundary in B1"
                )
            matrix = ProxyBinaryMatrix(
                codes,
                scales,
                compute_dtype=linear.weight.dtype,
                temperature=temperature,
                initial_proxy_magnitude=proxy_magnitude,
                fake_fp16_scale=fake_fp16_scale,
            )
            wrapper = ProxyBinaryLinear(matrix, bias)
        else:
            matrix = ProxyTernaryMatrix(
                codes,
                scales,
                compute_dtype=linear.weight.dtype,
                temperature=temperature,
                committed_mask=committed_mask,
                master_weight=(
                    linear.weight
                    if entry.get("base_weight") is None
                    else torch.as_tensor(
                        entry["base_weight"],
                        device=linear.weight.device,
                        dtype=linear.weight.dtype,
                    )
                ),
                initial_proxy_magnitude=proxy_magnitude,
                initial_zero_proxy_boundary=per_matrix_zero_boundaries.get(name),
                fake_fp16_scale=fake_fp16_scale,
            )
            wrapper = ProxyTernaryLinear(matrix, bias)
        wrapper.to(device=linear.weight.device)
        set_module(model, name, wrapper)
        matrices[name] = matrix
    return matrices


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_member(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    if path.parent != directory.resolve():
        raise ValueError("artifact member must be a direct child of the manifest directory")
    return path


@torch.no_grad()
def install_packed_t3_manifest(
    model: nn.Module,
    manifest_path: str | Path,
) -> tuple[str, ...]:
    """Load strict, fully committed T3 matrices from a WAL Q2 manifest.

    This is a correctness/reference loader. ``FixedGroupwiseLinear`` still
    materializes an evaluation tensor, so deployment speed requires a native
    packed kernel; however, every value installed here originates from the
    serialized two-bit codes and FP16 scales rather than the training checkpoint.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("precision") != "t3":
        raise ValueError("manifest is not a supported strict T3 artifact")
    directory = manifest_path.parent
    names: list[str] = []
    for entry in manifest.get("entries", ()):
        name = str(entry["name"])
        linear = get_module(model, name)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{name} is not an nn.Linear in the fresh model")
        matrix_path = _artifact_member(directory, str(entry["matrix_file"]))
        if matrix_path.stat().st_size != int(entry["matrix_bytes"]):
            raise ValueError(f"{name} packed matrix size does not match the manifest")
        if _sha256_file(matrix_path) != str(entry["matrix_sha256"]):
            raise ValueError(f"{name} packed matrix SHA-256 does not match the manifest")
        packed = read_packed_matrix(matrix_path)
        if packed.committed_groups != packed.total_groups:
            raise ValueError("strict T3 manifest cannot contain BF16 fallback groups")
        if packed.shape != (linear.out_features, linear.in_features):
            raise ValueError(f"{name} packed shape does not match the fresh model")
        groups_per_row = packed.total_groups // packed.shape[0]
        codes = unpack_ternary_codes(
            packed.codes_packed,
            packed.committed_groups * packed.group_size,
        ).view(packed.shape[0], groups_per_row, packed.group_size)
        scales = packed.scales_fp16.view(packed.shape[0], groups_per_row)

        bias = None
        if entry.get("bias_file") is not None:
            bias_path = _artifact_member(directory, str(entry["bias_file"]))
            if bias_path.stat().st_size != int(entry["bias_bytes"]):
                raise ValueError(f"{name} bias size does not match the manifest")
            if _sha256_file(bias_path) != str(entry["bias_sha256"]):
                raise ValueError(f"{name} bias SHA-256 does not match the manifest")
            raw = bias_path.read_bytes()
            bias = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone()
            if bias.numel() != linear.out_features:
                raise ValueError(f"{name} bias shape does not match the fresh model")
        elif linear.bias is not None:
            raise ValueError(f"{name} manifest omitted a required bias")

        fixed = FixedGroupwiseLinear(
            codes,
            scales,
            in_features=linear.in_features,
            compute_dtype=linear.weight.dtype,
            bias=bias,
            allowed_codes=(-1, 0, 1),
        ).to(device=linear.weight.device)
        set_module(model, name, fixed)
        names.append(name)
    if not names:
        raise ValueError("manifest contains no matrices")
    return tuple(names)


@torch.no_grad()
def install_packed_lowbit_manifest(
    model: nn.Module,
    manifest_path: str | Path,
) -> tuple[str, ...]:
    """Install a strict schema-v2 B1/T3 overlay from serialized payloads.

    This is a correctness loader. It materializes a reference evaluation tensor;
    native speed and memory savings still require a packed B1/T3 kernel.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    schema_policy = (
        manifest.get("schema_version"),
        manifest.get("precision_policy"),
    )
    if schema_policy not in {
        (2, "strict_b1_t3_only"),
        (3, "strict_b1_t3_partial_bf16_fallback"),
    }:
        raise ValueError("manifest is not a supported strict mixed low-bit artifact")
    directory = manifest_path.parent
    names: list[str] = []
    for entry in manifest.get("entries", ()):
        name = str(entry["name"])
        precision = str(entry["precision"])
        if precision not in {"b1", "t3"}:
            raise ValueError(f"{name} uses unsupported packed precision {precision}")
        linear = get_module(model, name)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{name} is not an nn.Linear in the fresh model")
        matrix_path = _artifact_member(directory, str(entry["matrix_file"]))
        if matrix_path.stat().st_size != int(entry["matrix_bytes"]):
            raise ValueError(f"{name} packed matrix size does not match the manifest")
        if _sha256_file(matrix_path) != str(entry["matrix_sha256"]):
            raise ValueError(f"{name} packed matrix SHA-256 does not match the manifest")

        if precision == "b1":
            packed_b1 = read_packed_binary_matrix(matrix_path)
            if packed_b1.shape != (linear.out_features, linear.in_features):
                raise ValueError(f"{name} packed shape does not match the fresh model")
            groups_per_row = packed_b1.total_groups // packed_b1.shape[0]
            codes = unpack_binary_codes(
                packed_b1.codes_packed,
                packed_b1.total_groups * packed_b1.group_size,
            ).view(packed_b1.shape[0], groups_per_row, packed_b1.group_size)
            scales = packed_b1.scales_fp16.view(packed_b1.shape[0], groups_per_row)
            allowed_codes = (-1, 1)
        else:
            packed_t3 = read_packed_matrix(matrix_path)
            if packed_t3.shape != (linear.out_features, linear.in_features):
                raise ValueError(f"{name} packed shape does not match the fresh model")
            groups_per_row = packed_t3.total_groups // packed_t3.shape[0]
            mask = packed_t3.committed_mask().view(
                packed_t3.shape[0], groups_per_row
            )
            active_codes = unpack_ternary_codes(
                packed_t3.codes_packed,
                packed_t3.committed_groups * packed_t3.group_size,
            ).view(packed_t3.committed_groups, packed_t3.group_size)
            codes = torch.zeros(
                (
                    packed_t3.shape[0],
                    groups_per_row,
                    packed_t3.group_size,
                ),
                dtype=torch.int8,
            )
            codes[mask] = active_codes
            scales = torch.ones(
                (packed_t3.shape[0], groups_per_row), dtype=torch.float16
            )
            scales[mask] = packed_t3.scales_fp16
            allowed_codes = (-1, 0, 1)

        bias = None
        if entry.get("bias_file") is not None:
            bias_path = _artifact_member(directory, str(entry["bias_file"]))
            if bias_path.stat().st_size != int(entry["bias_bytes"]):
                raise ValueError(f"{name} bias size does not match the manifest")
            if _sha256_file(bias_path) != str(entry["bias_sha256"]):
                raise ValueError(f"{name} bias SHA-256 does not match the manifest")
            bias = torch.frombuffer(
                bytearray(bias_path.read_bytes()), dtype=torch.bfloat16
            ).clone()
            if bias.numel() != linear.out_features:
                raise ValueError(f"{name} bias shape does not match the fresh model")
        elif linear.bias is not None:
            raise ValueError(f"{name} manifest omitted a required bias")

        if precision == "t3" and packed_t3.committed_groups != packed_t3.total_groups:
            grouped_base = torch.zeros(
                (
                    packed_t3.shape[0],
                    groups_per_row,
                    packed_t3.group_size,
                ),
                dtype=torch.bfloat16,
            )
            grouped_base[~mask] = packed_t3.fallback_bf16.view(
                packed_t3.total_groups - packed_t3.committed_groups,
                packed_t3.group_size,
            )
            base_weight = grouped_base.reshape(packed_t3.shape[0], -1)[
                :, : linear.in_features
            ]
            fixed = FixedPartialGroupwiseLinear(
                codes,
                scales,
                mask,
                base_weight,
                in_features=linear.in_features,
                compute_dtype=linear.weight.dtype,
                bias=bias,
                allowed_codes=allowed_codes,
            ).to(device=linear.weight.device)
        else:
            fixed = FixedGroupwiseLinear(
                codes,
                scales,
                in_features=linear.in_features,
                compute_dtype=linear.weight.dtype,
                bias=bias,
                allowed_codes=allowed_codes,
            ).to(device=linear.weight.device)
        set_module(model, name, fixed)
        names.append(name)
    if not names:
        raise ValueError("manifest contains no matrices")
    return tuple(names)
