"""Reference binary packing for strict ternary Q2 groupwise matrices."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct
from typing import Iterable

import torch
import torch.nn.functional as F


MAGIC = b"WALTQ2G1"
VERSION = 1
# magic, version, rows, cols, group_size, padding, total_groups,
# committed_groups, mask_bytes, code_bytes, scale_bytes, fallback_bytes
HEADER = struct.Struct("<8sIIIHHQQQQQQ")


def _require_uint8(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.dtype != torch.uint8:
        raise TypeError(f"{name} must use torch.uint8")
    return value.detach().contiguous().cpu()


def pack_ternary_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack {-1, 0, +1} into little-endian two-bit slots, four per byte."""
    values = codes.detach().to(torch.int8).contiguous().cpu().reshape(-1)
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)
    if not set(values.unique().tolist()) <= {-1, 0, 1}:
        raise ValueError("ternary codes must be in {-1, 0, +1}")
    encoded = (values + 1).to(torch.uint8)
    padding = (-encoded.numel()) % 4
    if padding:
        encoded = F.pad(encoded, (0, padding), value=0)
    lanes = encoded.view(-1, 4)
    return (
        lanes[:, 0]
        | (lanes[:, 1] << 2)
        | (lanes[:, 2] << 4)
        | (lanes[:, 3] << 6)
    ).contiguous()


def unpack_ternary_codes(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack two-bit slots and reject the reserved binary value 11."""
    if count < 0:
        raise ValueError("count must be non-negative")
    value = _require_uint8(packed, "packed")
    required = (count + 3) // 4
    if value.numel() != required:
        raise ValueError(f"packed code length is {value.numel()}, expected {required}")
    if count == 0:
        return torch.empty(0, dtype=torch.int8)
    lanes = torch.stack(
        tuple((value >> shift) & 0x03 for shift in (0, 2, 4, 6)), dim=1
    ).reshape(-1)[:count]
    if torch.any(lanes == 3):
        raise ValueError("reserved ternary code 11 encountered")
    return (lanes.to(torch.int8) - 1).contiguous()


def pack_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    value = mask.detach().bool().contiguous().cpu().reshape(-1)
    if value.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)
    padding = (-value.numel()) % 8
    if padding:
        value = F.pad(value, (0, padding), value=False)
    lanes = value.view(-1, 8).to(torch.uint8)
    packed = torch.zeros(lanes.shape[0], dtype=torch.uint8)
    for bit in range(8):
        packed |= lanes[:, bit] << bit
    return packed


def unpack_bool_mask(packed: torch.Tensor, count: int) -> torch.Tensor:
    if count < 0:
        raise ValueError("count must be non-negative")
    value = _require_uint8(packed, "packed")
    required = (count + 7) // 8
    if value.numel() != required:
        raise ValueError(f"packed mask length is {value.numel()}, expected {required}")
    if count == 0:
        return torch.empty(0, dtype=torch.bool)
    lanes = torch.stack(
        tuple(((value >> bit) & 1).bool() for bit in range(8)), dim=1
    )
    return lanes.reshape(-1)[:count].contiguous()


@dataclass(frozen=True)
class PackedQ2Matrix:
    shape: tuple[int, int]
    group_size: int
    padding: int
    total_groups: int
    committed_groups: int
    mask_packed: torch.Tensor
    codes_packed: torch.Tensor
    scales_fp16: torch.Tensor
    fallback_bf16: torch.Tensor

    def __post_init__(self) -> None:
        rows, cols = self.shape
        if rows <= 0 or cols <= 0:
            raise ValueError("matrix shape must be positive")
        if self.group_size <= 0 or self.group_size % 4:
            raise ValueError("group_size must be positive and divisible by four")
        if self.padding != (-cols) % self.group_size:
            raise ValueError("padding does not match shape/group_size")
        expected_groups = rows * ((cols + self.padding) // self.group_size)
        if self.total_groups != expected_groups:
            raise ValueError("total group count does not match shape")
        if not 0 <= self.committed_groups <= self.total_groups:
            raise ValueError("invalid committed group count")
        expected_mask = (
            0
            if self.committed_groups in (0, self.total_groups)
            else (self.total_groups + 7) // 8
        )
        if self.mask_packed.dtype != torch.uint8 or self.mask_packed.numel() != expected_mask:
            raise ValueError("invalid packed mask")
        expected_codes = (self.committed_groups * self.group_size + 3) // 4
        if self.codes_packed.dtype != torch.uint8 or self.codes_packed.numel() != expected_codes:
            raise ValueError("invalid packed codes")
        if self.scales_fp16.dtype != torch.float16 or self.scales_fp16.numel() != self.committed_groups:
            raise ValueError("invalid FP16 scales")
        expected_fallback = (self.total_groups - self.committed_groups) * self.group_size
        if self.fallback_bf16.dtype != torch.bfloat16 or self.fallback_bf16.numel() != expected_fallback:
            raise ValueError("invalid BF16 fallback")

    @property
    def payload_nbytes(self) -> int:
        return (
            self.mask_packed.numel()
            + self.codes_packed.numel()
            + self.scales_fp16.numel() * 2
            + self.fallback_bf16.numel() * 2
        )

    @property
    def serialized_nbytes(self) -> int:
        return HEADER.size + self.payload_nbytes

    def true_bpw(self, *, include_header: bool = True) -> float:
        size = self.serialized_nbytes if include_header else self.payload_nbytes
        return size * 8 / (self.shape[0] * self.shape[1])

    def committed_mask(self) -> torch.Tensor:
        if self.committed_groups == self.total_groups:
            return torch.ones(self.total_groups, dtype=torch.bool)
        if self.committed_groups == 0:
            return torch.zeros(self.total_groups, dtype=torch.bool)
        return unpack_bool_mask(self.mask_packed, self.total_groups)


def _group_codes(
    codes: torch.Tensor, shape: tuple[int, int], group_size: int
) -> torch.Tensor:
    rows, cols = shape
    if codes.ndim == 3:
        expected = (rows, (cols + (-cols) % group_size) // group_size, group_size)
        if tuple(codes.shape) != expected:
            raise ValueError(f"grouped code shape is {tuple(codes.shape)}, expected {expected}")
        return codes.to(torch.int8).contiguous().cpu()
    if tuple(codes.shape) != shape:
        raise ValueError("flat code matrix shape does not match weights")
    padding = (-cols) % group_size
    value = codes.to(torch.int8).contiguous().cpu()
    if padding:
        value = F.pad(value, (0, padding))
    return value.view(rows, -1, group_size)


def pack_partial_matrix(
    fp_master: torch.Tensor,
    ternary_codes: torch.Tensor,
    scales: torch.Tensor,
    committed_mask: torch.Tensor,
    *,
    group_size: int = 128,
) -> PackedQ2Matrix:
    """Pack committed Q2 groups and exact BF16 fallback for every other group."""
    if fp_master.ndim != 2:
        raise ValueError("fp_master must be a matrix")
    if group_size <= 0 or group_size % 4:
        raise ValueError("group_size must be positive and divisible by four")
    shape = tuple(fp_master.shape)
    rows, cols = shape
    padding = (-cols) % group_size
    groups_per_row = (cols + padding) // group_size
    expected_group_shape = (rows, groups_per_row)
    mask = committed_mask.detach().bool().contiguous().cpu()
    if tuple(mask.shape) != expected_group_shape:
        raise ValueError("committed mask shape does not match weights")
    grouped_codes = _group_codes(ternary_codes, shape, group_size)
    scale_values = scales.detach().half().contiguous().cpu()
    if tuple(scale_values.shape) != expected_group_shape:
        raise ValueError("scale shape does not match weights")
    active_codes = grouped_codes[mask]
    if active_codes.numel() and not set(active_codes.unique().tolist()) <= {-1, 0, 1}:
        raise ValueError("committed groups contain non-ternary codes")
    active_scales = scale_values[mask]
    if not torch.isfinite(active_scales).all() or torch.any(active_scales <= 0):
        raise ValueError("committed scales must be finite and positive")
    master = fp_master.detach().bfloat16().contiguous().cpu()
    if padding:
        master = F.pad(master, (0, padding))
    grouped_master = master.view(rows, groups_per_row, group_size)
    fallback = grouped_master[~mask].contiguous()
    committed_groups = int(mask.sum().item())
    total_groups = mask.numel()
    packed_mask = (
        torch.empty(0, dtype=torch.uint8)
        if committed_groups in (0, total_groups)
        else pack_bool_mask(mask)
    )
    return PackedQ2Matrix(
        shape=shape,
        group_size=group_size,
        padding=padding,
        total_groups=total_groups,
        committed_groups=committed_groups,
        mask_packed=packed_mask,
        codes_packed=pack_ternary_codes(active_codes),
        scales_fp16=active_scales.contiguous(),
        fallback_bf16=fallback,
    )


def unpack_partial_matrix(
    packed: PackedQ2Matrix, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Reconstruct the exact deployed matrix, not the training master weights."""
    mask = packed.committed_mask()
    groups = torch.empty(
        (packed.total_groups, packed.group_size), dtype=torch.float32
    )
    codes = unpack_ternary_codes(
        packed.codes_packed, packed.committed_groups * packed.group_size
    ).view(packed.committed_groups, packed.group_size)
    groups[mask] = codes.float() * packed.scales_fp16.float().unsqueeze(-1)
    groups[~mask] = packed.fallback_bf16.float().view(
        packed.total_groups - packed.committed_groups, packed.group_size
    )
    rows, cols = packed.shape
    matrix = groups.view(rows, -1)[:, :cols]
    return matrix.to(dtype)


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous().cpu()
    return value.view(torch.uint8).numpy().tobytes()


def write_packed_matrix(path: str | Path, packed: PackedQ2Matrix) -> int:
    """Write the versioned reference format without pickle/container overhead."""
    output = Path(path)
    header = HEADER.pack(
        MAGIC,
        VERSION,
        packed.shape[0],
        packed.shape[1],
        packed.group_size,
        packed.padding,
        packed.total_groups,
        packed.committed_groups,
        packed.mask_packed.numel(),
        packed.codes_packed.numel(),
        packed.scales_fp16.numel() * 2,
        packed.fallback_bf16.numel() * 2,
    )
    with output.open("xb") as handle:
        handle.write(header)
        handle.write(_raw_bytes(packed.mask_packed))
        handle.write(_raw_bytes(packed.codes_packed))
        handle.write(_raw_bytes(packed.scales_fp16))
        handle.write(_raw_bytes(packed.fallback_bf16))
        handle.flush()
        os.fsync(handle.fileno())
    size = output.stat().st_size
    if size != packed.serialized_nbytes:
        raise RuntimeError(f"serialized size is {size}, expected {packed.serialized_nbytes}")
    return size


def _tensor_from_bytes(data: bytes, dtype: torch.dtype) -> torch.Tensor:
    if not data:
        return torch.empty(0, dtype=dtype)
    return torch.frombuffer(bytearray(data), dtype=dtype).clone()


def read_packed_matrix(path: str | Path) -> PackedQ2Matrix:
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("packed file is shorter than the header")
    (
        magic,
        version,
        rows,
        cols,
        group_size,
        padding,
        total_groups,
        committed_groups,
        mask_bytes,
        code_bytes,
        scale_bytes,
        fallback_bytes,
    ) = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported packed matrix format")
    if scale_bytes % 2 or fallback_bytes % 2:
        raise ValueError("FP16/BF16 payload length is not aligned")
    lengths = (mask_bytes, code_bytes, scale_bytes, fallback_bytes)
    if HEADER.size + sum(lengths) != len(data):
        raise ValueError("packed file length does not match its header")
    position = HEADER.size
    parts = []
    for length in lengths:
        parts.append(data[position : position + length])
        position += length
    packed = PackedQ2Matrix(
        shape=(rows, cols),
        group_size=group_size,
        padding=padding,
        total_groups=total_groups,
        committed_groups=committed_groups,
        mask_packed=_tensor_from_bytes(parts[0], torch.uint8),
        codes_packed=_tensor_from_bytes(parts[1], torch.uint8),
        scales_fp16=_tensor_from_bytes(parts[2], torch.float16),
        fallback_bf16=_tensor_from_bytes(parts[3], torch.bfloat16),
    )
    # Validate reserved slots immediately instead of deferring to inference.
    unpack_ternary_codes(
        packed.codes_packed, packed.committed_groups * packed.group_size
    )
    return packed


def true_artifact_bpw(
    matrices: Iterable[PackedQ2Matrix],
    *,
    extra_bytes: int = 0,
    denominator_weights: int | None = None,
    include_headers: bool = True,
) -> float:
    """Count every serialized byte against a declared model-weight denominator."""
    values = tuple(matrices)
    if not values:
        raise ValueError("at least one packed matrix is required")
    if extra_bytes < 0:
        raise ValueError("extra_bytes must be non-negative")
    weights = (
        denominator_weights
        if denominator_weights is not None
        else sum(matrix.shape[0] * matrix.shape[1] for matrix in values)
    )
    if weights <= 0:
        raise ValueError("denominator_weights must be positive")
    matrix_bytes = sum(
        matrix.serialized_nbytes if include_headers else matrix.payload_nbytes
        for matrix in values
    )
    return (matrix_bytes + extra_bytes) * 8 / weights
