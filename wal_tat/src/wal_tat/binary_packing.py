"""Versioned reference packing for strict groupwise binary matrices."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct

import torch

from .binary import pack_binary_codes, unpack_binary_codes


MAGIC = b"WALB1G1\0"
VERSION = 1
# magic, version, rows, cols, group_size, padding, total_groups,
# code_bytes, scale_bytes
HEADER = struct.Struct("<8sIIIHHQQQ")


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous().cpu()
    return value.view(torch.uint8).numpy().tobytes()


def _tensor_from_bytes(data: bytes, dtype: torch.dtype) -> torch.Tensor:
    if not data:
        return torch.empty(0, dtype=dtype)
    return torch.frombuffer(bytearray(data), dtype=dtype).clone()


@dataclass(frozen=True)
class PackedB1Matrix:
    """A fully binary matrix: one sign bit per padded weight plus FP16 scales."""

    shape: tuple[int, int]
    group_size: int
    padding: int
    total_groups: int
    codes_packed: torch.Tensor
    scales_fp16: torch.Tensor

    def __post_init__(self) -> None:
        rows, cols = self.shape
        if rows <= 0 or cols <= 0:
            raise ValueError("matrix shape must be positive")
        if self.group_size <= 0 or self.group_size % 8:
            raise ValueError("group_size must be positive and divisible by eight")
        if self.padding != (-cols) % self.group_size:
            raise ValueError("padding does not match shape/group_size")
        expected_groups = rows * ((cols + self.padding) // self.group_size)
        if self.total_groups != expected_groups:
            raise ValueError("total group count does not match shape")
        expected_codes = (self.total_groups * self.group_size + 7) // 8
        if (
            self.codes_packed.dtype != torch.uint8
            or self.codes_packed.numel() != expected_codes
        ):
            raise ValueError("invalid packed binary codes")
        if (
            self.scales_fp16.dtype != torch.float16
            or self.scales_fp16.numel() != self.total_groups
        ):
            raise ValueError("invalid FP16 scales")
        if not torch.isfinite(self.scales_fp16).all() or torch.any(
            self.scales_fp16 <= 0
        ):
            raise ValueError("binary scales must be finite and positive")

    @property
    def payload_nbytes(self) -> int:
        return self.codes_packed.numel() + self.scales_fp16.numel() * 2

    @property
    def serialized_nbytes(self) -> int:
        return HEADER.size + self.payload_nbytes

    def true_bpw(self, *, include_header: bool = True) -> float:
        size = self.serialized_nbytes if include_header else self.payload_nbytes
        return size * 8 / (self.shape[0] * self.shape[1])


def pack_binary_matrix(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    shape: tuple[int, int],
    group_size: int = 128,
) -> PackedB1Matrix:
    """Pack a fully committed ``{-1,+1}`` matrix without an FP fallback."""
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        raise ValueError("matrix shape must be positive")
    if group_size <= 0 or group_size % 8:
        raise ValueError("group_size must be positive and divisible by eight")
    padding = (-cols) % group_size
    groups_per_row = (cols + padding) // group_size
    expected_codes = (rows, groups_per_row, group_size)
    if tuple(codes.shape) != expected_codes:
        raise ValueError(
            f"grouped code shape is {tuple(codes.shape)}, expected {expected_codes}"
        )
    if tuple(scales.shape) != (rows, groups_per_row):
        raise ValueError("scale shape does not match grouped binary codes")
    code_values = codes.detach().to(torch.int8).contiguous().cpu()
    if not torch.all((code_values == -1) | (code_values == 1)):
        raise ValueError("binary codes must be in {-1, +1}")
    scale_values = scales.detach().half().contiguous().cpu()
    if not torch.isfinite(scale_values).all() or torch.any(scale_values <= 0):
        raise ValueError("binary scales must be finite and positive")
    return PackedB1Matrix(
        shape=shape,
        group_size=group_size,
        padding=padding,
        total_groups=rows * groups_per_row,
        codes_packed=pack_binary_codes(code_values),
        scales_fp16=scale_values,
    )


def unpack_binary_matrix(
    packed: PackedB1Matrix, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    count = packed.total_groups * packed.group_size
    codes = unpack_binary_codes(packed.codes_packed, count).view(
        packed.total_groups, packed.group_size
    )
    groups = codes.float() * packed.scales_fp16.float().view(-1, 1)
    rows, cols = packed.shape
    return groups.view(rows, -1)[:, :cols].to(dtype)


def write_packed_binary_matrix(path: str | Path, packed: PackedB1Matrix) -> int:
    output = Path(path)
    header = HEADER.pack(
        MAGIC,
        VERSION,
        packed.shape[0],
        packed.shape[1],
        packed.group_size,
        packed.padding,
        packed.total_groups,
        packed.codes_packed.numel(),
        packed.scales_fp16.numel() * 2,
    )
    with output.open("xb") as handle:
        handle.write(header)
        handle.write(_raw_bytes(packed.codes_packed))
        handle.write(_raw_bytes(packed.scales_fp16))
        handle.flush()
        os.fsync(handle.fileno())
    size = output.stat().st_size
    if size != packed.serialized_nbytes:
        raise RuntimeError(f"serialized size is {size}, expected {packed.serialized_nbytes}")
    return size


def read_packed_binary_matrix(path: str | Path) -> PackedB1Matrix:
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("packed binary file is shorter than the header")
    (
        magic,
        version,
        rows,
        cols,
        group_size,
        padding,
        total_groups,
        code_bytes,
        scale_bytes,
    ) = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported packed binary matrix format")
    if scale_bytes % 2:
        raise ValueError("FP16 scale payload length is not aligned")
    if HEADER.size + code_bytes + scale_bytes != len(data):
        raise ValueError("packed binary file length does not match its header")
    split = HEADER.size + code_bytes
    packed = PackedB1Matrix(
        shape=(rows, cols),
        group_size=group_size,
        padding=padding,
        total_groups=total_groups,
        codes_packed=_tensor_from_bytes(data[HEADER.size:split], torch.uint8),
        scales_fp16=_tensor_from_bytes(data[split:], torch.float16),
    )
    # Force decoding now so malformed bit payloads cannot be deferred to inference.
    unpack_binary_codes(
        packed.codes_packed, packed.total_groups * packed.group_size
    )
    return packed
