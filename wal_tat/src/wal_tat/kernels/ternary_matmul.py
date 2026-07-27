"""Тернарное матричное умножение на Triton: считает прямо по 2-битным кодам.

Зачем
-----
Обучающий граф (:class:`wal_tat.qat.quant.QuantLinear`) держит латентный вес в
fp32 и на каждом форварде разворачивает коды в плотную матрицу.  Это правильно
для обучения и бессмысленно для вывода: замер показал, что такая модель
**медленнее** bf16-учителя в 1.56 раза и ест вдвое больше памяти.

Здесь вес живёт в развёрнутом виде: коды по два бита и один fp16-масштаб на
группу из 128 входов.  Ядро читает их как есть, поэтому трафик по весам падает
в восемь раз против bf16 — а на авторегрессивном декодировании с батчем 1
именно он и есть узкое место.

Раскладка
---------
``codes``  — ``uint8[out, in // 4]``, четыре кода в байте, младшие биты первыми,
             хранимое значение равно ``код + 1`` (0 → −1, 1 → 0, 2 → +1);
``scales`` — ``float16[out, in // 128]``, по одному на группу.

Блок по K взят равным размеру группы, поэтому на один блок приходится ровно
один масштаб на строку — не приходится перечитывать таблицу масштабов внутри
цикла.

Корректность
------------
:func:`ternary_linear` обязана совпадать с плотным произведением
``x @ (codes * scales).T`` до последнего бита в fp32.  Проверяется в
:mod:`tests.test_ternary_kernel`; расхождение здесь означает, что развёрнутая
модель считает не то, что обученная, и все замеры качества недействительны.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - зависит от окружения
    HAVE_TRITON = False

__all__ = [
    "HAVE_TRITON",
    "pack_codes_rowwise",
    "unpack_codes_rowwise",
    "ternary_linear",
    "TernaryLinear",
]

GROUP_SIZE = 128


def pack_codes_rowwise(codes: torch.Tensor) -> torch.Tensor:
    """``int8[out, in]`` из {−1,0,+1} → ``uint8[out, in // 4]``.

    Раскладка совпадает с :func:`wal_tat.packing.pack_ternary_codes`: четыре
    кода в байте, младшие биты первыми, хранится ``код + 1``.
    """
    if codes.ndim != 2:
        raise ValueError("codes must be a matrix")
    out_features, in_features = codes.shape
    if in_features % 4:
        raise ValueError(f"in_features={in_features} must be a multiple of 4")
    values = (codes.to(torch.int16) + 1).to(torch.uint8)
    if int(values.max()) > 2:
        raise ValueError("ternary codes must be in {-1, 0, +1}")
    lanes = values.reshape(out_features, in_features // 4, 4)
    return (
        lanes[..., 0]
        | (lanes[..., 1] << 2)
        | (lanes[..., 2] << 4)
        | (lanes[..., 3] << 6)
    ).contiguous()


def unpack_codes_rowwise(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Обратная операция — нужна для эталона в тестах."""
    out_features = packed.shape[0]
    lanes = torch.stack(
        tuple((packed >> shift) & 0x03 for shift in (0, 2, 4, 6)), dim=-1
    )
    return (lanes.reshape(out_features, in_features).to(torch.int8) - 1).contiguous()


if HAVE_TRITON:

    @triton.jit
    def _ternary_matmul_kernel(
        x_ptr, codes_ptr, scales_ptr, bias_ptr, y_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_cn, stride_ck,
        stride_sn, stride_sk,
        stride_ym, stride_yn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # BLOCK_K приходит равным размеру группы, поэтому на итерацию
        # приходится ровно один масштаб на строку выхода.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        # позиции внутри группы: байт и сдвиг для распаковки двух бит
        offs_k = tl.arange(0, BLOCK_K)
        byte_off = offs_k // 4
        shift = (offs_k % 4) * 2

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        n_groups = tl.cdiv(K, BLOCK_K)
        for g in range(n_groups):
            k = g * BLOCK_K + offs_k
            mask_k = k < K

            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)

            byte_index = g * (BLOCK_K // 4) + byte_off
            packed = tl.load(
                codes_ptr + offs_n[:, None] * stride_cn + byte_index[None, :] * stride_ck,
                mask=mask_n[:, None] & mask_k[None, :],
                other=1,          # 1 кодирует ноль, поэтому хвост не влияет
            )
            code = ((packed >> shift[None, :]) & 0x03).to(tl.float32) - 1.0

            scale = tl.load(
                scales_ptr + offs_n * stride_sn + g * stride_sk,
                mask=mask_n, other=0.0,
            ).to(tl.float32)
            w = code * scale[:, None]

            acc += tl.dot(x, tl.trans(w), allow_tf32=False)

        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
        )


def ternary_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    block_m: int = 32,
    block_n: int = 64,
) -> torch.Tensor:
    """``x @ (codes * scales).T + bias`` без разворачивания весов в память.

    ``x``      — ``[..., in_features]`` любого плавающего типа;
    ``codes``  — ``uint8[out, in // 4]`` из :func:`pack_codes_rowwise`;
    ``scales`` — ``float16[out, in // 128]``.
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton недоступен; используйте плотный путь")
    if not x.is_cuda:
        raise ValueError("ternary_linear требует тензор на CUDA")
    original_shape = x.shape
    flat = x.reshape(-1, original_shape[-1]).contiguous()
    m, k = flat.shape
    n = codes.shape[0]
    if scales.shape[0] != n:
        raise ValueError("codes и scales расходятся по числу выходов")
    if k % GROUP_SIZE:
        raise ValueError(f"in_features={k} должно делиться на {GROUP_SIZE}")

    y = torch.empty((m, n), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _ternary_matmul_kernel[grid](
        flat, codes, scales, bias if bias is not None else flat, y,
        m, n, k,
        flat.stride(0), flat.stride(1),
        codes.stride(0), codes.stride(1),
        scales.stride(0), scales.stride(1),
        y.stride(0), y.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=GROUP_SIZE,
    )
    return y.reshape(*original_shape[:-1], n).to(x.dtype)


class TernaryLinear(torch.nn.Module):
    """Развёрнутая замена ``nn.Linear`` — держит коды, а не плотную матрицу."""

    def __init__(self, codes: torch.Tensor, scales: torch.Tensor,
                 bias: Optional[torch.Tensor]) -> None:
        super().__init__()
        self.register_buffer("codes", codes)
        self.register_buffer("scales", scales)
        self.register_buffer("bias_term", bias)
        self.out_features = int(codes.shape[0])
        self.in_features = int(codes.shape[1]) * 4

    @classmethod
    def from_quant_linear(cls, module) -> "TernaryLinear":
        """Собрать из обученного :class:`QuantLinear`."""
        codes, scales = module.export_codes()
        out_features = codes.shape[0]
        flat = codes.reshape(out_features, -1)[:, : module.in_features]
        packed = pack_codes_rowwise(flat.to(torch.int8))
        bias = None if module.bias is None else module.bias.detach().clone()
        return cls(packed.contiguous(),
                   scales.reshape(out_features, -1).to(torch.float16).contiguous(),
                   bias)

    def bytes_on_device(self) -> Tuple[int, int]:
        """(байты кодов, байты масштабов) — для честного отчёта о памяти."""
        return (self.codes.numel(), self.scales.numel() * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ternary_linear(x, self.codes, self.scales, self.bias_term)

    def extra_repr(self) -> str:
        bits = 2 + 16 / GROUP_SIZE
        return (f"in={self.in_features}, out={self.out_features}, "
                f"{bits:.3f} бит/вес")
