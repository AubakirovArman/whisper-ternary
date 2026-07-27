"""Transactional matrix wrappers for staged ternarization."""
from __future__ import annotations

import uuid
from typing import Dict, Mapping, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantization import hard_codes_scales, hestia_quantize, padded_grouped


class TransactionalTernaryMatrix(nn.Module):
    """A matrix whose groups can be ternarized atomically.

    Unselected groups are exact, detached FP weights. Candidate groups are the
    only trainable part of an active transaction. Committed groups are frozen
    hard codes in ``{-1, 0, +1}`` with one scale per group.
    """

    def __init__(self, weight: torch.Tensor, *, group_size: int = 128):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("weight must be a matrix")
        self.compute_dtype = weight.dtype
        self.master_weight = nn.Parameter(weight.detach().float().clone())
        self.group_size = int(group_size)
        grouped, self.padding, self.effective_group_size = padded_grouped(
            weight.detach(), self.group_size
        )
        self.group_scale = nn.Parameter(grouped.abs().mean(-1).clamp_min(1e-5))
        mask_shape = grouped.shape[:2]
        self.register_buffer("committed_mask", torch.zeros(mask_shape, dtype=torch.bool))
        self.register_buffer("candidate_mask", torch.zeros(mask_shape, dtype=torch.bool))
        self.register_buffer("committed_codes", torch.zeros_like(grouped, dtype=torch.int8))
        self.register_buffer("candidate_codes", torch.zeros_like(grouped, dtype=torch.int8))
        self.register_buffer("candidate_override_mask", torch.zeros(mask_shape, dtype=torch.bool))
        self.register_buffer("continuous_mask", torch.zeros(mask_shape, dtype=torch.bool))
        self.candidate_pressure = 0.0
        self.candidate_temperature = 0.3508855606815209
        self.transaction_id: Optional[str] = None
        self._snapshot_weight: Optional[torch.Tensor] = None
        self._snapshot_scale: Optional[torch.Tensor] = None
        self._snapshot_codes: Optional[torch.Tensor] = None
        self._snapshot_committed_mask: Optional[torch.Tensor] = None
        self._snapshot_committed_codes: Optional[torch.Tensor] = None
        self._continuous_snapshot_weight: Optional[torch.Tensor] = None

    @property
    def out_features(self) -> int:
        return self.master_weight.shape[0]

    @property
    def in_features(self) -> int:
        return self.master_weight.shape[1]

    @property
    def groups(self) -> int:
        return self.committed_mask.shape[1]

    @property
    def in_transaction(self) -> bool:
        return self.transaction_id is not None

    @property
    def in_continuous_compensation(self) -> bool:
        return self._continuous_snapshot_weight is not None

    @torch.no_grad()
    def begin_continuous_compensation(self, mask: torch.Tensor) -> int:
        """Open a rollback-safe BF16 compensation window on uncommitted groups.

        The deployed forward value remains the exact master weight. Only the
        selected groups receive gradients. This lets adjacent FP groups absorb
        a ternary transaction without falsely increasing ternary coverage.
        """
        if self.in_transaction or self.in_continuous_compensation:
            raise RuntimeError("matrix already has an active transaction")
        mask = mask.to(self.continuous_mask.device, dtype=torch.bool)
        if mask.shape != self.continuous_mask.shape:
            raise ValueError("continuous compensation mask has the wrong shape")
        if not mask.any():
            raise ValueError("continuous compensation needs at least one group")
        if torch.any(mask & self.committed_mask):
            raise ValueError("continuous compensation cannot modify committed groups")
        grouped, _, _ = padded_grouped(self.master_weight.detach(), self.group_size)
        self._continuous_snapshot_weight = grouped[mask].clone()
        self.continuous_mask.copy_(mask)
        return int(mask.sum().item())

    @torch.no_grad()
    def commit_continuous_compensation(self) -> Dict[str, int]:
        if not self.in_continuous_compensation:
            raise RuntimeError("no active continuous compensation")
        groups = int(self.continuous_mask.sum().item())
        self.continuous_mask.zero_()
        self._continuous_snapshot_weight = None
        return {"groups": groups}

    @torch.no_grad()
    def rollback_continuous_compensation(self) -> Dict[str, int]:
        if not self.in_continuous_compensation:
            raise RuntimeError("no active continuous compensation")
        mask = self.continuous_mask.clone()
        grouped, _, _ = padded_grouped(self.master_weight.detach(), self.group_size)
        grouped[mask] = self._continuous_snapshot_weight
        self.master_weight.copy_(
            grouped.reshape(self.out_features, -1)[:, : self.in_features]
        )
        groups = int(mask.sum().item())
        self.continuous_mask.zero_()
        self._continuous_snapshot_weight = None
        return {"groups": groups}

    @torch.no_grad()
    def begin(
        self,
        mask: torch.Tensor,
        *,
        transaction_id: Optional[str] = None,
        allow_reopen: bool = False,
    ) -> str:
        """Open a transaction and snapshot exactly the selected groups."""
        if self.in_transaction:
            raise RuntimeError("a transaction is already active")
        mask = mask.to(self.candidate_mask.device, dtype=torch.bool)
        if mask.shape != self.candidate_mask.shape:
            raise ValueError("candidate mask has the wrong shape")
        overlap = mask & self.committed_mask
        if torch.any(overlap) and not allow_reopen:
            raise ValueError("candidate mask overlaps committed groups")
        if not mask.any():
            raise ValueError("a transaction must contain at least one group")
        self.candidate_mask.copy_(mask)
        grouped, _, _ = padded_grouped(self.master_weight.detach(), self.group_size)
        self._snapshot_weight = grouped[mask].clone()
        self._snapshot_scale = self.group_scale.detach()[mask].clone()
        codes = (grouped / self.group_scale.detach().unsqueeze(-1)).round().clamp(-1, 1)
        codes = torch.where(overlap.unsqueeze(-1), self.committed_codes, codes.to(torch.int8))
        self._snapshot_codes = codes[mask].to(torch.int8).clone()
        self._snapshot_committed_mask = self.committed_mask[mask].clone()
        self._snapshot_committed_codes = self.committed_codes[mask].clone()
        if overlap.any():
            # A reopened group starts at the exact deployed hard value. The
            # soft-to-hard path can then move its master/code, while rollback
            # restores both the former master and the committed code exactly.
            hard = self.committed_codes.float() * self.group_scale.detach().unsqueeze(-1)
            grouped[overlap] = hard[overlap]
            self.master_weight.copy_(
                grouped.reshape(self.out_features, -1)[:, : self.in_features]
            )
            self.committed_mask[overlap] = False
        self.transaction_id = transaction_id or uuid.uuid4().hex[:12]
        self.candidate_pressure = 0.0
        self.candidate_temperature = 0.3508855606815209
        return self.transaction_id

    @torch.no_grad()
    def set_candidate_state(self, pressure: float, temperature: float) -> None:
        if not self.in_transaction:
            raise RuntimeError("no active transaction")
        self.candidate_pressure = float(min(max(pressure, 0.0), 1.0))
        self.candidate_temperature = float(max(temperature, 0.0))

    @torch.no_grad()
    def set_candidate_codes(self, codes: torch.Tensor, scales: torch.Tensor) -> None:
        """Use externally searched codes/scales for the active groups."""
        if not self.in_transaction:
            raise RuntimeError("no active transaction")
        value = codes.to(self.candidate_codes.device, dtype=torch.int8)
        if value.shape == self.master_weight.shape:
            if self.padding:
                value = F.pad(value, (0, self.padding))
            value = value.view_as(self.candidate_codes)
        if value.shape != self.candidate_codes.shape:
            raise ValueError("codes must have matrix or padded grouped shape")
        scale_value = scales.to(self.group_scale.device, dtype=self.group_scale.dtype)
        if scale_value.shape != self.group_scale.shape:
            raise ValueError("scales have the wrong shape")
        self.candidate_codes[self.candidate_mask] = value[self.candidate_mask]
        self.group_scale[self.candidate_mask] = scale_value[self.candidate_mask]
        self.candidate_override_mask.copy_(self.candidate_mask)

    def effective_weight(self) -> torch.Tensor:
        """Materialize the mixed FP/candidate/committed matrix for training."""
        grouped, _, _ = padded_grouped(self.master_weight, self.group_size)
        result = grouped.detach()
        committed = self.committed_codes.float() * self.group_scale.detach().unsqueeze(-1)
        result = torch.where(self.committed_mask.unsqueeze(-1), committed, result)

        if self.in_transaction:
            candidate = hestia_quantize(
                self.master_weight,
                group_size=self.group_size,
                pressure=self.candidate_pressure,
                temperature=self.candidate_temperature,
                scales=self.group_scale,
            ).float()
            if self.padding:
                candidate = F.pad(candidate, (0, self.padding))
            candidate = candidate.view_as(grouped)
            if self.candidate_override_mask.any():
                override_hard = (
                    self.candidate_codes.detach().float() * self.group_scale.unsqueeze(-1)
                    + grouped
                    - grouped.detach()
                )
                override = torch.lerp(grouped, override_hard, self.candidate_pressure)
                candidate = torch.where(
                    self.candidate_override_mask.unsqueeze(-1), override, candidate
                )
            result = torch.where(self.candidate_mask.unsqueeze(-1), candidate, result)
        if self.in_continuous_compensation:
            # Zero-valued STE: forward is unchanged, while gradients reach
            # only selected uncommitted master weights.
            gradient_proxy = grouped - grouped.detach()
            result = result + torch.where(
                self.continuous_mask.unsqueeze(-1),
                gradient_proxy,
                torch.zeros_like(gradient_proxy),
            )
        return result.reshape(self.out_features, -1)[:, : self.in_features].to(self.compute_dtype)

    @torch.no_grad()
    def current_code_churn(self) -> float:
        if not self.in_transaction or self._snapshot_codes is None:
            return 0.0
        grouped, _, _ = padded_grouped(self.master_weight.detach(), self.group_size)
        codes = (grouped / self.group_scale.detach().unsqueeze(-1)).round().clamp(-1, 1)
        codes = torch.where(self.candidate_override_mask.unsqueeze(-1), self.candidate_codes, codes)
        current = codes[self.candidate_mask].to(torch.int8)
        return float((current != self._snapshot_codes).float().mean().item())

    @torch.no_grad()
    def commit(self) -> Dict[str, object]:
        """Freeze candidate codes and make the transaction durable in memory."""
        if not self.in_transaction:
            raise RuntimeError("no active transaction")
        transaction_id = self.transaction_id
        count = int(self.candidate_mask.sum().item())
        churn = self.current_code_churn()
        grouped, _, _ = padded_grouped(self.master_weight.detach(), self.group_size)
        codes = (
            grouped / self.group_scale.detach().abs().clamp_min(1e-5).unsqueeze(-1)
        ).round().clamp(-1, 1).to(torch.int8)
        codes = torch.where(self.candidate_override_mask.unsqueeze(-1), self.candidate_codes, codes)
        self.committed_codes[self.candidate_mask] = codes[self.candidate_mask]
        self.committed_mask.logical_or_(self.candidate_mask)
        self._clear_transaction()
        return {"transaction_id": transaction_id, "groups": count, "code_churn": churn}

    @torch.no_grad()
    def rollback(self) -> Dict[str, object]:
        """Restore candidate weights/scales exactly and close the transaction."""
        if not self.in_transaction:
            raise RuntimeError("no active transaction")
        transaction_id = self.transaction_id
        mask = self.candidate_mask.clone()
        grouped, _, _ = padded_grouped(self.master_weight.detach(), self.group_size)
        grouped[mask] = self._snapshot_weight
        self.master_weight.copy_(grouped.reshape(self.out_features, -1)[:, : self.in_features])
        self.group_scale[mask] = self._snapshot_scale
        self.committed_mask[mask] = self._snapshot_committed_mask
        self.committed_codes[mask] = self._snapshot_committed_codes
        count = int(mask.sum().item())
        self._clear_transaction()
        return {"transaction_id": transaction_id, "groups": count}

    @torch.no_grad()
    def _clear_transaction(self) -> None:
        self.candidate_mask.zero_()
        self.candidate_override_mask.zero_()
        self.transaction_id = None
        self._snapshot_weight = None
        self._snapshot_scale = None
        self._snapshot_codes = None
        self._snapshot_committed_mask = None
        self._snapshot_committed_codes = None
        self.candidate_pressure = 0.0
        self.candidate_temperature = 0.3508855606815209

    @torch.no_grad()
    def codes_scales(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return hard codes/scales; useful for exporters and validation."""
        codes, scales = hard_codes_scales(
            self.master_weight, self.group_size, self.group_scale
        )
        if self.padding:
            codes = F.pad(codes, (0, self.padding))
        grouped = codes.view_as(self.committed_codes)
        grouped = torch.where(self.committed_mask.unsqueeze(-1), self.committed_codes, grouped)
        return grouped.reshape(self.out_features, -1)[:, : self.in_features], scales

    def projected_mixed_bpw(self, *, fp_bits: int = 16, scale_bits: int = 16) -> float:
        """Matrix-only bpw if committed groups are packed and the rest stay FP."""
        committed = int(self.committed_mask.sum().item())
        total_groups = self.committed_mask.numel()
        low_group_bits = 2 * self.effective_group_size + scale_bits
        fp_group_bits = fp_bits * self.effective_group_size
        return (committed * low_group_bits + (total_groups - committed) * fp_group_bits) / (
            total_groups * self.effective_group_size
        )


class TransactionalTernaryLinear(nn.Module):
    """Drop-in ``nn.Linear`` wrapper around a transactional matrix."""

    def __init__(self, matrix: TransactionalTernaryMatrix, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.matrix = matrix
        self.in_features = matrix.in_features
        self.out_features = matrix.out_features
        self.bias = None if bias is None else nn.Parameter(
            bias.detach().clone(), requires_grad=False
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, group_size: int = 128):
        return cls(
            TransactionalTernaryMatrix(linear.weight, group_size=group_size),
            linear.bias,
        )

    @property
    def weight(self) -> nn.Parameter:
        return self.matrix.master_weight

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.matrix.effective_weight().to(value.dtype), self.bias)


class AtomicTernaryTransaction:
    """One commit/rollback boundary spanning multiple ternary matrices."""

    def __init__(self, matrices: Mapping[str, TransactionalTernaryMatrix]):
        if not matrices:
            raise ValueError("an atomic transaction needs at least one matrix")
        self.matrices = dict(matrices)
        self.transaction_id: Optional[str] = None

    @property
    def in_transaction(self) -> bool:
        return self.transaction_id is not None

    def begin(
        self,
        masks: Mapping[str, torch.Tensor],
        *,
        reopen: Optional[Set[str]] = None,
        transaction_id: Optional[str] = None,
    ) -> str:
        if self.in_transaction:
            raise RuntimeError("an atomic transaction is already active")
        if not masks or not set(masks).issubset(self.matrices):
            raise ValueError("masks must name one or more registered matrices")
        reopen = set(reopen or ())
        if not reopen.issubset(masks):
            raise ValueError("reopen names must also be present in masks")
        identifier = transaction_id or uuid.uuid4().hex[:12]
        begun = []
        try:
            for name, mask in masks.items():
                self.matrices[name].begin(
                    mask,
                    transaction_id=identifier,
                    allow_reopen=name in reopen,
                )
                begun.append(name)
        except Exception:
            for name in reversed(begun):
                self.matrices[name].rollback()
            raise
        self.transaction_id = identifier
        return identifier

    @torch.no_grad()
    def set_candidate_state(self, pressure: float, temperature: float) -> None:
        if not self.in_transaction:
            raise RuntimeError("no active atomic transaction")
        for matrix in self.matrices.values():
            if matrix.in_transaction:
                matrix.set_candidate_state(pressure, temperature)

    @torch.no_grad()
    def current_code_churn(self) -> Dict[str, float]:
        return {
            name: matrix.current_code_churn()
            for name, matrix in self.matrices.items()
            if matrix.in_transaction
        }

    @torch.no_grad()
    def commit(self) -> Dict[str, Dict[str, object]]:
        if not self.in_transaction:
            raise RuntimeError("no active atomic transaction")
        results = {
            name: matrix.commit()
            for name, matrix in self.matrices.items()
            if matrix.in_transaction
        }
        self.transaction_id = None
        return results

    @torch.no_grad()
    def rollback(self) -> Dict[str, Dict[str, object]]:
        if not self.in_transaction:
            raise RuntimeError("no active atomic transaction")
        results = {
            name: matrix.rollback()
            for name, matrix in reversed(tuple(self.matrices.items()))
            if matrix.in_transaction
        }
        self.transaction_id = None
        return results
