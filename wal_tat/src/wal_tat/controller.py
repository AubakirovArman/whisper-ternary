"""Quality gates that atomically commit or roll back a ternary transaction."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Mapping

import torch

from .transaction import TransactionalTernaryMatrix
from .wal import HashChainWAL


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    ratios: Dict[str, float]
    violations: Dict[str, float]


@dataclass(frozen=True)
class TransactionSizeDecision:
    """Auditable update of the next fraction of groups to attempt."""

    previous_fraction: float
    next_fraction: float
    reason: str
    worst_ratio: float
    gate_ratio: float
    normalized_headroom: float


class AdaptiveTransactionSizer:
    """Shrink risky ternary commits and cautiously regrow after safe streaks.

    Headroom is normalized by the gate's total allowance above one. This keeps
    the policy meaningful for strict cumulative gates such as 1.00194 as well
    as for wider diagnostic gates. A rollback always halves the atom; a pass
    close to the gate also halves it. Growth requires several roomy passes so
    that one unusually easy transaction cannot immediately undo the shrink.
    """

    def __init__(
        self,
        initial_fraction: float,
        *,
        minimum_fraction: float = 1 / 1024,
        maximum_fraction: float | None = None,
        shrink_factor: float = 0.5,
        grow_factor: float = 2.0,
        tight_headroom: float = 0.2,
        roomy_headroom: float = 0.75,
        grow_after: int = 2,
        roomy_passes: int = 0,
    ):
        maximum_fraction = initial_fraction if maximum_fraction is None else maximum_fraction
        if not 0 < minimum_fraction <= initial_fraction <= maximum_fraction <= 1:
            raise ValueError("fractions must satisfy 0 < minimum <= initial <= maximum <= 1")
        if not 0 < shrink_factor < 1:
            raise ValueError("shrink_factor must be in (0, 1)")
        if grow_factor <= 1:
            raise ValueError("grow_factor must be greater than 1")
        if not 0 <= tight_headroom < roomy_headroom <= 1:
            raise ValueError("headroom thresholds must satisfy 0 <= tight < roomy <= 1")
        if grow_after < 1:
            raise ValueError("grow_after must be positive")
        if not isinstance(roomy_passes, int) or not 0 <= roomy_passes < grow_after:
            raise ValueError("roomy_passes must satisfy 0 <= roomy_passes < grow_after")
        self.current_fraction = float(initial_fraction)
        self.minimum_fraction = float(minimum_fraction)
        self.maximum_fraction = float(maximum_fraction)
        self.shrink_factor = float(shrink_factor)
        self.grow_factor = float(grow_factor)
        self.tight_headroom = float(tight_headroom)
        self.roomy_headroom = float(roomy_headroom)
        self.grow_after = int(grow_after)
        self._roomy_passes = int(roomy_passes)

    @property
    def roomy_passes(self) -> int:
        """Number of consecutive roomy passes retained for the next decision."""

        return self._roomy_passes

    def observe(
        self, *, passed: bool, worst_ratio: float, gate_ratio: float
    ) -> TransactionSizeDecision:
        if gate_ratio <= 1:
            raise ValueError("gate_ratio must be greater than 1")
        if worst_ratio <= 0:
            raise ValueError("worst_ratio must be positive")
        previous = self.current_fraction
        headroom = (gate_ratio - worst_ratio) / (gate_ratio - 1)
        normalized_headroom = min(1.0, max(0.0, headroom))

        if not passed or worst_ratio > gate_ratio:
            self._roomy_passes = 0
            next_fraction = max(self.minimum_fraction, previous * self.shrink_factor)
            reason = "rollback_shrink"
        elif normalized_headroom <= self.tight_headroom:
            self._roomy_passes = 0
            next_fraction = max(self.minimum_fraction, previous * self.shrink_factor)
            reason = "tight_gate_shrink"
        elif normalized_headroom >= self.roomy_headroom:
            self._roomy_passes += 1
            if self._roomy_passes >= self.grow_after:
                next_fraction = min(self.maximum_fraction, previous * self.grow_factor)
                self._roomy_passes = 0
                reason = "safe_streak_grow"
            else:
                next_fraction = previous
                reason = "safe_streak_hold"
        else:
            self._roomy_passes = 0
            next_fraction = previous
            reason = "middle_headroom_hold"

        self.current_fraction = next_fraction
        return TransactionSizeDecision(
            previous_fraction=previous,
            next_fraction=next_fraction,
            reason=reason,
            worst_ratio=float(worst_ratio),
            gate_ratio=float(gate_ratio),
            normalized_headroom=normalized_headroom,
        )

    def group_count(self, total_groups: int) -> int:
        if total_groups < 1:
            raise ValueError("total_groups must be positive")
        return max(1, round(total_groups * self.current_fraction))


class RatioGate:
    """Accept when every candidate loss is within its baseline ratio limit."""

    def __init__(self, limits: float | Mapping[str, float] = 1.02):
        self.limits = float(limits) if isinstance(limits, (float, int)) else dict(limits)

    def evaluate(
        self, baseline: Mapping[str, float], candidate: Mapping[str, float]
    ) -> GateDecision:
        if baseline.keys() != candidate.keys() or not baseline:
            raise ValueError("baseline and candidate must have identical non-empty domains")
        ratios: Dict[str, float] = {}
        violations: Dict[str, float] = {}
        for domain, before in baseline.items():
            if before <= 0:
                raise ValueError(f"baseline for {domain!r} must be positive")
            ratio = float(candidate[domain]) / float(before)
            limit = self.limits if isinstance(self.limits, float) else self.limits[domain]
            ratios[domain] = ratio
            if ratio > limit:
                violations[domain] = ratio - limit
        return GateDecision(not violations, ratios, violations)


class TransactionController:
    """Connect an in-memory matrix transaction to the durable WAL v2."""

    def __init__(self, wal: HashChainWAL, matrix_name: str, gate: RatioGate | None = None):
        self.wal = wal
        self.matrix_name = matrix_name
        self.gate = gate or RatioGate()

    def begin(
        self,
        matrix: TransactionalTernaryMatrix,
        mask: torch.Tensor,
        *,
        selector: str,
        selected_damage_mean: float | None = None,
    ) -> str:
        transaction_id = matrix.begin(mask)
        mask_bytes = mask.detach().to("cpu", dtype=torch.uint8).numpy().tobytes()
        self.wal.append(
            "begin",
            transaction_id,
            {
                "matrix": self.matrix_name,
                "selector": selector,
                "groups": int(mask.sum().item()),
                "total_groups": mask.numel(),
                "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
                "selected_damage_mean": selected_damage_mean,
            },
        )
        return transaction_id

    def progress(
        self,
        matrix: TransactionalTernaryMatrix,
        *,
        step: int,
        pressure: float,
        temperature: float,
        loss: float | None = None,
    ) -> None:
        if not matrix.in_transaction:
            raise RuntimeError("no active transaction")
        matrix.set_candidate_state(pressure, temperature)
        self.wal.append(
            "progress",
            matrix.transaction_id,
            {
                "matrix": self.matrix_name,
                "step": int(step),
                "pressure": float(pressure),
                "temperature": float(temperature),
                "loss": loss,
                "code_churn": matrix.current_code_churn(),
            },
        )

    def decide(
        self,
        matrix: TransactionalTernaryMatrix,
        *,
        baseline: Mapping[str, float],
        candidate: Mapping[str, float],
    ) -> GateDecision:
        if not matrix.in_transaction:
            raise RuntimeError("no active transaction")
        transaction_id = matrix.transaction_id
        decision = self.gate.evaluate(baseline, candidate)
        kind = "commit" if decision.passed else "rollback"
        gate_payload = {
            "matrix": self.matrix_name,
            "baseline": dict(baseline),
            "candidate": dict(candidate),
            "ratios": decision.ratios,
            "violations": decision.violations,
        }
        # This record is durable before the in-memory state transition. If the
        # process stops here, recovery can distinguish an intent from an
        # applied commit/rollback and restart from the last model checkpoint.
        self.wal.append(f"{kind}_intent", transaction_id, gate_payload)
        if decision.passed:
            result = matrix.commit()
        else:
            result = matrix.rollback()
        self.wal.append(
            kind,
            transaction_id,
            {
                **gate_payload,
                **result,
            },
        )
        return decision
