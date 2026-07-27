"""Pure policy primitives for the Whisper low-bit FastLane.

This module intentionally contains no model, CUDA, filesystem, or subprocess
code.  It keeps three decisions explicit and independently testable:

* a permissive *working* gate used to keep conversion moving;
* a separate *strict* confidence gate used for quality milestones;
* an authoritative frontier head that cannot be advanced from a stale parent
  or by a checkpoint that drops already converted matrices.

The default working gate implements the project's current ``1.15`` policy:
``candidate WER / BF16 WER <= 1.15``.  It must not be presented as a strict
quality claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable, Mapping, Sequence


class QualityTier(str, Enum):
    """The quality claim attached to a frontier checkpoint."""

    WORKING = "working"
    STRICT = "strict"


@dataclass(frozen=True)
class GateThresholds:
    """Independent thresholds for working progress and strict milestones."""

    working_max_relative_wer: float = 1.15
    strict_max_absolute_delta_wer: float = 0.005

    def __post_init__(self) -> None:
        if not isfinite(self.working_max_relative_wer):
            raise ValueError("working_max_relative_wer must be finite")
        if self.working_max_relative_wer < 1.0:
            raise ValueError("working_max_relative_wer must be at least 1")
        if not isfinite(self.strict_max_absolute_delta_wer):
            raise ValueError("strict_max_absolute_delta_wer must be finite")
        if self.strict_max_absolute_delta_wer < 0:
            raise ValueError("strict_max_absolute_delta_wer must be non-negative")


@dataclass(frozen=True)
class GateEvidence:
    """WER evidence needed by the two FastLane gates.

    ``upper_absolute_delta_wer`` is deliberately optional because a cheap
    working evaluation need not run a paired bootstrap.  A strict evaluation
    requires it.
    """

    baseline_wer: float
    candidate_wer: float
    upper_absolute_delta_wer: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("baseline_wer", self.baseline_wer),
            ("candidate_wer", self.candidate_wer),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.baseline_wer == 0:
            raise ValueError("baseline_wer must be positive")
        if self.upper_absolute_delta_wer is not None and not isfinite(
            self.upper_absolute_delta_wer
        ):
            raise ValueError("upper_absolute_delta_wer must be finite when provided")

    @property
    def relative_wer(self) -> float:
        return self.candidate_wer / self.baseline_wer

    @property
    def absolute_delta_wer(self) -> float:
        return self.candidate_wer - self.baseline_wer


@dataclass(frozen=True)
class GateDecision:
    """Auditable result of exactly one gate."""

    tier: QualityTier
    passed: bool
    metric: str
    observed: float
    limit: float
    evidence: GateEvidence

    def to_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier.value,
            "passed": self.passed,
            "metric": self.metric,
            "observed": self.observed,
            "limit": self.limit,
            "evidence": {
                "baseline_wer": self.evidence.baseline_wer,
                "candidate_wer": self.evidence.candidate_wer,
                "upper_absolute_delta_wer": self.evidence.upper_absolute_delta_wer,
            },
        }


class WhisperFastLaneGate:
    """Evaluate working and strict quality without conflating their semantics."""

    def __init__(self, thresholds: GateThresholds | None = None):
        self.thresholds = thresholds or GateThresholds()

    def evaluate_working(self, evidence: GateEvidence) -> GateDecision:
        """Apply only the relative-WER working gate (default: ``1.15``)."""

        observed = evidence.relative_wer
        limit = self.thresholds.working_max_relative_wer
        return GateDecision(
            tier=QualityTier.WORKING,
            passed=observed <= limit,
            metric="candidate_wer/baseline_wer",
            observed=observed,
            limit=limit,
            evidence=evidence,
        )

    def evaluate_strict(self, evidence: GateEvidence) -> GateDecision:
        """Apply only the strict upper-confidence-bound absolute-delta gate."""

        if evidence.upper_absolute_delta_wer is None:
            raise ValueError(
                "strict gate requires upper_absolute_delta_wer from a paired audit"
            )
        observed = evidence.upper_absolute_delta_wer
        limit = self.thresholds.strict_max_absolute_delta_wer
        return GateDecision(
            tier=QualityTier.STRICT,
            passed=observed <= limit,
            metric="upper_absolute_delta_wer",
            observed=observed,
            limit=limit,
            evidence=evidence,
        )


def working_budget_envelope(
    coverage: float,
    *,
    final_absolute_delta_wer: float = 0.005,
    early_headroom: float = 0.0075,
) -> float:
    """Return a linearly tightening diagnostic absolute-WER budget.

    The helper is optional and does not replace the explicit 1.15 working
    relative-WER gate.  At zero coverage it returns ``final + headroom``; at
    full coverage it returns the final budget.
    """

    if not isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must be finite and in [0, 1]")
    if (
        not isfinite(final_absolute_delta_wer)
        or final_absolute_delta_wer < 0
        or not isfinite(early_headroom)
        or early_headroom < 0
    ):
        raise ValueError("WER budgets must be finite and non-negative")
    return final_absolute_delta_wer + early_headroom * (1.0 - coverage)


ALLOWED_PACKAGE_SIZES = (4, 2, 1)


def initial_packages(
    matrix_ids: Iterable[str],
    *,
    allowed_sizes: Sequence[int] = ALLOWED_PACKAGE_SIZES,
) -> tuple[tuple[str, ...], ...]:
    """Partition matrices greedily into deployable 4/2/1 packages.

    For example, 11 matrices become ``4 + 4 + 2 + 1`` rather than producing a
    trailing unsupported package of three.
    """

    sizes = _validated_package_sizes(allowed_sizes)
    remaining = list(matrix_ids)
    if len(remaining) != len(set(remaining)):
        raise ValueError("matrix_ids must be unique")
    packages: list[tuple[str, ...]] = []
    offset = 0
    while offset < len(remaining):
        left = len(remaining) - offset
        size = next((candidate for candidate in sizes if candidate <= left), None)
        if size is None:
            raise ValueError("allowed_sizes cannot represent the remaining matrices")
        packages.append(tuple(remaining[offset : offset + size]))
        offset += size
    return tuple(packages)


def split_failed_package(
    package: Sequence[str],
    *,
    allowed_sizes: Sequence[int] = ALLOWED_PACKAGE_SIZES,
) -> tuple[tuple[str, ...], ...]:
    """Split a failed package by one FastLane level: ``4 -> 2 -> 1``.

    A failed singleton has no smaller strict low-bit transaction and therefore
    returns an empty tuple.  No matrix is duplicated, reordered, or dropped by
    a non-terminal split.
    """

    sizes = _validated_package_sizes(allowed_sizes)
    items = tuple(package)
    if len(items) != len(set(items)):
        raise ValueError("package matrix IDs must be unique")
    if len(items) not in sizes:
        raise ValueError(f"package size must be one of {sizes}")
    index = sizes.index(len(items))
    if index == len(sizes) - 1:
        return ()
    child_size = sizes[index + 1]
    if len(items) % child_size:
        raise ValueError("package cannot be evenly split into the next size")
    return tuple(
        items[offset : offset + child_size]
        for offset in range(0, len(items), child_size)
    )


def _validated_package_sizes(sizes: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in sizes)
    if not values or any(value <= 0 for value in values):
        raise ValueError("allowed package sizes must be positive")
    if values != tuple(sorted(set(values), reverse=True)):
        raise ValueError("allowed package sizes must be unique and descending")
    if values[-1] != 1:
        raise ValueError("allowed package sizes must end at one")
    return values


class FrontierPolicyError(ValueError):
    """Base class for rejected authoritative-frontier updates."""


class StaleParentError(FrontierPolicyError):
    """The candidate was derived from an obsolete frontier checkpoint."""


class NonDominatingFrontierError(FrontierPolicyError):
    """The candidate lost converted coverage or made no forward progress."""


@dataclass(frozen=True)
class FrontierRecord:
    """Self-contained metadata for an authoritative FastLane checkpoint."""

    model_id: str
    checkpoint_sha256: str
    parent_sha256: str | None
    converted_matrices: frozenset[str]
    converted_weights: int
    total_target_weights: int
    gate: GateDecision

    def __post_init__(self) -> None:
        object.__setattr__(self, "converted_matrices", frozenset(self.converted_matrices))
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        _validate_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        if self.parent_sha256 is not None:
            _validate_sha256(self.parent_sha256, "parent_sha256")
        if not 0 <= self.converted_weights <= self.total_target_weights:
            raise ValueError(
                "weights must satisfy 0 <= converted_weights <= total_target_weights"
            )
        if self.total_target_weights <= 0:
            raise ValueError("total_target_weights must be positive")
        if not self.gate.passed:
            raise ValueError("an authoritative frontier record must pass its gate")

    @property
    def coverage(self) -> float:
        return self.converted_weights / self.total_target_weights

    @property
    def quality_tier(self) -> QualityTier:
        return self.gate.tier

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "wal-tat.whisper-fastlane-frontier.v1",
            "model_id": self.model_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "parent_sha256": self.parent_sha256,
            "converted_matrices": sorted(self.converted_matrices),
            "converted_matrix_count": len(self.converted_matrices),
            "converted_weights": self.converted_weights,
            "total_target_weights": self.total_target_weights,
            "coverage": self.coverage,
            "quality_tier": self.quality_tier.value,
            "gate": self.gate.to_dict(),
        }


@dataclass(frozen=True)
class AuthoritativeFrontiers:
    """Conversion head plus the last non-overwritten strict milestone."""

    head: FrontierRecord
    strict_milestone: FrontierRecord | None = None

    def __post_init__(self) -> None:
        if self.strict_milestone is not None:
            if self.strict_milestone.quality_tier is not QualityTier.STRICT:
                raise ValueError("strict_milestone must have passed the strict gate")
            if self.strict_milestone.model_id != self.head.model_id:
                raise ValueError("head and strict_milestone must use the same model")

    @classmethod
    def bootstrap(cls, record: FrontierRecord) -> "AuthoritativeFrontiers":
        """Create frontier state from a validated initial checkpoint record."""

        strict = record if record.quality_tier is QualityTier.STRICT else None
        return cls(head=record, strict_milestone=strict)

    def advance(self, candidate: FrontierRecord) -> "AuthoritativeFrontiers":
        """Advance only from the current head with monotonic low-bit coverage."""

        current = self.head
        if candidate.parent_sha256 != current.checkpoint_sha256:
            raise StaleParentError(
                "candidate parent does not match the authoritative frontier head"
            )
        if candidate.checkpoint_sha256 == current.checkpoint_sha256:
            raise NonDominatingFrontierError("candidate checkpoint must be new")
        if candidate.model_id != current.model_id:
            raise NonDominatingFrontierError("candidate model_id changed")
        if candidate.total_target_weights != current.total_target_weights:
            raise NonDominatingFrontierError("candidate target weight total changed")
        if not current.converted_matrices.issubset(candidate.converted_matrices):
            raise NonDominatingFrontierError(
                "candidate reverted one or more converted matrices"
            )
        if candidate.converted_weights <= current.converted_weights:
            raise NonDominatingFrontierError(
                "candidate must strictly increase converted weight coverage"
            )
        strict = self.strict_milestone
        if candidate.quality_tier is QualityTier.STRICT:
            strict = candidate
        return AuthoritativeFrontiers(head=candidate, strict_milestone=strict)


def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase hexadecimal SHA-256")

