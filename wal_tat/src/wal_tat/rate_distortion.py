"""Deterministic physical-bit allocation over low-bit format candidates."""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from typing import Any, Iterable, Mapping


def groupwise_payload_bits(
    *,
    rows: int,
    columns: int,
    group_size: int,
    code_bits: int,
    scale_bits: int = 16,
) -> int:
    """Count padded codes and one scale per row/group."""
    if rows < 1 or columns < 1:
        raise ValueError("matrix dimensions must be positive")
    if group_size < 1 or code_bits < 1 or scale_bits < 0:
        raise ValueError("group_size/code_bits must be positive and scale_bits non-negative")
    groups = (columns + group_size - 1) // group_size
    return rows * groups * (group_size * code_bits + scale_bits)


@dataclass(frozen=True)
class FormatOption:
    unit: str
    format: str
    payload_bits: int
    distortion: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.unit:
            raise ValueError("unit must not be empty")
        if not self.format:
            raise ValueError("format must not be empty")
        if self.payload_bits < 0:
            raise ValueError("payload_bits must be non-negative")
        if not math.isfinite(self.distortion) or self.distortion < 0:
            raise ValueError("distortion must be finite and non-negative")


@dataclass(frozen=True)
class RateDistortionAllocation:
    selected: Mapping[str, FormatOption]
    total_payload_bits: int
    total_distortion: float
    target_payload_bits: int
    unused_payload_bits: int


def _pareto_options(options: Iterable[FormatOption]) -> list[FormatOption]:
    by_bits: dict[int, FormatOption] = {}
    for option in options:
        current = by_bits.get(option.payload_bits)
        if current is None or (
            option.distortion,
            option.format,
        ) < (
            current.distortion,
            current.format,
        ):
            by_bits[option.payload_bits] = option
    ordered = [by_bits[key] for key in sorted(by_bits)]
    frontier: list[FormatOption] = []
    best_distortion = float("inf")
    for option in ordered:
        if option.distortion < best_distortion:
            frontier.append(option)
            best_distortion = option.distortion
    return frontier


def allocate_rate_distortion(
    options: Iterable[FormatOption],
    *,
    target_payload_bits: int,
) -> RateDistortionAllocation:
    """Greedily allocate upgrades by distortion reduction per physical bit.

    Every unit starts from its smallest non-dominated representation.  The
    allocator then takes the best available adjacent Pareto upgrade that fits
    the remaining exact bit budget.  This is deterministic and scales to the
    hundreds of thousands of groups encountered in model-wide searches.
    """
    if target_payload_bits < 0:
        raise ValueError("target_payload_bits must be non-negative")
    grouped: dict[str, list[FormatOption]] = {}
    for option in options:
        grouped.setdefault(option.unit, []).append(option)
    if not grouped:
        raise ValueError("at least one format option is required")
    frontiers = {
        unit: _pareto_options(values) for unit, values in grouped.items()
    }
    if any(not values for values in frontiers.values()):
        raise ValueError("every unit must have a non-dominated option")

    indices = {unit: 0 for unit in frontiers}
    total_bits = sum(values[0].payload_bits for values in frontiers.values())
    if total_bits > target_payload_bits:
        raise ValueError(
            f"minimum representation requires {total_bits} bits, "
            f"above target {target_payload_bits}"
        )
    total_distortion = sum(
        values[0].distortion for values in frontiers.values()
    )
    heap: list[tuple[float, str, int, int, float]] = []

    def enqueue(unit: str) -> None:
        index = indices[unit]
        values = frontiers[unit]
        if index + 1 >= len(values):
            return
        current = values[index]
        following = values[index + 1]
        cost = following.payload_bits - current.payload_bits
        gain = current.distortion - following.distortion
        if cost <= 0 or gain <= 0:
            return
        heapq.heappush(
            heap,
            (-gain / cost, unit, index, cost, gain),
        )

    for unit in sorted(frontiers):
        enqueue(unit)
    while heap:
        _negative_rate, unit, source_index, cost, gain = heapq.heappop(heap)
        if indices[unit] != source_index:
            continue
        if total_bits + cost > target_payload_bits:
            continue
        indices[unit] += 1
        total_bits += cost
        total_distortion -= gain
        enqueue(unit)

    selected = {
        unit: frontiers[unit][indices[unit]] for unit in sorted(frontiers)
    }
    return RateDistortionAllocation(
        selected=selected,
        total_payload_bits=total_bits,
        total_distortion=max(total_distortion, 0.0),
        target_payload_bits=target_payload_bits,
        unused_payload_bits=target_payload_bits - total_bits,
    )
