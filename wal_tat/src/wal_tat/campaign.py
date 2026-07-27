"""Durable helpers for multi-transaction WAL-TAT conversion campaigns."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch


def worst_ratio(documents: Sequence[Mapping]) -> float:
    """Return the worst NLL ratio across independent verification documents."""
    values = [
        float(value)
        for document in documents
        for value in document.get("ratios", {}).values()
    ]
    if not values:
        raise ValueError("at least one audit ratio is required")
    return max(values)


def validate_checkpoint_deletion_target(path: Path, checkpoint_dir: Path) -> Path:
    """Resolve an exact, narrow checkpoint target that is safe to unlink."""
    raw = path.expanduser()
    if raw.is_symlink():
        raise ValueError("checkpoint cleanup refuses symlinks")
    resolved_dir = checkpoint_dir.expanduser().resolve(strict=True)
    resolved = raw.resolve(strict=True)
    if resolved.parent != resolved_dir:
        raise ValueError("checkpoint is outside the exact checkpoint directory")
    if not resolved.name.startswith("wal-tat-") or resolved.suffix != ".pt":
        raise ValueError("checkpoint name must match wal-tat-*.pt")
    if not resolved.is_file():
        raise ValueError("checkpoint cleanup target is not a regular file")
    return resolved


def atomic_write_json(path: Path, payload: Mapping) -> None:
    """Durably replace campaign state without exposing partial JSON."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def accepted_weight_counts(checkpoint: Mapping) -> dict[str, int]:
    """Count exact hard-ternary weights from committed group masks."""
    counts: dict[str, int] = {}
    for name, entry in checkpoint.get("matrices", {}).items():
        rows, columns = map(int, entry["shape"])
        group_size = int(entry["group_size"])
        mask = torch.as_tensor(entry["committed_mask"], dtype=torch.bool)
        expected_groups = (columns + group_size - 1) // group_size
        if tuple(mask.shape) != (rows, expected_groups):
            raise ValueError(f"committed mask shape mismatch for {name}")
        full_groups, remainder = divmod(columns, group_size)
        count = int(mask[:, :full_groups].sum().item()) * group_size
        if remainder:
            count += int(mask[:, full_groups].sum().item()) * remainder
        counts[name] = count
    return counts


def coverage_proportional_nll_gate(
    accepted_weights: int, total_weights: int, full_model_nll_budget: float
) -> float:
    """Allocate a declared full-model NLL budget by converted weight coverage."""
    if not 0 <= accepted_weights <= total_weights or total_weights <= 0:
        raise ValueError("weights must satisfy 0 <= accepted <= total and total > 0")
    if full_model_nll_budget <= 0:
        raise ValueError("full-model NLL budget must be positive")
    return 1.0 + full_model_nll_budget * accepted_weights / total_weights
