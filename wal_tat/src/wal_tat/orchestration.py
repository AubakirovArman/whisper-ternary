"""Safety primitives for sequential multi-campaign WAL-TAT orchestration."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

from .campaign import atomic_write_json


WORKER_MARKERS = (
    "adaptive_campaign.py",
    "compensation_window.py",
    "verify_checkpoint.py",
)


def sha256_file(path: Path) -> str:
    """Hash a checkpoint without materializing it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_campaign_state(path: Path) -> dict:
    """Load one JSON campaign state as a mutable mapping."""
    return json.loads(path.read_text(encoding="utf-8"))


def common_campaign_frontier(state_paths: Sequence[Path]) -> tuple[Path, str]:
    """Require every campaign to reference the same existing checkpoint."""
    if not state_paths:
        raise ValueError("at least one campaign state is required")
    frontiers: list[tuple[Path, str]] = []
    for raw_path in state_paths:
        path = raw_path.expanduser().resolve(strict=True)
        state = load_campaign_state(path)
        frontier = state.get("frontier", {})
        checkpoint = Path(frontier.get("checkpoint", "")).expanduser().resolve(
            strict=True
        )
        digest = str(frontier.get("sha256", ""))
        if not digest:
            raise ValueError(f"campaign has no frontier SHA-256: {path}")
        frontiers.append((checkpoint, digest))
    first = frontiers[0]
    if any(frontier != first for frontier in frontiers[1:]):
        raise ValueError(f"campaign frontiers are not synchronized: {frontiers}")
    actual = sha256_file(first[0])
    if actual != first[1]:
        raise ValueError(
            f"frontier SHA-256 mismatch for {first[0]}: state={first[1]} actual={actual}"
        )
    return first


def synchronize_campaign_frontiers(
    state_paths: Sequence[Path], checkpoint: Path, digest: str | None = None
) -> str:
    """Atomically point all campaigns at one verified accepted checkpoint.

    Candidate-specific coverage is intentionally preserved in every state.
    """
    resolved_checkpoint = checkpoint.expanduser().resolve(strict=True)
    actual = sha256_file(resolved_checkpoint)
    if digest is not None and digest != actual:
        raise ValueError(
            f"refusing to publish mismatched frontier SHA-256: expected={digest} actual={actual}"
        )
    for raw_path in state_paths:
        state_path = raw_path.expanduser().resolve(strict=True)
        state = load_campaign_state(state_path)
        coverage = state.get("frontier", {}).get("candidate_coverage")
        state["frontier"] = {
            "checkpoint": str(resolved_checkpoint),
            "sha256": actual,
            "candidate_coverage": coverage,
        }
        state["updated_ns"] = time.time_ns()
        atomic_write_json(state_path, state)
    return actual


def active_campaign_workers(
    *,
    proc_root: Path = Path("/proc"),
    workspace: Path | None = None,
    exclude_pids: Iterable[int] = (),
) -> list[tuple[int, str]]:
    """List active WAL-TAT worker commands, optionally restricted to a workspace."""
    excluded = {os.getpid(), *map(int, exclude_pids)}
    workspace_text = None if workspace is None else str(workspace.resolve())
    workers: list[tuple[int, str]] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any(marker in command for marker in WORKER_MARKERS):
            continue
        if workspace_text is not None and workspace_text not in command:
            continue
        workers.append((pid, command.strip()))
    return sorted(workers)
