from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from advance_whisper_fastlane_manifest import advance_manifest
from validate_whisper_fastlane_manifest import (
    _group_maps,
    sha256_file,
    validate_manifest,
)


def _save(path: Path, payload: dict) -> Path:
    torch.save(payload, path)
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    matrix_by_group, weights_by_group = _group_maps()
    remaining_groups = [
        group
        for group in matrix_by_group
        if not group.split(".", 2)[2].startswith("mlp_")
    ][-2:]
    remaining_names = [matrix_by_group[group] for group in remaining_groups]
    current_names = set(matrix_by_group.values()) - set(remaining_names)

    current = _save(
        tmp_path / "frontier190.pt",
        {
            "accepted": True,
            "provisional": False,
            "quality_tier": "working",
            "strict_accepted": False,
            "matrices": {name: {} for name in current_names},
        },
    )
    current_sha = sha256_file(current)
    candidates = []
    for rank, (group_id, matrix_name) in enumerate(
        zip(remaining_groups, remaining_names, strict=True), start=1
    ):
        source = _save(
            tmp_path / f"{rank}.candidate.pt",
            {
                "accepted": False,
                "provisional": True,
                "parent_checkpoint": str(current.resolve()),
                "candidate_groups": [group_id],
                "matrices": {
                    **{name: {} for name in current_names},
                    matrix_name: {
                        "precision": "t3",
                        "codes": torch.zeros(
                            weights_by_group[group_id], dtype=torch.int8
                        ),
                    },
                },
            },
        )
        stack, block, category = group_id.split(".", 2)
        candidates.append(
            {
                "rank": rank,
                "group_id": group_id,
                "matrix_name": matrix_name,
                "stack": stack,
                "block": int(block),
                "category": category,
                "weights": weights_by_group[group_id],
                "source_checkpoint": str(source.resolve()),
                "source_checkpoint_sha256": sha256_file(source),
                "source_parent_path": str(current.resolve()),
                "source_parent_sha256": current_sha,
                "evidence": {"tier": f"tier_{rank}"},
            }
        )

    manifest = tmp_path / "remaining2.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "whisper-fastlane-singleton-source-manifest",
                "model": "openai/whisper-small",
                "precision": "t3",
                "group_size": 128,
                "current_frontier": {
                    "path": str(current.resolve()),
                    "sha256": current_sha,
                    "matrix_count": 190,
                    "quality_tier": "working",
                    "strict_accepted": False,
                },
                "source_parent": {
                    "path": str(current.resolve()),
                    "sha256": current_sha,
                    "matrix_count": 190,
                },
                "ordering_policy": ["preserve evidence order"],
                "expected": {
                    "target_matrix_count": 192,
                    "current_matrix_count": 190,
                    "remaining_matrix_count": 2,
                    "remaining_target_weights": sum(
                        weights_by_group[group] for group in remaining_groups
                    ),
                    "evidence_tier_counts": {"tier_1": 1, "tier_2": 1},
                },
                "candidates": candidates,
            }
        ),
        encoding="utf-8",
    )

    new = _save(
        tmp_path / "frontier191.pt",
        {
            "accepted": True,
            "provisional": False,
            "quality_tier": "working",
            "strict_accepted": False,
            "parent_checkpoint": str(current.resolve()),
            "parent_checkpoint_sha256": current_sha,
            "matrices": {
                **{name: {} for name in current_names},
                remaining_names[0]: {"precision": "t3"},
            },
        },
    )
    return manifest, new, remaining_groups


def test_dynamic_validator_accepts_190_plus_2_split(tmp_path: Path) -> None:
    manifest, _, _ = _fixture(tmp_path)

    result = validate_manifest(manifest)

    assert result["current_matrices"] == 190
    assert result["remaining_candidates"] == 2


def test_advance_removes_committed_candidate_and_reranks(tmp_path: Path) -> None:
    manifest, new, groups = _fixture(tmp_path)
    output = tmp_path / "remaining1.json"

    result = advance_manifest(manifest, new, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result["current_matrices"] == 191
    assert result["remaining_candidates"] == 1
    assert payload["candidates"][0]["rank"] == 1
    assert payload["candidates"][0]["group_id"] == groups[1]
    assert payload["expected"]["evidence_tier_counts"] == {"tier_2": 1}
    assert payload["advance_lineage"]["committed"][0]["group_id"] == groups[0]

    with pytest.raises(FileExistsError, match="overwrite immutable"):
        advance_manifest(manifest, new, output)


def test_advance_rejects_non_direct_frontier(tmp_path: Path) -> None:
    manifest, new, _ = _fixture(tmp_path)
    payload = torch.load(new, map_location="cpu", weights_only=True)
    payload["parent_checkpoint_sha256"] = "0" * 64
    torch.save(payload, new)

    with pytest.raises(ValueError, match="stale prior-frontier"):
        advance_manifest(manifest, new, tmp_path / "invalid.json")
