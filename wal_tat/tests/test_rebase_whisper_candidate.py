from pathlib import Path

import pytest
import torch

from rebase_whisper_candidate import rebase_checkpoint


def _entry(value: int, precision: str = "t3") -> dict:
    return {
        "precision": precision,
        "codes": torch.full((1, 1, 4), value, dtype=torch.int8),
        "scales": torch.full((1, 1), float(value + 2), dtype=torch.float16),
        "bias": None,
    }


def _save(path: Path, payload: dict) -> Path:
    torch.save(payload, path)
    return path


def test_rebase_keeps_new_parent_and_only_copies_candidate(tmp_path: Path) -> None:
    old_parent_path = _save(
        tmp_path / "old_parent.pt",
        {
            "model": "whisper-small",
            "group_size": 128,
            "accepted": True,
            "provisional": False,
            "groups": ["old"],
            "matrices": {"old": _entry(-1)},
            "history": [],
        },
    )
    source_path = _save(
        tmp_path / "source.pt",
        {
            "model": "whisper-small",
            "group_size": 128,
            "accepted": False,
            "provisional": True,
            "parent_checkpoint": str(old_parent_path),
            "candidate_groups": ["candidate.group"],
            "matrices": {
                "old": _entry(0),
                "candidate": _entry(1),
            },
        },
    )
    new_parent_path = _save(
        tmp_path / "new_parent.pt",
        {
            "model": "whisper-small",
            "group_size": 128,
            "accepted": True,
            "provisional": False,
            "groups": ["old", "new"],
            "matrices": {
                "old": _entry(-1),
                "new": _entry(0, "b1"),
            },
            "history": [{"kind": "accepted-new"}],
        },
    )

    rebased, summary = rebase_checkpoint(source_path, new_parent_path)

    assert set(rebased["matrices"]) == {"old", "new", "candidate"}
    assert torch.equal(rebased["matrices"]["old"]["codes"], _entry(-1)["codes"])
    assert torch.equal(rebased["matrices"]["new"]["codes"], _entry(0, "b1")["codes"])
    assert torch.equal(rebased["matrices"]["candidate"]["codes"], _entry(1)["codes"])
    assert rebased["parent_checkpoint"] == str(new_parent_path.resolve())
    assert rebased["accepted"] is False
    assert rebased["provisional"] is True
    assert rebased["candidate_groups"] == ["candidate.group"]
    assert summary["candidate_matrices"] == ["candidate"]
    assert rebased["history"][-1]["kind"] == "immutable-candidate-rebase"


def test_rebase_rejects_already_committed_candidate(tmp_path: Path) -> None:
    old_parent_path = _save(
        tmp_path / "old_parent.pt",
        {
            "model": "whisper-small",
            "group_size": 128,
            "accepted": True,
            "provisional": False,
            "matrices": {"old": _entry(-1)},
        },
    )
    source_path = _save(
        tmp_path / "source.pt",
        {
            "model": "whisper-small",
            "group_size": 128,
            "accepted": False,
            "provisional": True,
            "parent_checkpoint": str(old_parent_path),
            "candidate_groups": ["candidate.group"],
            "matrices": {
                "old": _entry(-1),
                "candidate": _entry(1),
            },
        },
    )
    new_parent_path = _save(
        tmp_path / "new_parent.pt",
        {
            "model": "whisper-small",
            "group_size": 128,
            "accepted": True,
            "provisional": False,
            "matrices": {
                "old": _entry(-1),
                "candidate": _entry(1),
            },
        },
    )

    with pytest.raises(ValueError, match="already committed"):
        rebase_checkpoint(source_path, new_parent_path)


def test_rebase_can_extract_strict_subset_from_macro_candidate(
    tmp_path: Path,
) -> None:
    old_parent_path = _save(
        tmp_path / "old_parent.pt",
        {
            "model": "whisper-small",
            "group_size": 128,
            "accepted": True,
            "provisional": False,
            "matrices": {"old": _entry(-1)},
        },
    )
    source_path = _save(
        tmp_path / "macro_source.pt",
        {
            "model": "whisper-small",
            "group_size": 128,
            "accepted": False,
            "provisional": True,
            "parent_checkpoint": str(old_parent_path),
            "candidate_groups": ["a.group", "b.group", "unused.group"],
            "matrices": {
                "old": _entry(-1),
                "a": _entry(0),
                "b": _entry(1),
                "unused": _entry(-1),
            },
        },
    )
    new_parent_path = _save(
        tmp_path / "new_parent.pt",
        {
            "model": "whisper-small",
            "group_size": 128,
            "accepted": True,
            "provisional": False,
            "matrices": {"old": _entry(-1), "new": _entry(1)},
        },
    )

    rebased, summary = rebase_checkpoint(
        source_path,
        new_parent_path,
        selected_candidate_matrices=["a", "b"],
        selected_candidate_groups=["a.group", "b.group"],
    )

    assert set(rebased["matrices"]) == {"old", "new", "a", "b"}
    assert rebased["candidate_groups"] == ["a.group", "b.group"]
    assert summary["candidate_matrices"] == ["a", "b"]
    assert rebased["rebase"]["source_candidate_matrices"] == [
        "a",
        "b",
        "unused",
    ]


def test_rebase_subset_requires_explicit_group_metadata(tmp_path: Path) -> None:
    old_parent_path = _save(
        tmp_path / "old_parent.pt",
        {
            "model": "whisper-small",
            "group_size": 128,
            "accepted": True,
            "provisional": False,
            "matrices": {"old": _entry(-1)},
        },
    )
    source_path = _save(
        tmp_path / "source.pt",
        {
            "model": "whisper-small",
            "group_size": 128,
            "accepted": False,
            "provisional": True,
            "parent_checkpoint": str(old_parent_path),
            "matrices": {"old": _entry(-1), "candidate": _entry(1)},
        },
    )
    new_parent_path = _save(
        tmp_path / "new_parent.pt",
        {
            "model": "whisper-small",
            "group_size": 128,
            "accepted": True,
            "provisional": False,
            "matrices": {"old": _entry(-1)},
        },
    )

    with pytest.raises(ValueError, match="require explicit candidate groups"):
        rebase_checkpoint(
            source_path,
            new_parent_path,
            selected_candidate_matrices=["candidate"],
        )
