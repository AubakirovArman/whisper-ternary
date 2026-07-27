from pathlib import Path

import pytest
import torch

from interpolate_whisper_checkpoints import interpolate_checkpoints


def _checkpoint(*, accepted: bool, provisional: bool, codes=None, scales=None):
    return {
        "model": "model",
        "group_size": 2,
        "accepted": accepted,
        "provisional": provisional,
        "matrices": {
            "matrix": {
                "precision": "t3",
                "codes": torch.tensor([[-1, 0, 1]]) if codes is None else codes,
                "scales": torch.tensor([1.0, 2.0], dtype=torch.float16)
                if scales is None
                else scales,
            }
        },
        "history": [],
    }


def test_interpolates_only_scales_and_preserves_codes(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save({}, parent_path)
    torch.save({}, candidate_path)
    parent = _checkpoint(accepted=True, provisional=False)
    candidate = _checkpoint(
        accepted=False,
        provisional=True,
        scales=torch.tensor([3.0, 4.0], dtype=torch.float16),
    )
    result = interpolate_checkpoints(
        parent,
        candidate,
        alpha=0.25,
        parent_path=parent_path,
        candidate_path=candidate_path,
    )
    assert result["matrices"]["matrix"]["scales"].tolist() == [1.5, 2.5]
    assert torch.equal(
        result["matrices"]["matrix"]["codes"],
        parent["matrices"]["matrix"]["codes"],
    )
    assert result["accepted"] is False
    assert result["provisional"] is True


def test_rejects_code_changes_and_invalid_alpha(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save({}, parent_path)
    torch.save({}, candidate_path)
    parent = _checkpoint(accepted=True, provisional=False)
    changed = _checkpoint(
        accepted=False,
        provisional=True,
        codes=torch.tensor([[1, 0, 1]]),
    )
    with pytest.raises(ValueError, match="hard codes differ"):
        interpolate_checkpoints(
            parent,
            changed,
            alpha=0.5,
            parent_path=parent_path,
            candidate_path=candidate_path,
        )
    with pytest.raises(ValueError, match="strictly between"):
        interpolate_checkpoints(
            parent,
            _checkpoint(accepted=False, provisional=True),
            alpha=1.0,
            parent_path=parent_path,
            candidate_path=candidate_path,
        )
