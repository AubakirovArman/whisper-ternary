from pathlib import Path

import torch

from scale_whisper_candidate import scale_candidate


def test_scale_candidate_changes_only_new_matrix_scales(
    tmp_path: Path, monkeypatch
) -> None:
    parent_path = tmp_path / "parent.pt"
    candidate_path = tmp_path / "candidate.pt"
    parent = {
        "accepted": True,
        "provisional": False,
        "model": "whisper",
        "group_size": 128,
        "matrices": {
            "old": {
                "precision": "t3",
                "codes": torch.zeros(2, dtype=torch.int8),
                "scales": torch.tensor([2.0], dtype=torch.float16),
            }
        },
    }
    candidate = {
        **parent,
        "accepted": False,
        "provisional": True,
        "matrices": {
            **parent["matrices"],
            "new": {
                "precision": "t3",
                "codes": torch.ones(2, dtype=torch.int8),
                "scales": torch.tensor([4.0], dtype=torch.float16),
            },
        },
        "history": [],
    }
    parent_path.write_bytes(b"parent")
    candidate_path.write_bytes(b"candidate")
    monkeypatch.setattr(
        "scale_whisper_candidate.sha256_file", lambda path: Path(path).name
    )

    result = scale_candidate(
        candidate,
        parent,
        factor=0.5,
        candidate_path=candidate_path,
        parent_path=parent_path,
    )

    assert torch.equal(
        result["matrices"]["old"]["scales"], torch.tensor([2.0], dtype=torch.float16)
    )
    assert torch.equal(
        result["matrices"]["new"]["scales"], torch.tensor([2.0], dtype=torch.float16)
    )
    assert torch.equal(result["matrices"]["new"]["codes"], candidate["matrices"]["new"]["codes"])
    assert result["candidate_scale_multiplier"]["new_matrices"] == ["new"]
