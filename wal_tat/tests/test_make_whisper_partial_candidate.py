from pathlib import Path

import torch

from make_whisper_partial_candidate import build_partial_checkpoint


def test_partial_candidate_selects_lowest_error_group(
    tmp_path: Path, monkeypatch
) -> None:
    parent_path = tmp_path / "parent.pt"
    candidate_path = tmp_path / "candidate.pt"
    parent_path.write_bytes(b"parent")
    candidate_path.write_bytes(b"candidate")
    monkeypatch.setattr(
        "make_whisper_partial_candidate.sha256_file",
        lambda path: Path(path).name,
    )
    parent = {
        "accepted": True,
        "provisional": False,
        "model": "whisper",
        "group_size": 2,
        "matrices": {},
        "history": [],
    }
    candidate = {
        "accepted": False,
        "provisional": True,
        "model": "whisper",
        "group_size": 2,
        "parent_checkpoint": str(parent_path),
        "matrices": {
            "target": {
                "precision": "t3",
                "codes": torch.tensor([[[1, -1], [1, 1]]], dtype=torch.int8),
                "scales": torch.ones(1, 2, dtype=torch.float16),
                "bias": None,
            }
        },
    }
    source = {"target": torch.tensor([[1.0, -1.0, 9.0, 9.0]])}
    result = build_partial_checkpoint(
        parent,
        candidate,
        source,
        fraction=0.5,
        parent_path=parent_path,
        candidate_path=candidate_path,
    )
    entry = result["matrices"]["target"]
    assert torch.equal(entry["committed_mask"], torch.tensor([[True, False]]))
    assert torch.equal(entry["base_weight"], source["target"].bfloat16())
    assert result["partial_t3"]["committed_groups"] == 1


def test_partial_candidate_can_rank_by_activation_weighted_error(
    tmp_path: Path, monkeypatch
) -> None:
    parent_path = tmp_path / "parent.pt"
    candidate_path = tmp_path / "candidate.pt"
    parent_path.write_bytes(b"parent")
    candidate_path.write_bytes(b"candidate")
    monkeypatch.setattr(
        "make_whisper_partial_candidate.sha256_file",
        lambda path: Path(path).name,
    )
    parent = {
        "accepted": True,
        "provisional": False,
        "model": "whisper",
        "group_size": 2,
        "matrices": {},
        "history": [],
    }
    candidate = {
        "accepted": False,
        "provisional": True,
        "model": "whisper",
        "group_size": 2,
        "parent_checkpoint": str(parent_path),
        "matrices": {
            "target": {
                "precision": "t3",
                "codes": torch.zeros((1, 2, 2), dtype=torch.int8),
                "scales": torch.ones((1, 2), dtype=torch.float16),
                "bias": None,
            }
        },
    }
    source = {"target": torch.tensor([[2.0, 2.0, 1.0, 1.0]])}
    result = build_partial_checkpoint(
        parent,
        candidate,
        source,
        input_moments={"target": torch.tensor([0.01, 0.01, 10.0, 10.0])},
        fraction=0.5,
        parent_path=parent_path,
        candidate_path=candidate_path,
    )

    assert torch.equal(
        result["matrices"]["target"]["committed_mask"],
        torch.tensor([[True, False]]),
    )
    assert "activation-weighted" in result["partial_t3"]["kind"]
