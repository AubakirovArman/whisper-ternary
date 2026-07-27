import sys
from pathlib import Path

import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from boundary_sparse_recode import (  # noqa: E402
    adjacent_move_proposals,
    apply_global_top_group_moves,
    apply_top_group_moves,
    refit_selected_scales,
)


def test_adjacent_moves_follow_negative_gradient_and_ignore_uncommitted():
    codes = torch.tensor([[[-1, 0, 1, 0], [1, 0, -1, 1]]], dtype=torch.int8)
    scales = torch.tensor([[2.0, 3.0]])
    gradient = torch.tensor([[1.0, 2.0, -3.0, -4.0, 9.0, 9.0, 9.0, 9.0]])
    committed = torch.tensor([[True, False]])
    proposed, delta, scores = adjacent_move_proposals(
        codes, scales, gradient, committed, in_features=8
    )
    assert torch.equal(proposed[0, 0], torch.tensor([0, -1, 0, 1]))
    assert torch.equal(delta[0, 0], torch.tensor([2.0, -2.0, -2.0, 2.0]))
    assert torch.isneginf(scores[0, 1]).all()
    assert scores[0, 0].tolist() == [-2.0, 4.0, -6.0, 8.0]


def test_top_group_moves_changes_at_most_one_code_per_group():
    source = torch.zeros((1, 3, 4), dtype=torch.int8)
    proposed = torch.ones_like(source)
    scores = torch.tensor([[[4.0, 1.0, 2.0, 3.0], [9.0, 1.0, 1.0, 1.0], [5.0, 6.0, 1.0, 2.0]]])
    candidate, groups, positions, chosen = apply_top_group_moves(
        source, proposed, scores, budget=2
    )
    assert groups.tolist() == [1, 2]
    assert positions.tolist() == [0, 1]
    assert chosen.tolist() == [9.0, 6.0]
    assert int((candidate != source).sum()) == 2


def test_global_top_group_moves_ranks_across_matrices():
    source = {
        "a": torch.zeros((1, 2, 2), dtype=torch.int8),
        "b": torch.zeros((1, 2, 2), dtype=torch.int8),
    }
    proposed = {name: torch.ones_like(value) for name, value in source.items()}
    scores = {
        "a": torch.tensor([[[7.0, 1.0], [2.0, 1.0]]]),
        "b": torch.tensor([[[9.0, 1.0], [6.0, 1.0]]]),
    }
    candidates, groups, positions, chosen = apply_global_top_group_moves(
        source, proposed, scores, budget=3
    )
    assert chosen.tolist() == [9.0, 7.0, 6.0]
    assert groups["a"].tolist() == [0]
    assert groups["b"].tolist() == [0, 1]
    assert positions["a"].tolist() == [0]
    assert int((candidates["a"] != source["a"]).sum()) == 1
    assert int((candidates["b"] != source["b"]).sum()) == 2


def test_refit_selected_scales_only_changes_selected_groups_and_rounds_fp16():
    target = torch.tensor([[[2.0, 0.0, -2.0, 2.0], [8.0, -8.0, 0.0, 8.0]]])
    codes = torch.tensor([[[1, 0, -1, 1], [1, -1, 0, 1]]], dtype=torch.int8)
    source_scales = torch.ones((1, 2))
    result = refit_selected_scales(
        target, codes, source_scales, torch.tensor([1])
    )
    assert result[0, 0].item() == 1.0
    assert result[0, 1].item() == 8.0
