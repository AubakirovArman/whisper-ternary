import pytest
import torch

from wal_tat import (
    apply_top_groupwise_ternary_moves,
    apply_top_groupwise_ternary_moves_across_matrices,
    consensus_ternary_move_scores,
    ternary_newton_proposals,
)


def test_newton_proposals_choose_adjacent_direction_and_include_curvature():
    codes = torch.tensor([[[-1, 0, 1, 0]]], dtype=torch.int8)
    scales = torch.tensor([[2.0]])
    gradient = torch.tensor([[1.0, 2.0, -3.0, -4.0]])
    curvature = torch.tensor([[0.5, 1.0, 2.0, 0.25]])
    proposed, delta, score = ternary_newton_proposals(
        codes,
        scales,
        gradient,
        curvature,
        in_features=4,
        curvature_multiplier=1.0,
    )
    assert proposed.tolist() == [[[0, -1, 0, 1]]]
    assert delta.tolist() == [[[2.0, -2.0, -2.0, 2.0]]]
    # score = -(g*delta + 0.5*h*delta^2)
    assert score.tolist() == [[[-3.0, 2.0, -10.0, 7.5]]]


def test_newton_proposals_reject_invalid_curvature():
    with pytest.raises(ValueError, match="non-negative"):
        ternary_newton_proposals(
            torch.zeros((1, 1, 2), dtype=torch.int8),
            torch.ones((1, 1)),
            torch.zeros((1, 2)),
            torch.tensor([[0.0, -1.0]]),
            in_features=2,
        )


def test_groupwise_selection_changes_one_code_per_selected_group():
    source = torch.zeros((1, 3, 4), dtype=torch.int8)
    proposed = torch.ones_like(source)
    scores = torch.tensor(
        [[[4.0, 1.0, 2.0, 3.0], [9.0, 8.0, 1.0, 1.0], [5.0, 6.0, 1.0, 2.0]]]
    )
    candidate, groups, positions, chosen = apply_top_groupwise_ternary_moves(
        source, proposed, scores, budget=2
    )
    assert groups.tolist() == [1, 2]
    assert positions.tolist() == [0, 1]
    assert chosen.tolist() == [9.0, 6.0]
    assert int((candidate != source).sum().item()) == 2


def test_groupwise_selection_spends_one_global_budget_across_matrices():
    source = {
        "fc1": torch.zeros((1, 2, 2), dtype=torch.int8),
        "fc2": torch.zeros((1, 2, 2), dtype=torch.int8),
    }
    proposed = {name: torch.ones_like(value) for name, value in source.items()}
    scores = {
        "fc1": torch.tensor([[[9.0, 1.0], [3.0, 2.0]]]),
        "fc2": torch.tensor([[[8.0, 7.0], [10.0, 1.0]]]),
    }
    candidates, names, groups, positions, chosen = (
        apply_top_groupwise_ternary_moves_across_matrices(
            source, proposed, scores, budget=3
        )
    )
    assert names == ["fc2", "fc1", "fc2"]
    assert groups.tolist() == [1, 0, 0]
    assert positions.tolist() == [0, 0, 0]
    assert chosen.tolist() == [10.0, 9.0, 8.0]
    assert int((candidates["fc1"] != source["fc1"]).sum()) == 1
    assert int((candidates["fc2"] != source["fc2"]).sum()) == 2


def test_consensus_score_uses_worst_replica():
    delta = torch.tensor([[[2.0, -1.0]]])
    gradients = torch.tensor([[[1.0, 2.0]], [[-0.5, 3.0]]])
    curvature = torch.zeros_like(gradients)
    score = consensus_ternary_move_scores(
        delta,
        gradients,
        curvature,
        in_features=2,
        curvature_multiplier=0.0,
    )
    # First move helps replica 2 but hurts replica 1, so worst score is -2.
    # Second move helps both; its weaker predicted improvement is 2.
    assert score.tolist() == [[[-2.0, 2.0]]]
