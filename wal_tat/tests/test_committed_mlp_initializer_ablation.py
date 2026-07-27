import pytest

from committed_mlp_initializer_ablation import (
    INITIALIZERS,
    aggregate_change_statistics,
    initializer_grid,
    parse_projection_list,
)


def test_projection_list_is_strict_and_ordered():
    assert parse_projection_list("gate_proj, down_proj") == (
        "gate_proj",
        "down_proj",
    )
    with pytest.raises(ValueError, match="at least one"):
        parse_projection_list(" , ")
    with pytest.raises(ValueError, match="unique"):
        parse_projection_list("gate_proj,gate_proj")
    with pytest.raises(ValueError, match="unknown"):
        parse_projection_list("gate_proj,q_proj")


def test_initializer_grid_is_exhaustive():
    grid = initializer_grid(("gate", "down"))
    assert len(grid) == len(INITIALIZERS) ** 2
    assert {tuple(item.values()) for item in grid} == {
        (left, right) for left in INITIALIZERS for right in INITIALIZERS
    }


def test_aggregate_change_statistics_uses_true_weighted_churn():
    result = aggregate_change_statistics(
        {
            "large": {
                "active_code_values": 90,
                "changed_code_values": 9,
                "active_scale_groups": 9,
                "changed_scale_groups": 3,
            },
            "small": {
                "active_code_values": 10,
                "changed_code_values": 5,
                "active_scale_groups": 1,
                "changed_scale_groups": 1,
            },
        }
    )
    assert result == {
        "active_code_values": 100,
        "changed_code_values": 14,
        "code_churn": 0.14,
        "active_scale_groups": 10,
        "changed_scale_groups": 4,
    }
