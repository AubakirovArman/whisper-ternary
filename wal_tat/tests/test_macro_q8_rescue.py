import sys
from pathlib import Path

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from macro_q8_rescue import (  # noqa: E402
    compose_q8_rescue_groups,
    passes_cumulative_and_incremental_gates,
)


def test_three_way_gate_rejects_source_incremental_failure():
    assert not passes_cumulative_and_incremental_gates(
        {"c4": 1.01, "code": 1.019},
        {"c4": 1.001, "code": 1.0051},
        {"c4": 1.001, "code": 1.002},
        gate_ratio=1.02,
        incremental_gate_ratio=1.005,
    )


def test_three_way_gate_requires_every_policy_to_pass():
    passing = {"c4": 1.001, "code": 1.002}
    assert passes_cumulative_and_incremental_gates(
        {"c4": 0.99, "code": 1.019},
        passing,
        passing,
        gate_ratio=1.02,
        incremental_gate_ratio=1.005,
    )
    assert not passes_cumulative_and_incremental_gates(
        {"c4": 0.99, "code": 1.0201},
        passing,
        passing,
        gate_ratio=1.02,
        incremental_gate_ratio=1.005,
    )
    assert not passes_cumulative_and_incremental_gates(
        {"c4": 0.99, "code": 1.019},
        passing,
        {"c4": 1.001, "code": 1.0051},
        gate_ratio=1.02,
        incremental_gate_ratio=1.005,
    )


def test_q8_rescue_preserves_existing_q8_groups():
    q2 = torch.tensor([[[2.0], [2.0], [2.0]]])
    q4 = torch.tensor([[[4.0], [4.0], [4.0]]])
    existing_q8 = torch.tensor([[[8.0], [8.0], [8.0]]])
    rescue_q8 = torch.tensor([[[9.0], [9.0], [9.0]]])
    q4_mask = torch.tensor([[True, False, True]])
    existing_q8_mask = torch.tensor([[False, True, False]])
    rescue_q8_mask = torch.tensor([[False, False, True]])

    grouped = compose_q8_rescue_groups(
        q2,
        q4,
        q4_mask,
        existing_q8,
        existing_q8_mask,
        rescue_q8,
        rescue_q8_mask,
    )

    assert torch.equal(grouped, torch.tensor([[[4.0], [8.0], [9.0]]]))


def test_q8_rescue_rejects_overlap_with_existing_q8():
    value = torch.zeros(1, 2, 1)
    with pytest.raises(ValueError, match="overlaps existing Q8"):
        compose_q8_rescue_groups(
            value,
            value,
            torch.tensor([[True, False]]),
            value,
            torch.tensor([[False, True]]),
            value,
            torch.tensor([[False, True]]),
        )
