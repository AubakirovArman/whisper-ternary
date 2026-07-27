from pathlib import Path
import sys

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from counterfactual_scale_recovery import slice_domain_gates  # noqa: E402


def sample_gates(count: int = 8) -> dict[str, list[torch.Tensor]]:
    return {
        "wiki": [torch.tensor([index]) for index in range(count)],
        "code": [torch.tensor([index + 100]) for index in range(count)],
    }


def test_slice_domain_gates_returns_exact_requested_window() -> None:
    selected = slice_domain_gates(
        sample_gates(), start=2, count=3, role="selection"
    )

    assert [int(item.item()) for item in selected["wiki"]] == [2, 3, 4]
    assert [int(item.item()) for item in selected["code"]] == [102, 103, 104]


@pytest.mark.parametrize(
    ("start", "count"),
    [(-1, 2), (0, 0), (7, 2)],
)
def test_slice_domain_gates_rejects_invalid_windows(start: int, count: int) -> None:
    with pytest.raises(ValueError):
        slice_domain_gates(
            sample_gates(), start=start, count=count, role="confirmation"
        )
