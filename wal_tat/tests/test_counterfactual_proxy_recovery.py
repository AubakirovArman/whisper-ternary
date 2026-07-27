from argparse import Namespace
from pathlib import Path
import sys

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from counterfactual_proxy_recovery import (  # noqa: E402
    change_statistics,
    deployed_representation,
    validate_disjoint_slices,
)
from wal_tat import ProxyTernaryMatrix  # noqa: E402


def test_validate_disjoint_slices_rejects_overlap() -> None:
    args = Namespace(
        selection_gate_start=0,
        selection_gate_sequences=4,
        confirmation_gate_start=3,
        confirmation_gate_sequences=4,
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_slices(args)


def test_deployed_representation_is_strict_and_fp16_rounded() -> None:
    source_codes = torch.tensor([[[-1, 0, 1, 0]]], dtype=torch.int8)
    source_scales = torch.tensor([[1.0]])
    proxy = ProxyTernaryMatrix(
        source_codes,
        source_scales,
        compute_dtype=torch.float32,
        fake_fp16_scale=True,
    )
    with torch.no_grad():
        proxy.proxy_code[0, 0, 1] = 0.6
        proxy.group_scale.fill_(1.001)

    codes, scales, statistics = deployed_representation(
        proxy, source_codes, source_scales
    )

    assert codes.dtype == torch.int8
    assert set(codes.flatten().tolist()) <= {-1, 0, 1}
    assert scales.dtype == torch.float16
    assert scales.item() == torch.tensor(1.001).half().item()
    assert statistics["changed_code_values"] == 1
    assert statistics["changed_scale_groups"] == 1
    assert statistics["code_churn"] == 0.25


def test_change_statistics_can_measure_total_churn_from_frontier() -> None:
    frontier = torch.tensor([[[-1, 0, 1, 0]]], dtype=torch.int8)
    starting = torch.tensor([[[0, 0, 1, 0]]], dtype=torch.int8)
    candidate = torch.tensor([[[0, 1, 1, 0]]], dtype=torch.int8)
    mask = torch.tensor([[True]])

    incremental = change_statistics(
        starting,
        torch.ones((1, 1)),
        candidate,
        torch.ones((1, 1)),
        mask,
    )
    total = change_statistics(
        frontier,
        torch.ones((1, 1)),
        candidate,
        torch.ones((1, 1)),
        mask,
    )

    assert incremental["changed_code_values"] == 1
    assert total["changed_code_values"] == 2
