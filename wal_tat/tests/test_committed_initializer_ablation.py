from pathlib import Path
import sys

import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from committed_initializer_ablation import (  # noqa: E402
    balanced_calibration,
    calibration_domain_labels,
    representation_statistics,
)


def sample_suite(interleaved: bool) -> dict:
    calibration = [torch.tensor([index]) for index in range(12)]
    return {
        "source_splits": {"c4": "c4", "squad": "squad", "code": "code"},
        "calibration_per_domain": 2,
        "c4_calibration_repeat": 1,
        "squad_calibration_repeat": 4,
        "code_calibration_repeat": 1,
        "interleave_calibration": interleaved,
        "calibration": calibration,
    }


def test_interleaved_calibration_labels_match_builder_order() -> None:
    labels = calibration_domain_labels(sample_suite(True))
    assert labels == [
        "c4",
        "squad",
        "squad",
        "squad",
        "squad",
        "code",
        "c4",
        "squad",
        "squad",
        "squad",
        "squad",
        "code",
    ]
    selected = balanced_calibration(sample_suite(True), count=2, offset=0)
    assert [int(item.item()) for item in selected] == [0, 6, 1, 2, 5, 11]


def test_representation_statistics_reports_improvement_and_churn() -> None:
    original = torch.tensor([[1.0, 0.0, -1.0, 0.0]])
    mask = torch.tensor([[True]])
    source_codes = torch.zeros((1, 1, 4), dtype=torch.int8)
    source_scales = torch.ones((1, 1))
    candidate_codes = torch.tensor([[[1, 0, -1, 0]]], dtype=torch.int8)
    candidate_scales = torch.ones((1, 1))

    result = representation_statistics(
        original,
        mask,
        source_codes,
        source_scales,
        candidate_codes,
        candidate_scales,
        torch.ones(4),
        4,
    )

    assert result["changed_code_values"] == 2
    assert result["code_churn"] == 0.5
    assert result["candidate_relative_weight_mse"] == 0.0
    assert result["weight_mse_reduction_fraction"] == 1.0
