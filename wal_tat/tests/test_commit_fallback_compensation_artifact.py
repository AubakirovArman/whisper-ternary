import pytest
import torch

from commit_fallback_compensation_artifact import (
    aggregate_relative_delta_to_source,
    matrix_change_statistics,
)


def test_matrix_changes_are_limited_to_declared_regions():
    mask = torch.tensor([[True, False]])
    source_codes = torch.tensor([[[1, -1], [0, 0]]], dtype=torch.int8)
    artifact_codes = source_codes.clone()
    artifact_codes[0, 0, 0] = 0
    source_scales = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    artifact_scales = source_scales.clone()
    artifact_scales[0, 0] = 1.5
    source_master = torch.tensor([[1.0, -1.0, 2.0, 3.0]], dtype=torch.bfloat16)
    artifact_master = source_master.clone()
    artifact_master[0, 2:] += 1
    result = matrix_change_statistics(
        shape=(1, 4),
        group_size=2,
        source_mask=mask,
        source_codes=source_codes,
        source_scales=source_scales,
        source_master=source_master,
        artifact_codes=artifact_codes,
        artifact_scales=artifact_scales,
        artifact_master=artifact_master,
    )
    assert result["changed_code_values"] == 1
    assert result["changed_scale_groups"] == 1
    assert result["changed_fallback_weights"] == 2
    assert result["committed_master_values_unchanged"]


def test_committed_master_delta_is_rejected():
    mask = torch.tensor([[True, False]])
    codes = torch.zeros((1, 2, 2), dtype=torch.int8)
    scales = torch.ones((1, 2), dtype=torch.float16)
    source = torch.ones((1, 4), dtype=torch.bfloat16)
    candidate = source.clone()
    candidate[0, 0] += 1
    with pytest.raises(ValueError, match="committed master"):
        matrix_change_statistics(
            shape=(1, 4),
            group_size=2,
            source_mask=mask,
            source_codes=codes,
            source_scales=scales,
            source_master=source,
            artifact_codes=codes,
            artifact_scales=scales,
            artifact_master=candidate,
        )


def test_aggregate_relative_delta_is_weighted_by_fallback_weights():
    statistics = {
        "small": {"fallback_relative_delta": 0.25, "fallback_weights": 1},
        "large": {"fallback_relative_delta": 0.5, "fallback_weights": 3},
    }
    assert aggregate_relative_delta_to_source(statistics) == pytest.approx(0.4375)
