import sys
from pathlib import Path

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from commit_sparse_ternary_recode_artifact import sparse_change_statistics  # noqa: E402


def test_sparse_change_statistics_counts_only_committed_changes():
    source_codes = torch.tensor([[[-1, 0, 1, 0], [1, 0, -1, 1]]], dtype=torch.int8)
    candidate_codes = source_codes.clone()
    candidate_codes[0, 0, 1] = 1
    source_scales = torch.ones((1, 2), dtype=torch.float16)
    candidate_scales = source_scales.clone()
    candidate_scales[0, 0] = 1.25
    mask = torch.tensor([[True, False]])
    result = sparse_change_statistics(
        source_codes, source_scales, candidate_codes, candidate_scales, mask
    )
    assert result["changed_code_values"] == 1
    assert result["changed_scale_groups"] == 1
    assert result["code_churn"] == 0.25


def test_sparse_change_statistics_rejects_uncommitted_code_change():
    source_codes = torch.zeros((1, 2, 4), dtype=torch.int8)
    candidate_codes = source_codes.clone()
    candidate_codes[0, 1, 0] = 1
    scales = torch.ones((1, 2), dtype=torch.float16)
    mask = torch.tensor([[True, False]])
    with pytest.raises(ValueError, match="outside committed"):
        sparse_change_statistics(
            source_codes, scales, candidate_codes, scales, mask
        )
