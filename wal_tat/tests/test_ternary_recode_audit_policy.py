from pathlib import Path
import sys


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import pytest
import torch

from ternary_recode_artifact_audit import (  # noqa: E402
    acceptance_decision,
    artifact_code_churn,
    artifact_matrix_entries,
    validated_compensated_master,
)
from wal_tat import TransactionalTernaryMatrix


def test_point_only_mode_still_requires_incremental_confidence() -> None:
    common = {
        "cumulative_point_passed": True,
        "cumulative_confidence_passed": False,
        "incremental_point_passed": True,
        "source_restoration_exact": True,
        "cumulative_point_only": True,
    }
    assert not acceptance_decision(
        **common, incremental_confidence_passed=False
    )
    assert acceptance_decision(**common, incremental_confidence_passed=True)


def test_default_mode_requires_cumulative_confidence() -> None:
    assert not acceptance_decision(
        cumulative_point_passed=True,
        cumulative_confidence_passed=False,
        incremental_point_passed=True,
        incremental_confidence_passed=True,
        source_restoration_exact=True,
        cumulative_point_only=False,
    )


def test_artifact_entries_normalize_legacy_and_bundle() -> None:
    tensor = torch.tensor([1])
    legacy = {
        "format": "wal-tat-ternary-recode-v1",
        "target_name": "a",
        "committed_mask": tensor,
        "ternary_codes_int8": tensor,
        "scales_fp16": tensor,
    }
    assert tuple(artifact_matrix_entries(legacy)) == ("a",)

    bundle = {
        "format": "wal-tat-ternary-recode-bundle-v1",
        "target_names": ["a", "b"],
        "matrices": {"a": {}, "b": {}},
    }
    assert tuple(artifact_matrix_entries(bundle)) == ("a", "b")


def test_bundle_targets_must_match_entries() -> None:
    with pytest.raises(ValueError, match="target_names"):
        artifact_matrix_entries(
            {
                "format": "wal-tat-ternary-recode-bundle-v1",
                "target_names": ["a"],
                "matrices": {"b": {}},
            }
        )


def test_fallback_compensation_entries_and_master_are_validated() -> None:
    matrix = TransactionalTernaryMatrix(
        torch.arange(8).float().view(2, 4), group_size=2
    )
    mask = torch.tensor([[True, False], [False, True]])
    matrix.begin(mask, transaction_id="audit-test")
    matrix.set_candidate_state(1.0, 0.0)
    matrix.commit()
    candidate = matrix.master_weight.detach().bfloat16().clone()
    grouped = candidate.view(2, 2, 2)
    grouped[~mask] += 1
    entry = {"fp_master_bf16": candidate}
    artifact = {
        "format": "wal-tat-ternary-fallback-compensation-v1",
        "target_names": ["a"],
        "matrices": {"a": entry},
    }
    assert artifact_matrix_entries(artifact) == {"a": entry}
    validated = validated_compensated_master(entry, matrix, "cpu")
    assert validated.dtype == torch.float32

    bad = candidate.clone()
    bad.view(2, 2, 2)[mask] += 1
    with pytest.raises(ValueError, match="committed"):
        validated_compensated_master({"fp_master_bf16": bad}, matrix, "cpu")


def test_optional_churn_metadata_supports_all_artifact_families() -> None:
    assert artifact_code_churn({"representation": {"code_churn": 0.1}}) == 0.1
    assert artifact_code_churn(
        {"total_change_from_checkpoint": {"code_churn": 0.2}}
    ) == 0.2
    assert artifact_code_churn({"format": "fallback"}) is None
