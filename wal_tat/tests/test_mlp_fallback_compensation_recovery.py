import pytest
import torch

from mlp_fallback_compensation_recovery import (
    aggregate_relative_delta,
    apply_strict_recode_artifact,
    fallback_change_statistics,
    strict_mlp_artifact_from_checkpoint,
    validate_disjoint_slices,
)
from wal_tat import TransactionalTernaryMatrix


def committed_matrix() -> TransactionalTernaryMatrix:
    matrix = TransactionalTernaryMatrix(torch.arange(8).float().view(2, 4), group_size=2)
    mask = torch.tensor([[True, False], [False, True]])
    matrix.begin(mask, transaction_id="test")
    matrix.set_candidate_state(1.0, 0.0)
    matrix.commit()
    return matrix


def test_apply_bundle_preserves_masks_and_updates_strict_codes():
    matrix = committed_matrix()
    codes = matrix.committed_codes.clone()
    codes[matrix.committed_mask] = 0
    scales = matrix.group_scale.detach().half().clone()
    artifact = {
        "format": "wal-tat-ternary-recode-bundle-v1",
        "source_checkpoint_sha256": "abc",
        "target_names": ["mlp.up_proj"],
        "matrices": {
            "mlp.up_proj": {
                "committed_mask": matrix.committed_mask.clone(),
                "ternary_codes_int8": codes,
                "scales_fp16": scales,
            }
        },
    }
    names = apply_strict_recode_artifact(
        artifact, {"mlp.up_proj": matrix}, "abc", "cpu"
    )
    assert names == ("mlp.up_proj",)
    assert torch.equal(matrix.committed_codes, codes)
    assert set(matrix.committed_codes.unique().tolist()) <= {-1, 0, 1}


def test_apply_bundle_restores_only_uncommitted_compensated_master_values():
    matrix = committed_matrix()
    source = matrix.master_weight.detach().bfloat16().clone()
    compensated = source.clone()
    grouped = compensated.view(2, 2, 2)
    grouped[~matrix.committed_mask.cpu()] += 0.5
    artifact = {
        "format": "wal-tat-ternary-fallback-compensation-v1",
        "source_checkpoint_sha256": "abc",
        "target_names": ["mlp.up_proj"],
        "matrices": {
            "mlp.up_proj": {
                "committed_mask": matrix.committed_mask.clone(),
                "ternary_codes_int8": matrix.committed_codes.clone(),
                "scales_fp16": matrix.group_scale.detach().half().clone(),
                "fp_master_bf16": compensated,
            }
        },
    }
    apply_strict_recode_artifact(artifact, {"mlp.up_proj": matrix}, "abc", "cpu")
    assert torch.equal(matrix.master_weight, compensated.float())


def test_apply_bundle_rejects_compensated_changes_to_committed_groups():
    matrix = committed_matrix()
    compensated = matrix.master_weight.detach().bfloat16().clone()
    grouped = compensated.view(2, 2, 2)
    grouped[matrix.committed_mask.cpu()] += 0.5
    artifact = {
        "format": "wal-tat-ternary-fallback-compensation-v1",
        "source_checkpoint_sha256": "abc",
        "target_names": ["mlp.up_proj"],
        "matrices": {
            "mlp.up_proj": {
                "committed_mask": matrix.committed_mask.clone(),
                "ternary_codes_int8": matrix.committed_codes.clone(),
                "scales_fp16": matrix.group_scale.detach().half().clone(),
                "fp_master_bf16": compensated,
            }
        },
    }
    with pytest.raises(ValueError, match="changes committed groups"):
        apply_strict_recode_artifact(
            artifact, {"mlp.up_proj": matrix}, "abc", "cpu"
        )


def test_fallback_statistics_ignore_committed_master_values():
    matrix = committed_matrix()
    source = matrix.master_weight.detach().bfloat16().cpu()
    candidate = source.clone()
    grouped = candidate.view(2, 2, 2)
    grouped[~matrix.committed_mask.cpu()] += 1
    statistics = fallback_change_statistics(matrix, source, candidate)
    assert statistics["fallback_weights"] == 4
    assert statistics["changed_fallback_weights"] == 4
    assert statistics["committed_master_values_unchanged"]
    assert aggregate_relative_delta({"one": statistics}) == statistics[
        "relative_delta_to_source_absmean"
    ]


def test_disjoint_slice_validation():
    class Args:
        selection_gate_start = 0
        selection_gate_sequences = 8
        confirmation_gate_start = 8
        confirmation_gate_sequences = 8

    validate_disjoint_slices(Args())
    Args.confirmation_gate_start = 7
    try:
        validate_disjoint_slices(Args())
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("overlapping slices must fail")


def test_strict_mlp_artifact_is_derived_exactly_from_checkpoint():
    matrices = {}
    expected_names = []
    for projection in ("up_proj", "gate_proj", "down_proj"):
        name = f"model.layers.3.mlp.{projection}"
        expected_names.append(name)
        codes = torch.tensor([[1, 0, -1, 1, 0]], dtype=torch.int8)
        matrices[name] = {
            "shape": (1, 5),
            "group_size": 2,
            "committed_mask": torch.tensor([[True, False, True]]),
            "ternary_codes_int8": codes,
            "scales_fp16": torch.tensor([[0.5, 0.25, 0.125]], dtype=torch.float16),
        }
    payload = {"format": "wal-tat-multi-g128-v2", "matrices": matrices}
    artifact = strict_mlp_artifact_from_checkpoint(payload, "checkpoint-hash", 3)

    assert artifact["source_checkpoint_sha256"] == "checkpoint-hash"
    assert artifact["target_names"] == expected_names
    assert artifact["derived_from_source_checkpoint"] is True
    for name in expected_names:
        entry = artifact["matrices"][name]
        assert entry["ternary_codes_int8"].shape == (1, 3, 2)
        assert entry["ternary_codes_int8"].dtype == torch.int8
        assert entry["ternary_codes_int8"][0, -1].tolist() == [0, 0]
        assert torch.equal(
            entry["committed_mask"], matrices[name]["committed_mask"]
        )
