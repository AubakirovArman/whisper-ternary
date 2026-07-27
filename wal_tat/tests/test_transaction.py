import pytest
import torch

from wal_tat import (
    AtomicTernaryTransaction,
    TransactionalTernaryLinear,
    TransactionalTernaryMatrix,
)


def make_matrix():
    weight = torch.tensor(
        [[0.10, -0.70, 1.20, -1.80], [0.30, 0.90, -0.20, -1.10]],
        dtype=torch.float32,
    )
    return weight, TransactionalTernaryMatrix(weight, group_size=2)


def test_unselected_groups_are_exact_and_have_no_gradient():
    original, matrix = make_matrix()
    mask = torch.tensor([[True, False], [False, False]])
    matrix.begin(mask)
    matrix.set_candidate_state(1.0, 0.0)
    effective = matrix.effective_weight()
    assert torch.equal(effective[:, 2:], original[:, 2:])
    assert torch.equal(effective[1], original[1])
    effective.sum().backward()
    gradient = matrix.master_weight.grad
    assert torch.count_nonzero(gradient[0, :2]) > 0
    assert torch.count_nonzero(gradient[0, 2:]) == 0
    assert torch.count_nonzero(gradient[1]) == 0


def test_commit_freezes_strict_ternary_codes():
    _, matrix = make_matrix()
    mask = torch.tensor([[True, False], [False, True]])
    matrix.begin(mask)
    matrix.set_candidate_state(1.0, 0.0)
    result = matrix.commit()
    assert result["groups"] == 2
    assert torch.equal(matrix.committed_mask, mask)
    assert set(matrix.committed_codes[mask].unique().tolist()) <= {-1, 0, 1}
    before = matrix.effective_weight().clone()
    with torch.no_grad():
        matrix.master_weight.add_(10)
    after = matrix.effective_weight()
    grouped_before = before.view(2, 2, 2)
    grouped_after = after.view(2, 2, 2)
    assert torch.equal(grouped_before[mask], grouped_after[mask])


def test_rollback_restores_candidate_weights_and_scales_exactly():
    original, matrix = make_matrix()
    scale = matrix.group_scale.detach().clone()
    mask = torch.tensor([[False, True], [True, False]])
    matrix.begin(mask)
    with torch.no_grad():
        grouped = matrix.master_weight.view(2, 2, 2)
        grouped[mask] += 7
        matrix.group_scale[mask] *= 3
    matrix.rollback()
    assert torch.equal(matrix.master_weight, original)
    assert torch.equal(matrix.group_scale, scale)
    assert not matrix.in_transaction


def test_overlap_and_empty_transactions_are_rejected():
    _, matrix = make_matrix()
    first = torch.tensor([[True, False], [False, False]])
    matrix.begin(first)
    matrix.commit()
    with pytest.raises(ValueError, match="overlaps"):
        matrix.begin(first)
    with pytest.raises(ValueError, match="at least one"):
        matrix.begin(torch.zeros_like(first))


def test_external_codes_are_used_exactly():
    _, matrix = make_matrix()
    mask = torch.tensor([[True, False], [False, False]])
    matrix.begin(mask)
    codes = torch.zeros_like(matrix.candidate_codes)
    codes[0, 0] = torch.tensor([-1, 1])
    scales = matrix.group_scale.detach().clone()
    scales[0, 0] = 2.0
    matrix.set_candidate_codes(codes, scales)
    matrix.set_candidate_state(1.0, 0.0)
    assert torch.equal(matrix.effective_weight()[0, :2], torch.tensor([-2.0, 2.0]))


def test_mixed_bpw_tracks_committed_fraction():
    _, matrix = make_matrix()
    assert matrix.projected_mixed_bpw() == 16.0
    mask = torch.tensor([[True, False], [True, False]])
    matrix.begin(mask)
    matrix.commit()
    assert matrix.projected_mixed_bpw() == pytest.approx(13.0)


def test_linear_wrapper_preserves_shape():
    _, matrix = make_matrix()
    linear = TransactionalTernaryLinear(matrix)
    assert linear(torch.ones(3, 4)).shape == (3, 2)


def test_continuous_compensation_is_exact_masked_and_rollback_safe():
    original, matrix = make_matrix()
    mask = torch.tensor([[False, True], [False, False]])
    before = matrix.effective_weight().clone()
    assert matrix.begin_continuous_compensation(mask) == 1
    assert torch.equal(matrix.effective_weight(), before)

    matrix.effective_weight().sum().backward()
    gradient = matrix.master_weight.grad.view(2, 2, 2)
    assert torch.count_nonzero(gradient[mask]) == 2
    assert torch.count_nonzero(gradient[~mask]) == 0
    with torch.no_grad():
        grouped = matrix.master_weight.view(2, 2, 2)
        grouped[mask] += 0.25
    assert not torch.equal(matrix.effective_weight(), before)
    matrix.rollback_continuous_compensation()
    assert torch.equal(matrix.master_weight, original)
    assert torch.equal(matrix.effective_weight(), before)


def test_continuous_compensation_commit_keeps_bf16_weights_without_coverage():
    original, matrix = make_matrix()
    mask = torch.tensor([[True, False], [False, False]])
    matrix.begin_continuous_compensation(mask)
    with torch.no_grad():
        matrix.master_weight[0, :2] += 0.5
    result = matrix.commit_continuous_compensation()
    assert result == {"groups": 1}
    assert not matrix.in_continuous_compensation
    assert not matrix.committed_mask.any()
    assert torch.equal(matrix.master_weight[0, :2], original[0, :2] + 0.5)


def test_continuous_compensation_rejects_committed_groups():
    _, matrix = make_matrix()
    mask = torch.tensor([[True, False], [False, False]])
    matrix.begin(mask)
    matrix.commit()
    with pytest.raises(ValueError, match="committed"):
        matrix.begin_continuous_compensation(mask)


def test_reopen_starts_exact_and_rollback_restores_committed_state():
    _, matrix = make_matrix()
    mask = torch.tensor([[True, False], [False, False]])
    matrix.begin(mask)
    matrix.commit()
    deployed = matrix.effective_weight().clone()
    original_master = matrix.master_weight.detach().clone()
    original_scale = matrix.group_scale.detach().clone()
    original_codes = matrix.committed_codes.clone()

    matrix.begin(mask, allow_reopen=True)
    assert torch.equal(matrix.effective_weight(), deployed)
    assert not matrix.committed_mask[0, 0]
    with torch.no_grad():
        matrix.master_weight[0, :2] += 4
        matrix.group_scale[0, 0] *= 2
    matrix.rollback()

    assert torch.equal(matrix.master_weight, original_master)
    assert torch.equal(matrix.group_scale, original_scale)
    assert torch.equal(matrix.committed_codes, original_codes)
    assert matrix.committed_mask[0, 0]
    assert torch.equal(matrix.effective_weight(), deployed)


def test_reopen_commit_replaces_codes_atomically():
    _, matrix = make_matrix()
    mask = torch.tensor([[True, False], [False, False]])
    matrix.begin(mask)
    matrix.commit()
    before = matrix.committed_codes[0, 0].clone()
    matrix.begin(mask, allow_reopen=True)
    codes = matrix.candidate_codes.clone()
    codes[0, 0] = -before
    scales = matrix.group_scale.detach().clone()
    matrix.set_candidate_codes(codes, scales)
    matrix.set_candidate_state(1.0, 0.0)
    matrix.commit()
    assert torch.equal(matrix.committed_codes[0, 0], -before)


def test_atomic_transaction_rolls_back_all_matrices():
    weight_a, matrix_a = make_matrix()
    weight_b, matrix_b = make_matrix()
    atomic = AtomicTernaryTransaction({"up": matrix_a, "down": matrix_b})
    up_mask = torch.tensor([[True, False], [False, False]])
    down_mask = torch.tensor([[False, True], [False, False]])
    atomic.begin({"up": up_mask, "down": down_mask}, transaction_id="pair-1")
    with torch.no_grad():
        matrix_a.master_weight[0, :2] += 3
        matrix_b.master_weight[0, 2:] -= 5
    result = atomic.rollback()
    assert set(result) == {"up", "down"}
    assert torch.equal(matrix_a.master_weight, weight_a)
    assert torch.equal(matrix_b.master_weight, weight_b)
    assert not atomic.in_transaction


def test_atomic_begin_failure_rolls_back_already_opened_matrix():
    _, matrix_a = make_matrix()
    _, matrix_b = make_matrix()
    committed = torch.tensor([[True, False], [False, False]])
    matrix_b.begin(committed)
    matrix_b.commit()
    atomic = AtomicTernaryTransaction({"up": matrix_a, "down": matrix_b})
    with pytest.raises(ValueError, match="overlaps"):
        atomic.begin({"up": committed, "down": committed})
    assert not matrix_a.in_transaction
    assert matrix_b.committed_mask[0, 0]
