import math
from itertools import product

import pytest
import torch

from wal_tat.scoring import exact_diagonal_ternary_project

from wal_tat import (
    activation_fisher_group_damage,
    diagonal_ternary_search,
    hard_codes_scales,
    q2_g128_physical_bpw,
    q4_g128_physical_bpw,
    q8_g128_physical_bpw,
    select_group_mask,
    sensitivity_decile_mask,
    transaction_schedule,
    weighted_symmetric_odd_level_project,
    weighted_symmetric_nz4_project,
    weighted_symmetric_q4_project,
    weighted_symmetric_q8_project,
)


def test_hard_quantizer_is_strictly_ternary():
    weight = torch.tensor([[0.1, -0.8, 1.7, -2.2]])
    codes, scales = hard_codes_scales(weight, group_size=2)
    assert set(codes.unique().tolist()) <= {-1, 0, 1}
    assert scales.shape == (1, 2)


def test_schedule_finishes_hard():
    assert transaction_schedule(0, 100) == pytest.approx((0.0, 0.3508855606815209))
    assert transaction_schedule(100, 100) == (1.0, 0.0)


def test_prism_compatible_layout_is_2_125_physical_bpw():
    assert q2_g128_physical_bpw() == 2.125
    assert math.log2(3) == pytest.approx(1.5849625)


def test_q4_g128_layout_is_4_125_physical_bpw():
    assert q4_g128_physical_bpw() == 4.125


def test_q8_g128_layout_is_8_125_physical_bpw():
    assert q8_g128_physical_bpw() == 8.125


def test_weighted_q4_project_uses_signed_codes_and_reconstructs_grid_values():
    weight = torch.tensor([[-7.0, -3.0, 0.0, 2.0, 7.0]])
    codes, scales, error = weighted_symmetric_q4_project(
        weight, torch.ones(5), group_size=5
    )
    assert set(codes.unique().tolist()) <= set(range(-8, 8))
    reconstructed = codes.float() * scales.unsqueeze(-1)
    assert torch.allclose(reconstructed.reshape_as(weight), weight, atol=1e-5)
    assert error.item() == pytest.approx(0.0, abs=1e-7)


def test_weighted_q8_project_uses_full_signed_int8_range():
    weight = torch.tensor([[-127.0, -64.0, 0.0, 63.0, 127.0]])
    codes, scales, error = weighted_symmetric_q8_project(
        weight, torch.ones(5), group_size=5
    )
    assert codes.min().item() >= -128
    assert codes.max().item() <= 127
    reconstructed = codes.float() * scales.unsqueeze(-1)
    assert torch.allclose(reconstructed.reshape_as(weight), weight, atol=1e-4)
    assert error.item() == pytest.approx(0.0, abs=1e-6)


def test_weighted_nz4_project_uses_exactly_four_no_zero_symbols():
    weight = torch.tensor([[-3.0, -1.0, 1.0, 3.0]])
    codes, scales, error = weighted_symmetric_nz4_project(
        weight, torch.ones(4), group_size=4
    )
    assert set(codes.unique().tolist()) == {-3, -1, 1, 3}
    assert scales.item() == pytest.approx(1.0, abs=1e-5)
    assert error.item() == pytest.approx(0.0, abs=1e-7)


def test_weighted_nz4_project_rejects_bad_moment_shape():
    with pytest.raises(ValueError, match="input features"):
        weighted_symmetric_nz4_project(
            torch.ones(1, 4), torch.ones(3), group_size=4
        )


def test_weighted_odd_level_projection_supports_progressive_collapse():
    weight = torch.tensor([[4.0, 2.0, 0.2, -3.0]])
    moment = torch.ones(4)
    previous_error = None
    for levels in (7, 5, 3):
        codes, scales, error = weighted_symmetric_odd_level_project(
            weight, moment, levels=levels, group_size=4
        )
        radius = levels // 2
        assert int(codes.min()) >= -radius
        assert int(codes.max()) <= radius
        assert scales.shape == error.shape == (1, 1)
        if previous_error is not None:
            assert float(error) >= previous_error - 1e-6
        previous_error = float(error)


def test_weighted_odd_level_projection_rejects_even_codebooks():
    with pytest.raises(ValueError, match="odd integer"):
        weighted_symmetric_odd_level_project(
            torch.ones(1, 4), torch.ones(4), levels=4, group_size=4
        )


def test_fisher_score_uses_causal_moments():
    weight = torch.tensor([[0.1, 0.3, 0.1, 0.3]])
    output = torch.ones(1)
    uniform = activation_fisher_group_damage(
        weight, torch.ones(4), output, group_size=2, relative=False
    )
    shifted = activation_fisher_group_damage(
        weight, torch.tensor([100.0, 100.0, 1.0, 1.0]), output, group_size=2, relative=False
    )
    assert uniform[0, 0] == pytest.approx(uniform[0, 1])
    assert shifted[0, 0] > shifted[0, 1]


def test_select_mask_respects_eligibility():
    scores = torch.tensor([[4.0, 1.0], [3.0, 2.0]])
    eligible = torch.tensor([[True, False], [True, True]])
    mask = select_group_mask(scores, 2, eligible=eligible, strategy="lowest")
    assert torch.equal(mask, torch.tensor([[False, False], [True, True]]))


def test_diagonal_search_never_worse_than_its_tested_zero_threshold():
    weight = torch.tensor([[0.1, -0.4, 1.2, -2.0]])
    moment = torch.tensor([1.0, 5.0, 2.0, 0.5])
    codes, scales, error = diagonal_ternary_search(weight, moment, group_size=4)
    assert set(codes.unique().tolist()) <= {-1, 0, 1}
    baseline_scale = (moment * weight.abs()).sum() / moment.sum()
    baseline_error = (moment * (weight - weight.sign() * baseline_scale).square()).sum()
    assert error.item() <= baseline_error.item() + 1e-6


def test_exact_diagonal_projector_matches_brute_force_optimum():
    weight = torch.tensor([[0.12, -0.51, 1.13, -2.07]])
    moment = torch.tensor([4.0, 0.25, 2.0, 1.5])
    codes, scales, error = exact_diagonal_ternary_project(
        weight, moment, group_size=4
    )
    brute_error = float("inf")
    for values in product((-1.0, 0.0, 1.0), repeat=4):
        candidate = torch.tensor(values)
        denominator = (moment * candidate.square()).sum()
        if denominator == 0:
            scale = torch.tensor(1e-5)
        else:
            scale = ((moment * candidate * weight[0]).sum() / denominator).clamp_min(0)
        candidate_error = (
            moment * (weight[0] - scale * candidate).square()
        ).sum().item()
        brute_error = min(brute_error, candidate_error)
    assert set(codes.unique().tolist()) <= {-1, 0, 1}
    assert scales.item() > 0
    assert error.item() == pytest.approx(brute_error, abs=1e-6)


def test_exact_diagonal_projector_is_no_worse_than_threshold_grid():
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn((3, 17), generator=generator)
    moment = torch.rand(17, generator=generator).add_(0.01)
    _, _, grid_error = diagonal_ternary_search(weight, moment, group_size=8)
    _, _, exact_error = exact_diagonal_ternary_project(
        weight, moment, group_size=8
    )
    assert torch.all(exact_error <= grid_error + 1e-5)


def test_sensitivity_decile_mask_covers_requested_rank_bucket():
    scores = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    eligible = torch.ones_like(scores, dtype=torch.bool)
    mask = sensitivity_decile_mask(scores, eligible, decile=9, count=5)
    selected = scores[mask]
    assert mask.sum().item() == 5
    assert selected.min().item() >= 80
    assert selected.max().item() <= 89
    assert selected.tolist() == [80.0, 82.0, 84.0, 87.0, 89.0]


def test_sensitivity_decile_mask_respects_sparse_eligibility():
    scores = torch.arange(120, dtype=torch.float32).reshape(12, 10)
    eligible = scores.remainder(2).eq(0)
    mask = sensitivity_decile_mask(scores, eligible, decile=10, count=3)
    assert torch.all(eligible[mask])
    assert scores[mask].tolist() == [108.0, 112.0, 118.0]
