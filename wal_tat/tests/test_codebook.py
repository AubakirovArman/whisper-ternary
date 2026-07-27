import pytest
import torch

from wal_tat.codebook import (
    FixedColumnCodebookLinear,
    column_codebook_physical_bpw,
    column_codebook_payload_bits,
    column_outlier_density,
    mixed_column_codebook_payload_bits,
    mixed_column_codebook_project,
    reconstruct_column_codebook,
    weighted_column_codebook_project,
)


def test_column_q2_project_recovers_four_level_columns_exactly():
    weight = torch.tensor(
        [
            [-3.0, 10.0],
            [-1.0, 20.0],
            [1.0, 30.0],
            [3.0, 40.0],
        ]
    )
    codes, centroids, error = weighted_column_codebook_project(
        weight, torch.ones(2), bits=2, iterations=4, chunk_columns=1
    )
    reconstructed = reconstruct_column_codebook(codes, centroids)
    assert torch.equal(reconstructed, weight)
    assert torch.equal(error, torch.zeros(2))
    assert set(codes[:, 0].tolist()) == {0, 1, 2, 3}


def test_column_codebook_counts_dense_metadata_in_physical_bpw():
    assert column_codebook_physical_bpw(
        out_features=768, code_bits=2
    ) == pytest.approx(2 + 64 / 768)
    assert column_codebook_payload_bits(
        out_features=768, in_features=3072, code_bits=2
    ) == 768 * 3072 * 2 + 3072 * 4 * 16


def test_weighted_column_project_respects_output_importance():
    weight = torch.tensor([[-4.0], [-1.0], [1.0], [4.0], [100.0]])
    uniform_codes, uniform_centroids, _ = weighted_column_codebook_project(
        weight, torch.ones(1), bits=2, iterations=8
    )
    weighted_codes, weighted_centroids, _ = weighted_column_codebook_project(
        weight,
        torch.ones(1),
        bits=2,
        output_importance=torch.tensor([1.0, 1.0, 1.0, 1.0, 100.0]),
        iterations=8,
    )
    uniform = reconstruct_column_codebook(uniform_codes, uniform_centroids)
    weighted = reconstruct_column_codebook(weighted_codes, weighted_centroids)
    assert abs(weighted[-1, 0] - 100) <= abs(uniform[-1, 0] - 100)


def test_outlier_density_is_column_local():
    weight = torch.zeros((20, 2))
    weight[:, 0] = 1
    weight[0, 1] = 100
    density = column_outlier_density(weight, threshold_multiplier=5)
    assert density[0] == 0
    assert density[1] == pytest.approx(0.05)


def test_mixed_projection_assigns_outlier_column_to_q4_and_counts_mask():
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn((32, 4), generator=generator)
    weight[0, 2] = 100
    projection = mixed_column_codebook_project(
        weight,
        torch.ones(4),
        q4_fraction=0.25,
        outlier_threshold_multiplier=5,
        iterations=4,
        chunk_columns=2,
    )
    assert projection.q4_mask.tolist() == [False, False, True, False]
    expected_bits = mixed_column_codebook_payload_bits(
        projection.q4_mask, out_features=32
    )
    assert projection.payload_bits == expected_bits
    assert projection.physical_bpw == pytest.approx(expected_bits / weight.numel())
    assert projection.effective_weight().shape == weight.shape
    assert projection.selection == "outlier"


def test_error_gain_selection_promotes_column_with_largest_measured_gain():
    generator = torch.Generator().manual_seed(19)
    weight = torch.randn((32, 4), generator=generator)
    weight[:, 1] = torch.linspace(-20, 20, 32)
    projection = mixed_column_codebook_project(
        weight,
        torch.ones(4),
        q4_fraction=0.25,
        selection="error_gain",
        iterations=8,
        chunk_columns=2,
    )
    q2_codes, q2_centroids, q2_error = weighted_column_codebook_project(
        weight, torch.ones(4), bits=2, iterations=8, chunk_columns=2
    )
    del q2_codes, q2_centroids
    q4_codes, q4_centroids, q4_error = weighted_column_codebook_project(
        weight, torch.ones(4), bits=4, iterations=8, chunk_columns=2
    )
    del q4_codes, q4_centroids
    expected = int(torch.argmax(q2_error - q4_error))
    assert projection.q4_mask.nonzero().flatten().tolist() == [expected]
    assert projection.selection == "error_gain"


def test_mixed_projection_rejects_unknown_selection():
    with pytest.raises(ValueError, match="selection"):
        mixed_column_codebook_project(
            torch.randn(16, 2),
            torch.ones(2),
            selection="unknown",
        )


def test_fixed_column_codebook_linear_matches_materialized_linear():
    weight = torch.tensor(
        [
            [-3.0, 10.0],
            [-1.0, 20.0],
            [1.0, 30.0],
            [3.0, 40.0],
        ]
    )
    codes, centroids, _ = weighted_column_codebook_project(
        weight, torch.ones(2), bits=2, iterations=4
    )
    module = FixedColumnCodebookLinear(codes, centroids, bias=torch.arange(4.0))
    value = torch.tensor([[2.0, -1.0]])
    expected = torch.nn.functional.linear(value, weight, torch.arange(4.0))
    assert torch.equal(module(value), expected)
