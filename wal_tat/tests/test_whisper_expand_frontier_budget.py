import torch

from whisper_expand_frontier import _matrix_budget


def test_matrix_budget_counts_only_committed_partial_groups() -> None:
    matrices = {
        "full": {
            "precision": "b1",
            "codes": torch.ones((1, 2, 4), dtype=torch.int8),
            "scales": torch.ones((1, 2), dtype=torch.float16),
        },
        "partial": {
            "precision": "t3",
            "codes": torch.zeros((1, 2, 4), dtype=torch.int8),
            "scales": torch.ones((1, 2), dtype=torch.float16),
            "committed_mask": torch.tensor([[True, False]]),
        },
    }

    budget = _matrix_budget(matrices, total_weights=24)

    assert budget["converted_weights"] == 12
    assert budget["weights_by_precision"] == {
        "b1": 8,
        "t3": 4,
        "nz4": 0,
        "q4": 0,
    }
    assert budget["fully_converted_matrices"] == 1
    assert budget["partially_converted_matrices"] == 1
    assert budget["projected_main_linear_bpw"] == (8 + 32 + 8 + 16 + 12 * 16) / 24
