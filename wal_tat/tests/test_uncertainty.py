import pytest

from wal_tat import paired_block_bootstrap_nll


def test_paired_bootstrap_identical_losses_have_exact_unit_ratio():
    result = paired_block_bootstrap_nll(
        [10.0, 11.0, 9.0, 12.0],
        [10.0, 11.0, 9.0, 12.0],
        [5, 5, 5, 5],
        gate_ratio=1.001,
        samples=128,
        block_size=2,
    )
    assert result["observed_ratio"] == pytest.approx(1.0)
    assert result["ratio_ci"] == pytest.approx(
        {"lower": 1.0, "median": 1.0, "upper": 1.0}
    )
    assert result["observed_delta_nll"] == pytest.approx(0.0)
    assert result["paired_window_win_rate"] == pytest.approx(1.0)
    assert result["confidence_passed"] is True


def test_paired_bootstrap_rejects_uniform_excess_over_gate():
    baseline = [10.0, 20.0, 30.0, 40.0]
    candidate = [value * 1.01 for value in baseline]
    result = paired_block_bootstrap_nll(
        baseline,
        candidate,
        [10, 10, 10, 10],
        gate_ratio=1.005,
        samples=128,
        block_size=2,
    )
    assert result["observed_ratio"] == pytest.approx(1.01)
    assert result["ratio_ci"]["upper"] == pytest.approx(1.01)
    assert result["point_passed"] is False
    assert result["confidence_passed"] is False


def test_paired_bootstrap_is_seed_reproducible():
    arguments = dict(
        baseline_nll_sums=[10.0, 11.0, 8.0, 15.0, 9.0, 12.0],
        candidate_nll_sums=[9.9, 11.2, 8.1, 14.8, 9.2, 11.9],
        token_counts=[4, 4, 4, 4, 4, 4],
        gate_ratio=1.02,
        samples=256,
        block_size=2,
        seed=17,
    )
    assert paired_block_bootstrap_nll(**arguments) == paired_block_bootstrap_nll(
        **arguments
    )


@pytest.mark.parametrize(
    "override",
    [
        {"candidate_nll_sums": [1.0]},
        {"token_counts": [1]},
        {"block_size": 0},
        {"samples": 1},
        {"confidence": 1.0},
    ],
)
def test_paired_bootstrap_validates_inputs(override):
    arguments = {
        "baseline_nll_sums": [1.0, 2.0],
        "candidate_nll_sums": [1.0, 2.0],
        "token_counts": [1, 1],
        "gate_ratio": 1.01,
        "samples": 16,
        "block_size": 1,
        "confidence": 0.95,
    }
    arguments.update(override)
    with pytest.raises(ValueError):
        paired_block_bootstrap_nll(**arguments)
