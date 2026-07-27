import pytest
import torch

from wal_tat import InputSecondMomentCollector


def test_input_second_moment_collector_is_forward_only_and_exact():
    linear = torch.nn.Linear(3, 2)
    values = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]])
    collector = InputSecondMomentCollector(linear)
    try:
        linear(values)
        assert torch.equal(collector.moment(), torch.tensor([5.0, 4.0, 5.0]))
    finally:
        collector.close()


def test_input_second_moment_requires_samples():
    collector = InputSecondMomentCollector(torch.nn.Linear(2, 2))
    try:
        with pytest.raises(RuntimeError, match="no forward"):
            collector.moment()
    finally:
        collector.close()
