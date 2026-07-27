import sys
from pathlib import Path

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from macro_mlp_q4_rescue import global_rescue_masks, parse_fractions  # noqa: E402


def test_parse_rescue_fractions_is_sorted_and_unique():
    assert parse_fractions("0.2,0.01,0.2,1") == (0.01, 0.2, 1.0)


def test_parse_rescue_fractions_can_explicitly_include_strict_q2():
    assert parse_fractions("0.1,0,1", allow_zero=True) == (0.0, 0.1, 1.0)


def test_parse_rescue_fractions_rejects_zero_by_default():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        parse_fractions("0,0.1")


def test_global_rescue_masks_selects_largest_eligible_benefits():
    benefit = {
        "a": torch.tensor([[1.0, 9.0, 2.0]]),
        "b": torch.tensor([[8.0, 7.0]]),
    }
    eligible = {
        "a": torch.tensor([[True, False, True]]),
        "b": torch.tensor([[True, True]]),
    }
    masks = global_rescue_masks(benefit, eligible, count=2)
    assert torch.equal(masks["a"], torch.tensor([[False, False, False]]))
    assert torch.equal(masks["b"], torch.tensor([[True, True]]))
