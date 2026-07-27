import sys
from pathlib import Path

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from reverse_q4_to_q2_fraction import (  # noqa: E402
    build_candidate,
    global_lowest_masks,
    parse_fractions,
)


def test_global_lowest_masks_selects_across_matrices():
    damage = {"a": torch.tensor([[4.0, 1.0]]), "b": torch.tensor([[3.0, 2.0]])}
    eligible = {
        "a": torch.tensor([[True, False]]),
        "b": torch.tensor([[True, True]]),
    }
    masks = global_lowest_masks(damage, eligible, 2)
    assert torch.equal(masks["a"], torch.tensor([[False, False]]))
    assert torch.equal(masks["b"], torch.tensor([[True, True]]))


def test_build_candidate_moves_only_selected_q4_groups_to_q2():
    parent = {
        "matrices": {
            "m": {
                "q2_codes_int8": torch.zeros((1, 2, 2), dtype=torch.int8),
                "q2_scales_fp16": torch.ones((1, 2), dtype=torch.float16),
                "q4_codes_int8": torch.tensor([[[2, -2], [3, -3]]]),
                "q4_scales_fp16": torch.ones((1, 2), dtype=torch.float16),
                "q4_mask": torch.tensor([[True, True]]),
                "q8_mask": torch.tensor([[False, False]]),
            }
        }
    }
    selected = {"m": torch.tensor([[True, False]])}
    codes = {"m": torch.tensor([[[1, -1], [1, -1]]], dtype=torch.int8)}
    scales = {"m": torch.tensor([[0.5, 0.75]])}
    candidate = build_candidate(parent, selected, codes, scales)
    entry = candidate["matrices"]["m"]
    assert torch.equal(entry["q4_mask"], torch.tensor([[False, True]]))
    assert torch.equal(entry["q2_codes_int8"][0, 0], torch.tensor([1, -1]))
    assert entry["q2_scales_fp16"][0, 0].item() == 0.5
    assert parent["matrices"]["m"]["q4_mask"].all()


def test_fraction_parser_validates_range():
    assert parse_fractions("0.2,0.1,0.2") == (0.1, 0.2)
    with pytest.raises(ValueError, match="fractions"):
        parse_fractions("0,0.5")
