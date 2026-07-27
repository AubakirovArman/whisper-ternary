import sys
from pathlib import Path

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from reverse_q8_to_q4_fraction import build_candidate  # noqa: E402


def parent_artifact() -> dict:
    return {
        "matrices": {
            "m": {
                "q2_codes_int8": torch.zeros((1, 3, 2), dtype=torch.int8),
                "q2_scales_fp16": torch.ones((1, 3), dtype=torch.float16),
                "q4_codes_int8": torch.tensor([[[2, -2], [3, -3], [4, -4]]]),
                "q4_scales_fp16": torch.ones((1, 3), dtype=torch.float16),
                "q4_mask": torch.tensor([[True, False, False]]),
                "q8_codes_int8": torch.tensor([[[20, -20], [30, -30], [40, -40]]]),
                "q8_scales_fp16": torch.full((1, 3), 0.25, dtype=torch.float16),
                "q8_mask": torch.tensor([[False, True, True]]),
            }
        }
    }


def test_build_candidate_moves_only_selected_q8_groups_to_q4():
    parent = parent_artifact()
    selected = {"m": torch.tensor([[False, True, False]])}
    codes = {"m": torch.tensor([[[1, -1], [6, -6], [7, -7]]], dtype=torch.int8)}
    scales = {"m": torch.tensor([[0.5, 0.75, 1.0]])}
    candidate = build_candidate(parent, selected, codes, scales)
    entry = candidate["matrices"]["m"]
    assert torch.equal(entry["q4_mask"], torch.tensor([[True, True, False]]))
    assert torch.equal(entry["q8_mask"], torch.tensor([[False, False, True]]))
    assert torch.equal(entry["q4_codes_int8"][0, 1], torch.tensor([6, -6]))
    assert entry["q4_scales_fp16"][0, 1].item() == 0.75
    assert torch.equal(entry["q4_codes_int8"][0, 0], torch.tensor([2, -2]))
    assert torch.equal(entry["q8_codes_int8"], parent["matrices"]["m"]["q8_codes_int8"])
    assert torch.equal(parent["matrices"]["m"]["q4_mask"], torch.tensor([[True, False, False]]))
    assert torch.equal(parent["matrices"]["m"]["q8_mask"], torch.tensor([[False, True, True]]))


def test_build_candidate_rejects_non_q8_selection():
    parent = parent_artifact()
    selected = {"m": torch.tensor([[True, False, False]])}
    codes = {"m": torch.zeros((1, 3, 2), dtype=torch.int8)}
    scales = {"m": torch.ones((1, 3))}
    with pytest.raises(ValueError, match="Q8 eligibility"):
        build_candidate(parent, selected, codes, scales)
