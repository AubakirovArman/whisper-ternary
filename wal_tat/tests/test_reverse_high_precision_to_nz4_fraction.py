import sys
from pathlib import Path

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from reverse_high_precision_to_nz4_fraction import (  # noqa: E402
    build_candidate,
    snapshot_projection_weights,
)


def parent_artifact() -> dict:
    return {
        "matrices": {
            "m": {
                "q2_codes_int8": torch.zeros((1, 3, 2), dtype=torch.int8),
                "q2_scales_fp16": torch.ones((1, 3), dtype=torch.float16),
                "q4_mask": torch.tensor([[True, False, False]]),
                "q8_mask": torch.tensor([[False, True, True]]),
                "q8_codes_int8": torch.tensor(
                    [[[20, -20], [30, -30], [40, -40]]], dtype=torch.int8
                ),
                "q8_scales_fp16": torch.full((1, 3), 0.25, dtype=torch.float16),
            }
        }
    }


def test_build_candidate_moves_selected_q8_group_to_nz4():
    parent = parent_artifact()
    selected = {"m": torch.tensor([[False, True, False]])}
    codes = {
        "m": torch.tensor([[[-1, 1], [-3, 3], [-1, 3]]], dtype=torch.int8)
    }
    scales = {"m": torch.tensor([[0.5, 0.75, 1.0]])}

    candidate = build_candidate(
        parent, selected, codes, scales, source_precision="q8"
    )
    entry = candidate["matrices"]["m"]

    assert torch.equal(entry["q8_mask"], torch.tensor([[False, False, True]]))
    assert torch.equal(entry["q4_mask"], torch.tensor([[True, False, False]]))
    assert torch.equal(entry["nz4_mask"], torch.tensor([[False, True, False]]))
    assert torch.equal(entry["q2_codes_int8"][0, 1], torch.tensor([-3, 3]))
    assert entry["q2_scales_fp16"][0, 1].item() == 0.75
    assert torch.equal(parent["matrices"]["m"]["q8_mask"], torch.tensor([[False, True, True]]))
    assert "nz4_mask" not in parent["matrices"]["m"]


def test_build_candidate_moves_selected_q4_group_to_nz4():
    parent = parent_artifact()
    selected = {"m": torch.tensor([[True, False, False]])}
    codes = {
        "m": torch.tensor([[[-3, 1], [-1, 3], [-1, 1]]], dtype=torch.int8)
    }
    scales = {"m": torch.tensor([[0.25, 0.5, 1.0]])}

    candidate = build_candidate(
        parent, selected, codes, scales, source_precision="q4"
    )
    entry = candidate["matrices"]["m"]

    assert torch.equal(entry["q4_mask"], torch.tensor([[False, False, False]]))
    assert torch.equal(entry["q8_mask"], torch.tensor([[False, True, True]]))
    assert torch.equal(entry["nz4_mask"], torch.tensor([[True, False, False]]))
    assert torch.equal(entry["q2_codes_int8"][0, 0], torch.tensor([-3, 1]))


def test_build_candidate_rejects_noneligible_selection():
    parent = parent_artifact()
    selected = {"m": torch.tensor([[True, False, False]])}
    codes = {"m": torch.ones((1, 3, 2), dtype=torch.int8)}
    scales = {"m": torch.ones((1, 3))}

    with pytest.raises(ValueError, match="Q8 eligibility"):
        build_candidate(
            parent, selected, codes, scales, source_precision="q8"
        )


def test_snapshot_projection_weights_is_independent_of_later_mutation():
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(4, 2, bias=False)
    expected = model.proj.weight.detach().clone()

    snapshot = snapshot_projection_weights(model, ("proj",))
    with torch.no_grad():
        model.proj.weight.zero_()

    assert snapshot["proj"].device.type == "cpu"
    torch.testing.assert_close(snapshot["proj"], expected)
