import sys
from pathlib import Path

import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from build_matched_validation_recovery_suite import (  # noqa: E402
    interleave,
    resolved_calibration_strides,
    resolved_domain_offsets,
    windows,
)


def test_windows_are_contiguous_and_include_next_token():
    ids = torch.arange(30)
    result = windows(ids, count=2, length=4, offset=3)
    assert [value.tolist() for value in result] == [
        [3, 4, 5, 6, 7],
        [7, 8, 9, 10, 11],
    ]


def test_interleave_preserves_weighted_domain_order():
    values = {
        "c4": [torch.tensor([0]), torch.tensor([1])],
        "squad": [
            torch.tensor([10]),
            torch.tensor([11]),
            torch.tensor([12]),
            torch.tensor([13]),
        ],
    }
    result = interleave(values, {"c4": 1, "squad": 2}, base_count=2)
    assert [int(value.item()) for value in result] == [0, 10, 12, 1, 11, 13]


def test_per_domain_offsets_override_legacy_offsets():
    class Args:
        calibration_offset = 100
        gate_offset = 200
        c4_calibration_offset = 11
        c4_gate_offset = None
        squad_calibration_offset = None
        squad_gate_offset = 22
        code_calibration_offset = 33
        code_gate_offset = 44

    calibration, gates = resolved_domain_offsets(
        Args(), {"c4": "c4_train", "squad": "squad", "code": "torch_code"}
    )
    assert calibration == {"c4_train": 11, "squad": 100, "torch_code": 33}
    assert gates == {"c4_train": 200, "squad": 22, "torch_code": 44}


def test_per_domain_calibration_strides_are_resolved():
    class Args:
        c4_calibration_stride = 1000
        squad_calibration_stride = None
        code_calibration_stride = 3000

    assert resolved_calibration_strides(
        Args(), {"c4": "c4", "squad": "squad", "code": "code"}
    ) == {"c4": 1000, "squad": None, "code": 3000}
