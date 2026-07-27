import sys
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from block_proxy_recovery import calibration_indices_for_domain  # noqa: E402


def test_calibration_domain_indices_follow_interleaved_repeats():
    suite = {
        "calibration": list(range(16)),
        "gates": {"c4": [], "squad": [], "code": []},
        "calibration_per_domain": 2,
        "c4_calibration_repeat": 2,
        "squad_calibration_repeat": 4,
        "code_calibration_repeat": 2,
        "interleave_calibration": True,
    }
    assert calibration_indices_for_domain(suite, "c4") == (0, 1, 8, 9)
    assert calibration_indices_for_domain(suite, "squad") == (
        2,
        3,
        4,
        5,
        10,
        11,
        12,
        13,
    )
    assert calibration_indices_for_domain(suite, "code") == (6, 7, 14, 15)
    assert calibration_indices_for_domain(suite, "all") == tuple(range(16))


def test_calibration_domain_indices_follow_contiguous_layout():
    suite = {
        "calibration": list(range(16)),
        "gates": {"c4": [], "squad": [], "code": []},
        "calibration_per_domain": 2,
        "c4_calibration_repeat": 2,
        "squad_calibration_repeat": 4,
        "code_calibration_repeat": 2,
        "interleave_calibration": False,
    }
    assert calibration_indices_for_domain(suite, "c4") == (0, 1, 2, 3)
    assert calibration_indices_for_domain(suite, "squad") == tuple(range(4, 12))
    assert calibration_indices_for_domain(suite, "code") == (12, 13, 14, 15)
