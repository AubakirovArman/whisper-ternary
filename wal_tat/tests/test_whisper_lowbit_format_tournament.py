import sys
from pathlib import Path

import pytest


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from whisper_lowbit_format_tournament import (
    _parse_positive_floats,
    _parse_positive_ints,
    select_manifest_modules,
)


def test_parse_lists_are_positive_and_deduplicated():
    assert _parse_positive_ints("128,64,128") == (128, 64)
    assert _parse_positive_floats("2.25,3,2.25") == (2.25, 3.0)
    with pytest.raises(ValueError, match="positive"):
        _parse_positive_ints("128,0")


def test_manifest_selection_uses_ranked_unconverted_mlp_only():
    manifest = {
        "candidates": [
            {"matrix_name": "attention", "category": "self_q"},
            {"matrix_name": "already", "category": "mlp_out"},
            {"matrix_name": "first", "category": "mlp_out"},
            {"matrix_name": "second", "category": "mlp_in"},
        ]
    }
    assert select_manifest_modules(
        manifest, top_mlp=2, already_converted={"already"}
    ) == ("first", "second")


def test_manifest_selection_rejects_short_mlp_list():
    with pytest.raises(ValueError, match="only 1"):
        select_manifest_modules(
            {"candidates": [{"matrix_name": "one", "category": "mlp_in"}]},
            top_mlp=2,
            already_converted=set(),
        )
