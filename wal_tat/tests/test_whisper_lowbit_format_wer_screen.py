import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from whisper_lowbit_format_wer_screen import (
    _group_label,
    _mixed_label,
    build_format_projection,
)


@pytest.mark.parametrize(
    "label",
    (
        "t3-g4",
        "nz4-g4",
        "q4-g4",
        "cq2",
        "mixed-cq2q4-gain",
        "mixed-cq2q4-outlier",
    ),
)
def test_projection_builder_returns_matching_reference_layer(label):
    generator = torch.Generator().manual_seed(5)
    linear = nn.Linear(8, 16, bias=True)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(linear.weight.shape, generator=generator))
    projected, statistics = build_format_projection(
        linear,
        torch.ones(8),
        label=label,
        q4_fraction=0.25,
        kmeans_iterations=4,
        column_chunk_size=4,
    )
    value = torch.randn((2, 8), generator=generator)
    assert projected(value).shape == (2, 16)
    assert statistics["payload_bits"] > 0
    assert statistics["physical_bpw"] > 0
    assert statistics["relative_weighted_error"] >= 0


def test_group_label_rejects_unknown_formats():
    assert _group_label("t3-g128") == ("t3", 128)
    with pytest.raises(ValueError, match="unsupported"):
        _group_label("cq2")


def test_mixed_label_supports_per_format_fraction_override():
    assert _mixed_label("mixed-cq2q4-gain", 0.05) == (
        "mixed-cq2q4-gain",
        0.05,
    )
    assert _mixed_label("mixed-cq2q4-gain@0.20", 0.05) == (
        "mixed-cq2q4-gain",
        0.20,
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _mixed_label("mixed-cq2q4-gain@1.20", 0.05)
