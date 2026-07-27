import json
from pathlib import Path
import sys

import pytest


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from whisper_lowbit_package_wer_screen import (  # noqa: E402
    load_packages,
    package_representation_summary,
)


def test_load_packages_validates_and_normalizes(tmp_path):
    path = tmp_path / "packages.json"
    path.write_text(
        json.dumps(
            {
                "packages": [
                    {
                        "name": "mixed",
                        "assignments": {
                            "model.encoder.layers.8.fc2": "mixed-cq2q4-gain"
                        },
                    }
                ]
            }
        )
    )
    packages = load_packages(path)
    assert packages == [
        {
            "name": "mixed",
            "assignments": {
                "model.encoder.layers.8.fc2": "mixed-cq2q4-gain"
            },
            "description": None,
        }
    ]


def test_load_packages_rejects_duplicate_names(tmp_path):
    path = tmp_path / "packages.json"
    path.write_text(
        json.dumps(
            {
                "packages": [
                    {"name": "same", "assignments": {"a": "t3-g128"}},
                    {"name": "same", "assignments": {"b": "nz4-g128"}},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="unique"):
        load_packages(path)


def test_package_representation_summary_is_weighted_by_payload():
    summary = package_representation_summary(
        {
            "a": {"payload_bits": 200, "weighted_error": 3.0},
            "b": {"payload_bits": 600, "weighted_error": 5.0},
        },
        {"a": 100, "b": 300},
    )
    assert summary == {
        "payload_bits": 800,
        "weights": 400,
        "physical_bpw": 2.0,
        "weighted_error": 8.0,
    }
