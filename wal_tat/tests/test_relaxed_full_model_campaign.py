import argparse
import sys
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from relaxed_full_model_campaign import generation_command, q8_command  # noqa: E402


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        source_checkpoint=Path("source.pt"),
        suite=Path("suite.pt"),
        moment_sequences=64,
        selection_gate_sequences=64,
        gate_ratio=1.1,
        incremental_gate_ratio=1.05,
        q4_candidate_generation_gate_ratio=1000.0,
        device="cuda",
    )


def _flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_q2_generation_uses_acceptance_gates():
    command = generation_command(
        _args(),
        parent=Path("parent.pt"),
        layer=12,
        tag="q2",
        precision="q2",
    )

    assert _flag_value(command, "--gate-ratio") == "1.1"
    assert _flag_value(command, "--incremental-gate-ratio") == "1.05"
    assert "--allow-zero-rescue" in command


def test_q4_generation_materializes_candidate_before_fresh_gate():
    command = generation_command(
        _args(),
        parent=Path("parent.pt"),
        layer=12,
        tag="q4",
        precision="q4",
    )

    assert _flag_value(command, "--gate-ratio") == "1000.0"
    assert _flag_value(command, "--incremental-gate-ratio") == "1000.0"
    assert "--allow-zero-rescue" not in command


def test_q8_generation_materializes_candidate_before_fresh_gate():
    command = q8_command(
        _args(),
        parent=Path("parent.pt"),
        q4_candidate=Path("q4.pt"),
        layer=12,
        tag="q8",
    )

    assert _flag_value(command, "--gate-ratio") == "1000.0"
    assert _flag_value(command, "--incremental-gate-ratio") == "1000.0"
