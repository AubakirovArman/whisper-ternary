import pytest
from pathlib import Path
from types import SimpleNamespace

from whisper_parallel_screen import _worker_command, _worker_slots


def test_worker_slots_repeat_each_physical_device() -> None:
    assert _worker_slots(
        ["2", "3"],
        workers_per_device=2,
        group_count=10,
    ) == ["2", "2", "3", "3"]


def test_worker_slots_do_not_exceed_group_count() -> None:
    assert _worker_slots(
        ["2", "3"],
        workers_per_device=3,
        group_count=3,
    ) == ["2", "2", "2"]


def test_worker_slots_reject_non_positive_count() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _worker_slots(["2"], workers_per_device=0, group_count=1)


def test_worker_command_forwards_offsets_and_feature_caches() -> None:
    args = SimpleNamespace(
        model="model",
        parent_checkpoint=Path("/parent.pt"),
        formats="t3",
        dataset="dataset",
        dataset_config="clean",
        train_split="train.100",
        train_offset=5120,
        train_feature_cache_manifest=Path("/train/manifest.json"),
        eval_split="test",
        eval_samples=256,
        eval_offset=2048,
        eval_feature_cache_manifest=Path("/eval/manifest.json"),
        moment_samples=256,
        batch_size=16,
        eval_batch_size=16,
        bootstrap_replicates=20_000,
        seed=157,
        train_dataset_config=None,
        eval_dataset_config=None,
        local_files_only=True,
    )
    command = _worker_command(
        args,
        [SimpleNamespace(name="decoder.0.cross_k")],
        Path("/output.json"),
    )
    assert command[command.index("--train-offset") + 1] == "5120"
    assert command[command.index("--train-feature-cache-manifest") + 1] == (
        "/train/manifest.json"
    )
    assert command[command.index("--eval-feature-cache-manifest") + 1] == (
        "/eval/manifest.json"
    )
