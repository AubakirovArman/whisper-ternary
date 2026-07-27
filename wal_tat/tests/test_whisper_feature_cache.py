import json
from types import SimpleNamespace

import pytest

from whisper_baseline import _cache_entries


def _args(manifest, *, offset=16, max_samples=16):
    return SimpleNamespace(
        feature_cache_manifest=manifest,
        model="model-id",
        dataset="dataset-id",
        dataset_config="clean",
        split="validation",
        batch_size=16,
        language="en",
        task="transcribe",
        dtype="bf16",
        offset=offset,
        max_samples=max_samples,
    )


def _manifest(path):
    payload = {
        "schema_version": 1,
        "kind": "whisper-batch-exact-feature-cache",
        "model_revision": "model-id",
        "dataset": "dataset-id",
        "dataset_config": "clean",
        "split": "validation",
        "batch_size": 16,
        "language": "en",
        "task": "transcribe",
        "feature_dtype": "bfloat16",
        "batches": [
            {"offset": 0, "samples": 16},
            {"offset": 16, "samples": 16},
            {"offset": 32, "samples": 16},
        ],
    }
    path.write_text(json.dumps(payload))
    return path


def test_cache_entries_select_complete_contiguous_batches(tmp_path) -> None:
    manifest, entries = _cache_entries(_args(_manifest(tmp_path / "manifest.json")))
    assert manifest["batch_size"] == 16
    assert entries == [{"offset": 16, "samples": 16}]


def test_cache_entries_reject_window_that_changes_batch_membership(tmp_path) -> None:
    args = _args(
        _manifest(tmp_path / "manifest.json"),
        offset=8,
        max_samples=16,
    )
    with pytest.raises(ValueError, match="complete cache batches"):
        _cache_entries(args)
