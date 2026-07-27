"""Unit tests for the QAT feature-shard corpus pipeline.

These exercise the reader, collator and split logic against a synthetic cache,
so they run without a network, a Whisper checkpoint, or LibriSpeech on disk.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from wal_tat.campaign import atomic_write_json
from wal_tat.orchestration import sha256_file
from wal_tat.qat.data import (
    CACHE_FORMAT_VERSION,
    ShardRecord,
    WhisperCollator,
    WhisperFeatureShards,
    build_dataloader,
    load_manifest,
    summarize_manifest,
)
from wal_tat.qat.data import _normalise_text, _select_validation

N_MELS = 4
N_FRAMES = 6
SOT = 50258
EOT = 50257


def _write_cache(
    root: Path,
    *,
    counts=(3, 3, 2),
    validation_indices=(6, 7),
) -> Path:
    """Materialise a tiny cache whose feature values encode their index."""
    (root / "shards").mkdir(parents=True, exist_ok=True)
    shards = []
    items = []
    global_index = 0
    for shard_index, count in enumerate(counts):
        block = np.stack(
            [
                np.full((N_MELS, N_FRAMES), global_index + offset, dtype=np.float16)
                for offset in range(count)
            ]
        )
        features_path = root / "shards" / f"features_{shard_index:05d}.npy"
        np.save(features_path, block, allow_pickle=False)
        shard_items = [
            {
                "utterance_id": f"spk{(global_index + offset) // 3}-{global_index + offset}",
                "text": f" utterance {global_index + offset}",
                "speaker_id": str((global_index + offset) // 3),
                "duration_seconds": 10.0 + offset,
                # Variable-length labels so padding is actually exercised.
                "labels": [SOT, 100 + global_index + offset] * (1 + offset) + [EOT],
            }
            for offset in range(count)
        ]
        record = ShardRecord(
            index=shard_index,
            count=count,
            features_file=f"shards/{features_path.name}",
            meta_file=f"shards/shard_{shard_index:05d}.json",
            features_sha256=sha256_file(features_path),
            features_bytes=features_path.stat().st_size,
        )
        atomic_write_json(
            root / record.meta_file,
            {"record": record.to_json(), "items": shard_items},
        )
        shards.append(record.to_json())
        items.extend(shard_items)
        global_index += count

    atomic_write_json(
        root / "manifest.json",
        {
            "format_version": CACHE_FORMAT_VERSION,
            "model_id": "openai/whisper-small",
            "source": {"repo": "openslr/librispeech_asr", "splits": ["train-clean-100"]},
            "features": {
                "dtype": "float16",
                "n_mels": N_MELS,
                "n_frames": N_FRAMES,
                "sampling_rate": 16_000,
                "padding": "max_length",
            },
            "labels": {
                "tokenizer": "openai/whisper-small",
                "language": "en",
                "task": "transcribe",
                "text_case": "lower",
                "leading_space": True,
                "pad_token_id": EOT,
                "eos_token_id": EOT,
                "decoder_start_token_id": SOT,
            },
            "split": {
                "mode": "speaker",
                "seed": 0,
                "requested_validation": len(validation_indices),
                "num_train": sum(counts) - len(validation_indices),
                "num_validation": len(validation_indices),
                "validation_indices": list(validation_indices),
                "validation_speakers": sorted(
                    {items[i]["speaker_id"] for i in validation_indices}
                ),
            },
            "shard_size": max(counts),
            "shards": shards,
            "num_utterances": sum(counts),
            "total_audio_seconds": sum(i["duration_seconds"] for i in items),
        },
    )
    return root


@pytest.fixture()
def cache(tmp_path: Path) -> Path:
    return _write_cache(tmp_path / "cache")


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #


def test_subsets_partition_the_corpus_without_overlap(cache: Path) -> None:
    train = WhisperFeatureShards(cache, subset="train")
    validation = WhisperFeatureShards(cache, subset="validation")
    everything = WhisperFeatureShards(cache, subset="all")

    assert (len(train), len(validation), len(everything)) == (6, 2, 8)
    train_ids = {train.utterance(i).utterance_id for i in range(len(train))}
    val_ids = {validation.utterance(i).utterance_id for i in range(len(validation))}
    assert not train_ids & val_ids
    assert len(train_ids | val_ids) == len(everything)


def test_getitem_reads_the_right_row_across_shard_boundaries(cache: Path) -> None:
    everything = WhisperFeatureShards(cache, subset="all")
    for position in range(len(everything)):
        sample = everything[position]
        assert sample.input_features.shape == (N_MELS, N_FRAMES)
        assert sample.input_features.dtype is torch.float16
        # Feature values were seeded with the global utterance index.
        assert torch.all(sample.input_features == float(position))
        assert sample.utterance_id.endswith(f"-{position}")


def test_returned_features_do_not_alias_the_memmap(cache: Path) -> None:
    everything = WhisperFeatureShards(cache, subset="all")
    sample = everything[0]
    sample.input_features.add_(1.0)
    assert torch.all(everything[0].input_features == 0.0)


def test_verify_sha256_detects_a_corrupted_shard(cache: Path) -> None:
    WhisperFeatureShards(cache, subset="all", verify_sha256=True)
    target = cache / "shards" / "features_00001.npy"
    block = np.load(target)
    block[0, 0, 0] += 1
    np.save(target, block, allow_pickle=False)
    with pytest.raises(ValueError, match="hashes to"):
        WhisperFeatureShards(cache, subset="all", verify_sha256=True)


def test_manifest_version_is_enforced(cache: Path) -> None:
    payload = load_manifest(cache)
    payload["format_version"] = CACHE_FORMAT_VERSION + 1
    atomic_write_json(cache / "manifest.json", payload)
    with pytest.raises(ValueError, match="format_version"):
        WhisperFeatureShards(cache, subset="all")


def test_unknown_subset_is_rejected(cache: Path) -> None:
    with pytest.raises(ValueError, match="unknown subset"):
        WhisperFeatureShards(cache, subset="dev-clean")


def test_summary_reports_hours_and_size(cache: Path) -> None:
    summary = summarize_manifest(cache)
    assert summary.num_utterances == 8
    assert summary.num_train == 6
    assert summary.num_validation == 2
    assert summary.total_audio_seconds == pytest.approx(8 * 10.0 + 0 + 1 + 2 + 0 + 1 + 2 + 0 + 1)
    assert summary.total_hours == pytest.approx(summary.total_audio_seconds / 3600)
    assert summary.on_disk_bytes >= summary.features_bytes > 0
    assert "utterances" in summary.describe()


# --------------------------------------------------------------------------- #
# Collation
# --------------------------------------------------------------------------- #


def test_collator_strips_decoder_start_and_masks_padding(cache: Path) -> None:
    everything = WhisperFeatureShards(cache, subset="all")
    batch = everything.default_collator()([everything[0], everything[1]])

    assert batch["input_features"].shape == (2, N_MELS, N_FRAMES)
    assert batch["input_features"].dtype is torch.float32
    # Row 0 has 3 label tokens and row 1 has 5, both minus the leading SOT.
    assert batch["labels"].shape == (2, 4)
    assert batch["labels"].tolist() == [
        [100, EOT, -100, -100],
        [101, SOT, 101, EOT],
    ]


def test_collator_keeps_decoder_start_when_disabled(cache: Path) -> None:
    everything = WhisperFeatureShards(cache, subset="all")
    collator = WhisperCollator(decoder_start_token_id=SOT, strip_decoder_start=False)
    batch = collator([everything[0]])
    assert batch["labels"][0, 0].item() == SOT


def test_collator_can_return_metadata(cache: Path) -> None:
    everything = WhisperFeatureShards(cache, subset="all")
    batch = everything.default_collator(include_metadata=True)(
        [everything[0], everything[3]]
    )
    assert batch["utterance_ids"] == ["spk0-0", "spk1-3"]
    assert batch["texts"] == [" utterance 0", " utterance 3"]


def test_collator_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        WhisperCollator(decoder_start_token_id=SOT)([])


# --------------------------------------------------------------------------- #
# DataLoader
# --------------------------------------------------------------------------- #


def _epoch_ids(dataset, **kwargs) -> list[str]:
    loader = build_dataloader(
        dataset,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        collator=dataset.default_collator(include_metadata=True),
        **kwargs,
    )
    return [uid for batch in loader for uid in batch["utterance_ids"]]


def test_dataloader_covers_the_subset_exactly_once(cache: Path) -> None:
    train = WhisperFeatureShards(cache, subset="train")
    ids = _epoch_ids(train, shuffle=True, seed=0)
    assert sorted(ids) == sorted(
        train.utterance(i).utterance_id for i in range(len(train))
    )


def test_seeded_shuffle_is_reproducible_and_seed_dependent(cache: Path) -> None:
    train = WhisperFeatureShards(cache, subset="train")
    assert _epoch_ids(train, shuffle=True, seed=7) == _epoch_ids(
        train, shuffle=True, seed=7
    )
    assert _epoch_ids(train, shuffle=True, seed=7) != _epoch_ids(
        train, shuffle=True, seed=8
    )


def test_unshuffled_loader_preserves_cache_order(cache: Path) -> None:
    train = WhisperFeatureShards(cache, subset="train")
    assert _epoch_ids(train, shuffle=False) == [
        train.utterance(i).utterance_id for i in range(len(train))
    ]


def test_dataloader_works_with_worker_processes(cache: Path) -> None:
    train = WhisperFeatureShards(cache, subset="train")
    loader = build_dataloader(
        train, batch_size=2, shuffle=False, num_workers=2, pin_memory=False
    )
    seen = sum(batch["input_features"].shape[0] for batch in loader)
    assert seen == len(train)


# --------------------------------------------------------------------------- #
# Split and text policy
# --------------------------------------------------------------------------- #


def test_speaker_holdout_never_splits_a_speaker() -> None:
    speakers = [str(i // 10) for i in range(100)]
    held = _select_validation(speakers, size=15, mode="speaker", seed=0)
    held_speakers = {speakers[i] for i in held}
    assert len(held) >= 15
    # Every utterance of a held-out speaker is held out.
    assert all((speakers[i] in held_speakers) == (i in set(held)) for i in range(100))


def test_tail_holdout_takes_the_final_utterances() -> None:
    speakers = [str(i // 10) for i in range(100)]
    assert _select_validation(speakers, size=4, mode="tail", seed=0) == (96, 97, 98, 99)


def test_holdout_rejects_a_size_that_swallows_the_corpus() -> None:
    with pytest.raises(ValueError, match="must be smaller"):
        _select_validation(["a", "b"], size=2, mode="speaker", seed=0)
    assert _select_validation(["a", "b"], size=0, mode="speaker", seed=0) == ()


def test_unknown_holdout_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown validation mode"):
        _select_validation(["a", "b"], size=1, mode="random", seed=0)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("lower", " mister quilter is"),
        ("upper", " MISTER QUILTER IS"),
        ("raw", " MISTER QUILTER is"),
    ],
)
def test_text_normalisation_cases(case: str, expected: str) -> None:
    assert (
        _normalise_text("MISTER   QUILTER is ", text_case=case, leading_space=True)
        == expected
    )


def test_text_normalisation_can_drop_the_leading_space() -> None:
    assert (
        _normalise_text("HELLO", text_case="lower", leading_space=False) == "hello"
    )


def test_unknown_text_case_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown text_case"):
        _normalise_text("HELLO", text_case="title", leading_space=True)
