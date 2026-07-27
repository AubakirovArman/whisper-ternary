"""Materialise a 30 s *packed* Whisper QAT corpus from LibriSpeech.

Why this exists
---------------
The per-utterance cache built by :mod:`qat_build_data` tops out at ~104 label
tokens (median 45): LibriSpeech utterances average 12.3 s, so no training target
ever spans a full 30 s Whisper window.  A ternary student trained on that cache
learns to transcribe but never learns to *stop* on long audio -- with the decode
cap lifted it runs away to hundreds of tokens and pours in insertions, while its
substitution/deletion counts stay unchanged.  Fixing that needs targets that
cover the whole length range, which means training windows that are actually
full.

How packing works
-----------------
LibriSpeech utterance ids read ``speaker-chapter-index``, and consecutive
indices inside a chapter are consecutive slices of one continuous audiobook
reading.  Gluing adjacent utterances back together therefore reconstructs real,
coherent speech rather than a collage.

**Nothing is ever cut.**  Utterances are appended whole; a group is closed as
soon as the next utterance would push it past the 30 s window, or the chapter or
speaker changes, or the index stops being consecutive.  Every group keeps
complete audio and the complete concatenated transcript, so the target always
describes exactly what is in the window.

The parquet stream already emits each chapter as one contiguous, gap-free run
(verified by assertion below), so packing is a single online pass: no shuffling,
no second read of the 57 GiB of source FLAC.

Output layout is byte-for-byte the layout of :mod:`qat_build_data`
(``manifest.json`` + ``shards/features_%05d.npy`` + ``shards/shard_%05d.json``),
so :class:`wal_tat.qat.data.WhisperFeatureShards` and ``qat_train.py --data``
read it with no changes.  Each packed group simply plays the role one utterance
used to play.

Only *training* splits are read.  ``dev-clean``/``test-clean`` are never opened,
and the model-selection holdout is carved out of the training corpus by whole
speakers.

Example::

    CUDA_VISIBLE_DEVICES=2 python experiments/qat_build_packed.py \\
        --model openai/whisper-small \\
        --output-dir cache/qat/features/whisper-small-packed30-955h \\
        --splits train-clean-100 train-clean-360 train-other-500 \\
        --device cuda
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Iterator, Optional, Sequence

import numpy as np
import torch

from wal_tat.campaign import atomic_write_json
from wal_tat.orchestration import sha256_file
from wal_tat.qat.data import (
    CACHE_FORMAT_VERSION,
    LIBRISPEECH_REPO,
    LIBRISPEECH_TRAIN_SPLITS,
    SAMPLING_RATE,
    SHARD_DIRNAME,
    CacheSummary,
    ShardRecord,
    Utterance,
    _decode_flac,
    _normalise_text,
    _select_validation,
    _shard_paths,
    _tree_bytes,
    _write_features,
    librispeech_parquet_shards,
    summarize_manifest,
)

#: Whisper's receptive field.  480000 samples at 16 kHz.
DEFAULT_WINDOW_SECONDS = 30.0

#: ``WhisperConfig.max_target_positions`` for every Whisper checkpoint.  A label
#: sequence longer than this cannot be fed to the decoder at all, so we refuse
#: to write one rather than let training die thousands of steps in.
DECODER_POSITION_LIMIT = 448

#: Acceptance thresholds for the packed label-length distribution.
#:
#: These are set from a measured ceiling, not from a guess.  Enumerating the
#: *best* contiguous <=30 s window at all 281241 positions of the 955 h corpus
#: -- an upper bound no packing strategy can beat -- gives median 78, p90 101,
#: p99 117, max 156 tokens.  Exactly one window in 955 hours reaches 150 tokens
#: and none reaches 180.  LibriSpeech read speech runs at 3.03 Whisper BPE
#: tokens per second, so a 30 s window simply cannot carry more than ~156.
#: Thresholds above that would be unsatisfiable by construction; these sit just
#: under the ceiling so they still fail loudly if packing regresses.
MIN_P99_TOKENS = 110.0
MIN_MAX_TOKENS = 150.0

#: A packed window must also be *full*, which is the property that actually
#: teaches the decoder to stop: the per-utterance cache averaged 12.3 s of
#: speech per 30 s window, packing lifts that to 23.4 s (77.9%, provably the
#: maximum, see ``_Packer``).
MIN_MEAN_FILL_SECONDS = 22.0

_PARQUET_COLUMNS = ("id", "audio", "text", "speaker_id", "chapter_id")
_PROGRESS_NAME = "_progress.json"


# --------------------------------------------------------------------------- #
# Source stream
# --------------------------------------------------------------------------- #


def _count_rows(shards: Sequence[Path]) -> int:
    """Total source rows, read from the parquet footers alone."""
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(path).metadata.num_rows for path in shards)


def _iter_rows(
    shards: Sequence[Path], *, skip: int = 0, limit: Optional[int] = None,
    batch_size: int = 128,
) -> Iterator[dict[str, Any]]:
    """Yield source rows in canonical (file, row) order, skipping the first ``skip``.

    ``limit`` is an *absolute* position in the stream, not a count of yielded
    rows, so it means the same thing whether or not the build is resuming.

    Whole parquet files inside the skipped prefix are stepped over using their
    footer metadata, so resuming a half-finished build costs a few milliseconds
    instead of re-reading tens of gigabytes of FLAC.
    """
    import pyarrow.parquet as pq

    consumed = 0
    for shard in shards:
        reader = pq.ParquetFile(shard)
        rows_here = reader.metadata.num_rows
        if consumed + rows_here <= skip:
            consumed += rows_here
            reader.close()
            continue
        for record_batch in reader.iter_batches(
            batch_size=batch_size, columns=list(_PARQUET_COLUMNS)
        ):
            if consumed + record_batch.num_rows <= skip:
                consumed += record_batch.num_rows
                continue
            for row in record_batch.to_pylist():
                if consumed < skip:
                    consumed += 1
                    continue
                if limit is not None and consumed >= limit:
                    reader.close()
                    return
                yield row
                consumed += 1
        reader.close()


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #


@dataclass
class _Group:
    """One packed 30 s window: whole utterances only, in reading order."""

    speaker_id: str
    chapter_id: str
    utterance_ids: list[str]
    indices: list[int]
    texts: list[str]
    waveforms: list[np.ndarray]

    @property
    def num_samples(self) -> int:
        return sum(int(w.size) for w in self.waveforms)

    @property
    def utterance_id(self) -> str:
        return (
            f"{self.speaker_id}-{self.chapter_id}-"
            f"{self.indices[0]:04d}_{self.indices[-1]:04d}"
        )

    def concatenate(self) -> np.ndarray:
        """Glue the member waveforms, asserting not one sample was lost."""
        expected = self.num_samples
        joined = np.concatenate(self.waveforms) if len(self.waveforms) > 1 else (
            self.waveforms[0]
        )
        if int(joined.size) != expected:
            raise AssertionError(
                f"{self.utterance_id}: concatenated {joined.size} samples but the "
                f"{len(self.waveforms)} sources hold {expected} -- audio was lost"
            )
        return np.ascontiguousarray(joined, dtype=np.float32)


class _Packer:
    """Greedy online packer over the canonical LibriSpeech row stream.

    Closes the current group -- never truncates it -- whenever the next
    utterance would overflow the window, the (speaker, chapter) changes, or the
    utterance index is not the successor of the previous one.

    Greedy is *optimal* here, not merely convenient.  The utterances cannot be
    reordered (consecutive indices are what makes the glued audio continuous),
    so this is sequence partitioning under a hard capacity, where first-fit
    provably minimises the number of parts.  Total audio is fixed, so the
    minimum part count is also the maximum mean fill: 23.4 s of speech per 30 s
    window.  The 6.6 s of headroom that remains is structural -- LibriSpeech
    utterances average 12.3 s, so two fit (24.6 s) and three do not (36.9 s).
    """

    def __init__(self, window_samples: int) -> None:
        self.window_samples = int(window_samples)
        self._open: Optional[_Group] = None
        self.oversized: list[tuple[str, int]] = []
        self.boundary_counts = {"chapter": 0, "gap": 0, "window": 0, "eof": 0}

    def push(
        self,
        *,
        utterance_id: str,
        speaker_id: str,
        chapter_id: str,
        index: int,
        text: str,
        waveform: np.ndarray,
    ) -> Optional[_Group]:
        samples = int(waveform.size)
        if samples > self.window_samples:
            # Cannot be represented without cutting, and cutting is forbidden.
            self.oversized.append((utterance_id, samples))
            closed = self.flush("window")
            return closed

        closed: Optional[_Group] = None
        group = self._open
        if group is not None:
            if group.speaker_id != speaker_id or group.chapter_id != chapter_id:
                closed = self.flush("chapter")
            elif index != group.indices[-1] + 1:
                closed = self.flush("gap")
            elif group.num_samples + samples > self.window_samples:
                closed = self.flush("window")

        if self._open is None:
            self._open = _Group(
                speaker_id=speaker_id,
                chapter_id=chapter_id,
                utterance_ids=[utterance_id],
                indices=[index],
                texts=[text],
                waveforms=[waveform],
            )
        else:
            self._open.utterance_ids.append(utterance_id)
            self._open.indices.append(index)
            self._open.texts.append(text)
            self._open.waveforms.append(waveform)
        return closed

    def flush(self, reason: str) -> Optional[_Group]:
        group = self._open
        self._open = None
        if group is not None:
            self.boundary_counts[reason] = self.boundary_counts.get(reason, 0) + 1
        return group


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def _percentiles(values: Sequence[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def build_packed_cache(
    output_dir: Path | str,
    *,
    model_id: str,
    splits: Sequence[str],
    dataset_cache_dir: Optional[Path | str] = None,
    download: bool = False,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    shard_size: int = 256,
    max_source_utterances: Optional[int] = None,
    validation_size: int = 1536,
    validation_mode: str = "speaker",
    validation_seed: int = 0,
    language: str = "en",
    task: str = "transcribe",
    text_case: str = "lower",
    leading_space: bool = True,
    device: str = "cuda",
    decode_workers: int = 32,
    extract_batch: int = 64,
    local_files_only: bool = False,
    min_p99_tokens: float = MIN_P99_TOKENS,
    min_max_tokens: float = MIN_MAX_TOKENS,
    min_mean_fill_seconds: float = MIN_MEAN_FILL_SECONDS,
    verify_samples: int = 64,
    resume: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Pack whole LibriSpeech utterances into 30 s windows and cache the features."""
    from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if not splits:
        raise ValueError("at least one split is required")
    unknown = [name for name in splits if name not in LIBRISPEECH_TRAIN_SPLITS]
    if unknown:
        raise ValueError(
            f"unknown split(s) {unknown}; expected {sorted(LIBRISPEECH_TRAIN_SPLITS)}"
        )

    root = Path(output_dir).expanduser().resolve()
    (root / SHARD_DIRNAME).mkdir(parents=True, exist_ok=True)
    emit = progress or (lambda _message: None)

    window_samples = int(round(window_seconds * SAMPLING_RATE))
    emit(f"window {window_seconds:.1f} s = {window_samples} samples @ {SAMPLING_RATE} Hz")

    extractor = WhisperFeatureExtractor.from_pretrained(
        model_id, local_files_only=local_files_only
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        model_id, language=language, task=task, local_files_only=local_files_only
    )
    n_mels = int(extractor.feature_size)
    n_frames = int(extractor.nb_max_frames)
    if int(extractor.sampling_rate) != SAMPLING_RATE:
        raise ValueError(
            f"{model_id} expects {extractor.sampling_rate} Hz, "
            f"LibriSpeech is {SAMPLING_RATE} Hz"
        )
    if window_samples > int(extractor.n_samples):
        raise ValueError(
            f"window of {window_samples} samples exceeds the extractor's "
            f"{extractor.n_samples}-sample field; features would be truncated"
        )

    shard_files: list[Path] = []
    for name in splits:
        config, split = LIBRISPEECH_TRAIN_SPLITS[name]
        if "train" not in split:
            raise ValueError(f"{name} is not a training split; refusing to read it")
        found = librispeech_parquet_shards(
            config, split, cache_dir=dataset_cache_dir, download=download
        )
        emit(f"  {name:<20} {config}/{split:<10} {len(found)} parquet shards")
        shard_files.extend(found)
    if not shard_files:
        raise FileNotFoundError("no parquet shards resolved for the requested splits")
    for path in shard_files:
        # dev-clean / test-clean must stay pristine for final reporting.
        if any(part in {"validation", "test"} for part in path.parts):
            raise ValueError(f"refusing to read an evaluation split: {path}")

    # -- resume ------------------------------------------------------------- #
    progress_path = root / _PROGRESS_NAME
    start_shard = 0
    rows_skipped = 0
    if resume and progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            state.get("window_samples") == window_samples
            and state.get("shard_size") == shard_size
            and state.get("splits") == list(splits)
            and state.get("model_id") == model_id
        ):
            start_shard = int(state["num_shards"])
            rows_skipped = int(state["rows_consumed"])
            paths_ok = all(
                _shard_paths(root, i)[0].exists() and _shard_paths(root, i)[1].exists()
                for i in range(start_shard)
            )
            if paths_ok and start_shard:
                emit(
                    f"resuming after shard {start_shard - 1}: "
                    f"{rows_skipped} source utterances already packed"
                )
            else:
                start_shard, rows_skipped = 0, 0
        if start_shard == 0:
            emit("existing progress file does not match this configuration; rebuilding")

    records: list[ShardRecord] = []
    items: list[dict[str, Any]] = []
    features_bytes = 0
    total_seconds = 0.0
    rows_consumed = rows_skipped

    for index in range(start_shard):
        payload = json.loads(
            _shard_paths(root, index)[1].read_text(encoding="utf-8")
        )
        record = ShardRecord.from_json(payload["record"])
        records.append(record)
        items.extend(payload["items"])
        features_bytes += record.features_bytes
        total_seconds += sum(float(i["duration_seconds"]) for i in payload["items"])

    packer = _Packer(window_samples)
    pending: list[_Group] = []
    started = time.time()

    def emit_shard(groups: Sequence[_Group], consumed_after: int) -> None:
        """Extract, tokenize and durably write one shard of packed groups."""
        nonlocal features_bytes, total_seconds
        index = len(records)
        features_path, meta_path = _shard_paths(root, index)

        waveforms: list[np.ndarray] = []
        texts: list[str] = []
        for group in groups:
            joined = group.concatenate()
            if joined.size > window_samples:
                raise AssertionError(
                    f"{group.utterance_id}: {joined.size / SAMPLING_RATE:.3f} s "
                    f"exceeds the {window_seconds:.1f} s window"
                )
            waveforms.append(joined)
            parts = [
                _normalise_text(text, text_case=text_case, leading_space=False)
                for text in group.texts
            ]
            merged = " ".join(part for part in parts if part)
            if leading_space and merged:
                merged = " " + merged
            texts.append(merged)

        blocks: list[np.ndarray] = []
        for begin in range(0, len(waveforms), extract_batch):
            chunk = waveforms[begin : begin + extract_batch]
            encoded = extractor(
                chunk,
                sampling_rate=SAMPLING_RATE,
                padding="max_length",
                return_tensors="pt",
                device=device,
            )["input_features"]
            blocks.append(encoded.to(torch.float16).cpu().numpy())
        block = np.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0]
        if block.shape[1:] != (n_mels, n_frames):
            raise ValueError(
                f"feature extractor returned {block.shape[1:]}, "
                f"expected {(n_mels, n_frames)}"
            )
        _write_features(features_path, block)

        label_ids = tokenizer(texts)["input_ids"]
        shard_items: list[dict[str, Any]] = []
        for position, group in enumerate(groups):
            labels = tuple(int(token) for token in label_ids[position])
            if len(labels) > DECODER_POSITION_LIMIT:
                raise AssertionError(
                    f"{group.utterance_id}: {len(labels)} label tokens exceeds the "
                    f"decoder's {DECODER_POSITION_LIMIT} positions"
                )
            samples = int(waveforms[position].size)
            record_item = Utterance(
                utterance_id=group.utterance_id,
                text=texts[position],
                speaker_id=group.speaker_id,
                duration_seconds=samples / SAMPLING_RATE,
                labels=labels,
            ).to_json()
            record_item["chapter_id"] = group.chapter_id
            record_item["num_source_utterances"] = len(group.utterance_ids)
            record_item["source_utterance_ids"] = list(group.utterance_ids)
            record_item["num_samples"] = samples
            shard_items.append(record_item)

        record = ShardRecord(
            index=index,
            count=len(groups),
            features_file=f"{SHARD_DIRNAME}/{features_path.name}",
            meta_file=f"{SHARD_DIRNAME}/{meta_path.name}",
            features_sha256=sha256_file(features_path),
            features_bytes=features_path.stat().st_size,
        )
        atomic_write_json(
            meta_path, {"record": record.to_json(), "items": shard_items}
        )
        records.append(record)
        items.extend(shard_items)
        features_bytes += record.features_bytes
        total_seconds += sum(float(i["duration_seconds"]) for i in shard_items)
        # Shards hold whole groups and groups hold whole source utterances, so
        # restarting the stream here reproduces the identical greedy packing.
        atomic_write_json(
            progress_path,
            {
                "num_shards": len(records),
                "rows_consumed": consumed_after,
                "window_samples": window_samples,
                "shard_size": shard_size,
                "splits": list(splits),
                "model_id": model_id,
            },
        )

    def open_rows() -> int:
        return 0 if packer._open is None else len(packer._open.utterance_ids)

    def handle(group: Optional[_Group]) -> None:
        nonlocal pending
        if group is None:
            return
        pending.append(group)
        if len(pending) >= shard_size:
            # Rows already committed to *closed* groups; the open group's rows
            # are replayed on resume.
            emit_shard(pending, rows_consumed - open_rows())
            pending = []
            if len(records) % 25 == 0:
                rate = (rows_consumed - rows_skipped) / max(time.time() - started, 1e-9)
                emit(
                    f"shard {len(records):5d} | {len(items):7d} groups | "
                    f"{rows_consumed:7d}/{total_source} utts | "
                    f"{total_seconds / 3600:7.2f} h | "
                    f"{features_bytes / 2**30:6.2f} GiB | {rate:6.1f} utt/s"
                )

    def consume(rows: Sequence[dict[str, Any]]) -> None:
        nonlocal rows_consumed
        waves = list(pool.map(lambda r: _decode_flac(r["audio"]), rows))
        for source, waveform in zip(rows, waves):
            utterance_id = str(source["id"])
            speaker_id = str(source["speaker_id"])
            chapter_id = str(source["chapter_id"])
            parts = utterance_id.split("-")
            if len(parts) != 3 or parts[0] != speaker_id or parts[1] != chapter_id:
                raise AssertionError(
                    f"utterance id {utterance_id!r} disagrees with "
                    f"speaker {speaker_id} / chapter {chapter_id}"
                )
            rows_consumed += 1
            handle(
                packer.push(
                    utterance_id=utterance_id,
                    speaker_id=speaker_id,
                    chapter_id=chapter_id,
                    index=int(parts[2]),
                    text=str(source["text"]),
                    waveform=waveform,
                )
            )

    total_source = _count_rows(shard_files)
    if max_source_utterances is not None:
        total_source = min(total_source, max_source_utterances)
    emit(f"source utterances: {total_source}")

    batch: list[dict[str, Any]] = []
    pool = ThreadPoolExecutor(max_workers=decode_workers)
    try:
        for row in _iter_rows(
            shard_files, skip=rows_skipped, limit=max_source_utterances
        ):
            batch.append(row)
            if len(batch) >= decode_workers * 4:
                consume(batch)
                batch = []
        if batch:
            consume(batch)
            batch = []
    finally:
        pool.shutdown(wait=True)

    tail = packer.flush("eof")
    if tail is not None:
        pending.append(tail)
    if pending:
        emit_shard(pending, rows_consumed)
        pending = []

    if not items:
        raise ValueError("no packed groups were materialised")
    if packer.oversized:
        emit(
            f"WARNING: {len(packer.oversized)} source utterance(s) are longer than "
            f"the window and were dropped rather than cut: {packer.oversized[:5]}"
        )

    # -- holdout ------------------------------------------------------------ #
    validation = _select_validation(
        [str(item["speaker_id"]) for item in items],
        size=validation_size,
        mode=validation_mode,
        seed=validation_seed,
    )
    validation_set = set(validation)
    num_train = len(items) - len(validation_set)
    validation_speakers = sorted({str(items[i]["speaker_id"]) for i in validation})
    train_speakers = {
        str(item["speaker_id"])
        for i, item in enumerate(items)
        if i not in validation_set
    }
    leak = train_speakers.intersection(validation_speakers)
    if leak:
        raise AssertionError(f"speaker leak between train and validation: {sorted(leak)}")

    # -- statistics + assertions -------------------------------------------- #
    label_lengths = [len(item["labels"]) for item in items]
    durations = [float(item["duration_seconds"]) for item in items]
    members = [int(item["num_source_utterances"]) for item in items]
    label_stats = _percentiles(label_lengths)
    duration_stats = _percentiles([int(round(d * 1000)) for d in durations])
    duration_stats = {k: (v / 1000.0 if k != "count" else v) for k, v in duration_stats.items()}
    member_stats = _percentiles(members)

    over_window = [
        item["utterance_id"]
        for item in items
        if int(item["num_samples"]) > window_samples
    ]
    if over_window:
        raise AssertionError(
            f"{len(over_window)} group(s) exceed the {window_seconds:.1f} s window: "
            f"{over_window[:5]}"
        )
    mismatched = [
        item["utterance_id"]
        for item in items
        if len(item["source_utterance_ids"]) != int(item["num_source_utterances"])
    ]
    if mismatched:
        raise AssertionError(f"group membership bookkeeping is inconsistent: {mismatched[:5]}")

    packed_sources = sum(members)
    expected_sources = rows_consumed - len(packer.oversized)
    if packed_sources != expected_sources:
        raise AssertionError(
            f"packed {packed_sources} source utterances but consumed "
            f"{rows_consumed} rows ({len(packer.oversized)} oversized) -- "
            "utterances were lost or duplicated"
        )
    seen_sources = set()
    for item in items:
        for uid in item["source_utterance_ids"]:
            if uid in seen_sources:
                raise AssertionError(f"source utterance {uid} packed twice")
            seen_sources.add(uid)

    # -- round-trip verification -------------------------------------------- #
    rng = random.Random(validation_seed + 1)
    sample_positions = rng.sample(range(len(items)), k=min(verify_samples, len(items)))
    for position in sample_positions:
        item = items[position]
        decoded = tokenizer.decode(item["labels"], skip_special_tokens=True)
        if decoded != item["text"]:
            raise AssertionError(
                f"{item['utterance_id']}: labels decode to {decoded!r}, "
                f"expected the joined transcript {item['text']!r}"
            )
    emit(f"verified {len(sample_positions)} groups: labels decode back to the joined text")

    failures: list[str] = []
    if label_stats["p99"] < min_p99_tokens:
        failures.append(
            f"label p99 is {label_stats['p99']:.0f} tokens, below the required "
            f"{min_p99_tokens:.0f}"
        )
    if label_stats["max"] < min_max_tokens:
        failures.append(
            f"longest label is {label_stats['max']:.0f} tokens, below the required "
            f"{min_max_tokens:.0f}"
        )
    mean_fill = float(np.mean(durations))
    if mean_fill < min_mean_fill_seconds:
        failures.append(
            f"windows average {mean_fill:.2f} s of speech, below the required "
            f"{min_mean_fill_seconds:.2f} s -- packing left the windows mostly empty"
        )

    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "model_id": model_id,
        "source": {
            "repo": LIBRISPEECH_REPO,
            "splits": list(splits),
            "parquet_root": str(dataset_cache_dir) if dataset_cache_dir else None,
            "max_utterances": max_source_utterances,
            "num_source_utterances": packed_sources,
        },
        "features": {
            "dtype": "float16",
            "n_mels": n_mels,
            "n_frames": n_frames,
            "sampling_rate": SAMPLING_RATE,
            "padding": "max_length",
        },
        "labels": {
            "tokenizer": model_id,
            "language": language,
            "task": task,
            "text_case": text_case,
            "leading_space": leading_space,
            "pad_token_id": int(tokenizer.pad_token_id),
            "eos_token_id": int(tokenizer.eos_token_id),
            "decoder_start_token_id": int(
                tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
            ),
        },
        "packing": {
            "window_seconds": window_seconds,
            "window_samples": window_samples,
            "policy": "whole utterances only; no audio or text is ever cut",
            "group_key": "(speaker_id, chapter_id) with consecutive utterance index",
            "boundary_reasons": packer.boundary_counts,
            "oversized_dropped": [uid for uid, _ in packer.oversized],
            "label_tokens": label_stats,
            "group_seconds": duration_stats,
            "group_members": member_stats,
            "audio_fill_ratio": round(
                float(np.mean(durations)) / window_seconds, 4
            ),
        },
        "split": {
            "mode": validation_mode,
            "seed": validation_seed,
            "requested_validation": validation_size,
            "num_train": num_train,
            "num_validation": len(validation),
            "validation_indices": list(validation),
            "validation_speakers": validation_speakers,
        },
        "shard_size": shard_size,
        "shards": [record.to_json() for record in records],
        "num_utterances": len(items),
        "total_audio_seconds": round(total_seconds, 3),
    }
    atomic_write_json(root / "manifest.json", manifest)

    summary = CacheSummary(
        root=root,
        num_utterances=len(items),
        num_train=num_train,
        num_validation=len(validation),
        num_shards=len(records),
        total_audio_seconds=total_seconds,
        features_bytes=features_bytes,
        on_disk_bytes=_tree_bytes(root),
    )
    report = summary.to_json() | {
        "model_id": model_id,
        "splits": list(splits),
        "num_source_utterances": packed_sources,
        "label_tokens": label_stats,
        "group_seconds": duration_stats,
        "group_members": member_stats,
        "audio_fill_seconds": float(np.mean(durations)),
        "audio_fill_ratio": float(np.mean(durations)) / window_seconds,
        "boundary_reasons": packer.boundary_counts,
        "oversized_dropped": len(packer.oversized),
        "assertion_failures": failures,
    }
    emit(summary.describe())
    emit(
        "label tokens   median {median:.0f} | p90 {p90:.0f} | p99 {p99:.0f} | "
        "max {max:.0f}".format(**label_stats)
    )
    emit(
        "group seconds  median {median:.2f} | p90 {p90:.2f} | p99 {p99:.2f} | "
        "max {max:.2f}".format(**duration_stats)
    )
    emit(
        "group members  median {median:.0f} | mean {mean:.2f} | "
        "max {max:.0f}".format(**member_stats)
    )
    emit(
        f"audio fill     {np.mean(durations):.2f} s of speech per "
        f"{window_seconds:.0f} s window "
        f"({100 * np.mean(durations) / window_seconds:.1f}%)"
    )
    if failures:
        for line in failures:
            emit(f"ASSERTION FAILED: {line}")
        raise AssertionError(
            "packing did not reach the required label lengths: " + "; ".join(failures)
        )
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train-clean-100", "train-clean-360", "train-other-500"],
        choices=sorted(LIBRISPEECH_TRAIN_SPLITS),
    )
    parser.add_argument("--dataset-cache-dir", type=Path, default=None)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument(
        "--max-source-utterances",
        type=int,
        default=None,
        help="Truncate the *source* stream, mainly for smoke tests.",
    )
    parser.add_argument("--validation-size", type=int, default=1536)
    parser.add_argument(
        "--validation-mode", default="speaker", choices=("speaker", "tail")
    )
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument("--language", default="en")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--text-case", default="lower", choices=("lower", "upper", "raw"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode-workers", type=int, default=32)
    parser.add_argument("--extract-batch", type=int, default=64)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--min-p99-tokens", type=float, default=MIN_P99_TOKENS)
    parser.add_argument("--min-max-tokens", type=float, default=MIN_MAX_TOKENS)
    parser.add_argument(
        "--min-mean-fill-seconds", type=float, default=MIN_MEAN_FILL_SECONDS
    )
    parser.add_argument("--verify-samples", type=int, default=64)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()

    def emit(message: str) -> None:
        print(f"[{time.time() - started:8.1f}s] {message}", flush=True)

    report = build_packed_cache(
        args.output_dir,
        model_id=args.model,
        splits=tuple(args.splits),
        dataset_cache_dir=args.dataset_cache_dir,
        download=args.download,
        window_seconds=args.window_seconds,
        shard_size=args.shard_size,
        max_source_utterances=args.max_source_utterances,
        validation_size=args.validation_size,
        validation_mode=args.validation_mode,
        validation_seed=args.validation_seed,
        language=args.language,
        task=args.task,
        text_case=args.text_case,
        device=args.device,
        decode_workers=args.decode_workers,
        extract_batch=args.extract_batch,
        local_files_only=args.local_files_only,
        min_p99_tokens=args.min_p99_tokens,
        min_max_tokens=args.min_max_tokens,
        min_mean_fill_seconds=args.min_mean_fill_seconds,
        verify_samples=args.verify_samples,
        resume=not args.no_resume,
        progress=emit,
    )

    # Re-read from disk so the printed report is what a consumer actually sees.
    verified = summarize_manifest(args.output_dir)
    payload = report | {"elapsed_seconds": round(time.time() - started, 1)}
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    emit("reloaded manifest:")
    print(verified.describe(), flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    if verified.num_utterances != report["num_utterances"]:
        raise RuntimeError("manifest disagrees with the in-memory build summary")
    print("[DONE]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
