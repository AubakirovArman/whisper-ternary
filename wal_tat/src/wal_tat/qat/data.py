"""Precomputed Whisper feature shards for quantisation-aware training.

QAT replays the same corpus for many epochs.  Decoding FLAC and recomputing
log-mel spectrograms every epoch dominates wall clock and pins CPU cores that
the training loop needs, so this module materialises the corpus *once* into
fixed-stride ``.npy`` shards and then serves them through an ordinary
``torch.utils.data`` pipeline backed by ``mmap``.

Cache layout::

    manifest.json                  corpus metadata, tokenizer ids, shard index
    shards/features_00000.npy      float16 [count, n_mels, n_frames]
    shards/shard_00000.json        per-utterance ids, texts, label token ids

Features are stored padded to Whisper's fixed 30 s window, so shard ``k`` is a
contiguous ``count x n_mels x n_frames`` block that ``numpy.load(mmap_mode="r")``
can index in constant time without deserialising the whole file.  float16 halves
the footprint versus float32 and is lossless for this use: log-mel values live
in roughly ``[-1.0, 1.0]`` after Whisper's normalisation, far inside half
precision's dynamic range.

Audio is read straight from the canonical ``openslr/librispeech_asr`` parquet
shards with ``pyarrow``.  That matches the reader in :mod:`wal_tat.qat.evaluate`
utterance for utterance, and avoids the multi-gigabyte Arrow copy that
``datasets.load_dataset`` writes alongside the parquet files it already
downloaded.

Typical use::

    summary = build_feature_cache(cache_dir, model_id="openai/whisper-small")
    print(summary.describe())

    train = WhisperFeatureShards(cache_dir, subset="train")
    loader = build_dataloader(train, batch_size=16, shuffle=True, seed=0)
    for batch in loader:
        loss = model(**batch).loss
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..campaign import atomic_write_json
from ..orchestration import sha256_file

__all__ = [
    "CACHE_FORMAT_VERSION",
    "DEFAULT_LIBRISPEECH_CACHE",
    "LIBRISPEECH_TRAIN_SPLITS",
    "CacheSummary",
    "FeatureSample",
    "ShardRecord",
    "Utterance",
    "WhisperCollator",
    "WhisperFeatureShards",
    "build_dataloader",
    "build_feature_cache",
    "librispeech_parquet_shards",
    "load_manifest",
    "summarize_manifest",
]


#: Bump whenever the on-disk layout changes incompatibly.
CACHE_FORMAT_VERSION = 1

MANIFEST_NAME = "manifest.json"
SHARD_DIRNAME = "shards"

LIBRISPEECH_REPO = "openslr/librispeech_asr"

#: Friendly name -> ``(dataset config, dataset split)`` for the training splits.
LIBRISPEECH_TRAIN_SPLITS: Mapping[str, tuple[str, str]] = {
    "train-clean-100": ("clean", "train.100"),
    "train-clean-360": ("clean", "train.360"),
    "train-other-500": ("other", "train.500"),
}

#: Shared parquet root, identical to the one :mod:`wal_tat.qat.evaluate` uses.
DEFAULT_LIBRISPEECH_CACHE = (
    Path(__file__).resolve().parents[3] / "cache" / "qat" / "librispeech_asr"
)

SAMPLING_RATE = 16_000
_PARQUET_COLUMNS = ("id", "audio", "text", "speaker_id")


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Utterance:
    """Everything about one utterance except its (large) feature block."""

    utterance_id: str
    text: str
    speaker_id: str
    duration_seconds: float
    labels: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "text": self.text,
            "speaker_id": self.speaker_id,
            "duration_seconds": round(self.duration_seconds, 4),
            "labels": list(self.labels),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "Utterance":
        return cls(
            utterance_id=str(payload["utterance_id"]),
            text=str(payload["text"]),
            speaker_id=str(payload["speaker_id"]),
            duration_seconds=float(payload["duration_seconds"]),
            labels=tuple(int(token) for token in payload["labels"]),
        )


@dataclass(frozen=True)
class ShardRecord:
    """One ``features_*.npy`` / ``shard_*.json`` pair."""

    index: int
    count: int
    features_file: str
    meta_file: str
    features_sha256: str
    features_bytes: int

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "count": self.count,
            "features_file": self.features_file,
            "meta_file": self.meta_file,
            "features_sha256": self.features_sha256,
            "features_bytes": self.features_bytes,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "ShardRecord":
        return cls(
            index=int(payload["index"]),
            count=int(payload["count"]),
            features_file=str(payload["features_file"]),
            meta_file=str(payload["meta_file"]),
            features_sha256=str(payload["features_sha256"]),
            features_bytes=int(payload["features_bytes"]),
        )


@dataclass(frozen=True)
class CacheSummary:
    """Human-facing size report for a materialised cache."""

    root: Path
    num_utterances: int
    num_train: int
    num_validation: int
    num_shards: int
    total_audio_seconds: float
    features_bytes: int
    on_disk_bytes: int

    @property
    def total_hours(self) -> float:
        return self.total_audio_seconds / 3600.0

    @property
    def mean_duration_seconds(self) -> float:
        if self.num_utterances == 0:
            return 0.0
        return self.total_audio_seconds / self.num_utterances

    def describe(self) -> str:
        return "\n".join(
            (
                f"cache            {self.root}",
                f"utterances       {self.num_utterances} "
                f"(train {self.num_train}, validation {self.num_validation})",
                f"audio            {self.total_hours:.2f} h "
                f"({self.total_audio_seconds:.0f} s, "
                f"mean {self.mean_duration_seconds:.2f} s/utt)",
                f"shards           {self.num_shards}",
                f"features on disk {self.features_bytes / 2**30:.2f} GiB",
                f"total on disk    {self.on_disk_bytes / 2**30:.2f} GiB",
            )
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "num_utterances": self.num_utterances,
            "num_train": self.num_train,
            "num_validation": self.num_validation,
            "num_shards": self.num_shards,
            "total_audio_seconds": round(self.total_audio_seconds, 3),
            "total_hours": round(self.total_hours, 4),
            "mean_duration_seconds": round(self.mean_duration_seconds, 4),
            "features_bytes": self.features_bytes,
            "on_disk_bytes": self.on_disk_bytes,
        }


# --------------------------------------------------------------------------- #
# Source audio
# --------------------------------------------------------------------------- #


def librispeech_parquet_shards(
    config: str,
    split: str,
    *,
    cache_dir: Optional[Path | str] = None,
    download: bool = True,
) -> list[Path]:
    """Resolve the parquet shards of one LibriSpeech split, fetching if needed."""
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_LIBRISPEECH_CACHE
    root = root.expanduser()
    local = root / config / split
    shards = sorted(local.glob("*.parquet"))
    if shards:
        return shards
    if not download:
        raise FileNotFoundError(
            f"no parquet shards under {local}; re-run with download=True"
        )
    from huggingface_hub import snapshot_download

    snapshot_download(
        LIBRISPEECH_REPO,
        repo_type="dataset",
        allow_patterns=f"{config}/{split}/*.parquet",
        local_dir=str(root),
    )
    shards = sorted(local.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"download produced no parquet shards under {local}")
    return shards


def _decode_flac(payload: Mapping[str, Any]) -> np.ndarray:
    """Decode one ``datasets``-style audio payload to mono float32 @ 16 kHz."""
    import soundfile as sf

    raw = payload.get("bytes")
    source: Any = io.BytesIO(raw) if raw is not None else str(payload["path"])
    waveform, rate = sf.read(source, dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("decoded audio must be a non-empty mono waveform")
    if int(rate) != SAMPLING_RATE:
        raise ValueError(
            f"expected {SAMPLING_RATE} Hz LibriSpeech audio, got {int(rate)} Hz"
        )
    return waveform


def _iter_source_rows(
    shards: Sequence[Path], *, limit: Optional[int], batch_size: int = 256
) -> Iterable[dict[str, Any]]:
    """Yield rows in canonical (file, row) order.

    Streams row batches rather than materialising a whole parquet file: each
    LibriSpeech shard holds roughly half a gigabyte of FLAC, and the decoded
    Python objects cost several times that again.
    """
    import pyarrow.parquet as pq

    emitted = 0
    for shard in shards:
        reader = pq.ParquetFile(shard)
        for record_batch in reader.iter_batches(
            batch_size=batch_size, columns=list(_PARQUET_COLUMNS)
        ):
            for row in record_batch.to_pylist():
                if limit is not None and emitted >= limit:
                    return
                yield row
                emitted += 1
        reader.close()


# --------------------------------------------------------------------------- #
# Text handling
# --------------------------------------------------------------------------- #


def _normalise_text(text: str, *, text_case: str, leading_space: bool) -> str:
    """Map a LibriSpeech transcript onto Whisper-like orthography.

    LibriSpeech ships unpunctuated ALL-CAPS text.  Whisper's BPE was trained on
    natural text, so caps fragment into many more tokens than the same words in
    lower case and drag the decoder away from the teacher's token distribution.
    Lower casing plus Whisper's customary leading space keeps the labels close
    to what the FP teacher emits; the WER normaliser folds case anyway.
    """
    value = " ".join(text.split())
    if text_case == "lower":
        value = value.lower()
    elif text_case == "upper":
        value = value.upper()
    elif text_case != "raw":
        raise ValueError(f"unknown text_case {text_case!r}")
    if leading_space and value:
        value = " " + value
    return value


# --------------------------------------------------------------------------- #
# Validation holdout
# --------------------------------------------------------------------------- #


def _select_validation(
    speaker_ids: Sequence[str], *, size: int, mode: str, seed: int
) -> tuple[int, ...]:
    """Pick held-out indices *from the training corpus*.

    ``"speaker"`` (default) removes whole speakers, so the holdout measures
    something closer to generalisation than a random utterance split would --
    LibriSpeech utterances from one chapter are near-duplicates of each other.
    ``"tail"`` takes the literal last ``size`` utterances.
    """
    total = len(speaker_ids)
    if size <= 0:
        return ()
    if size >= total:
        raise ValueError(
            f"validation size {size} must be smaller than the corpus ({total})"
        )
    if mode == "tail":
        return tuple(range(total - size, total))
    if mode != "speaker":
        raise ValueError(f"unknown validation mode {mode!r}")

    per_speaker: dict[str, int] = {}
    for speaker in speaker_ids:
        per_speaker[speaker] = per_speaker.get(speaker, 0) + 1
    order = sorted(per_speaker)
    random.Random(seed).shuffle(order)
    chosen: set[str] = set()
    held = 0
    for speaker in order:
        if held >= size:
            break
        chosen.add(speaker)
        held += per_speaker[speaker]
    return tuple(i for i, speaker in enumerate(speaker_ids) if speaker in chosen)


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


def _shard_paths(root: Path, index: int) -> tuple[Path, Path]:
    shards = root / SHARD_DIRNAME
    return (
        shards / f"features_{index:05d}.npy",
        shards / f"shard_{index:05d}.json",
    )


def _write_features(path: Path, block: np.ndarray) -> None:
    """Write a feature block so readers never observe a half-written shard."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, block, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_shard_meta(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_feature_cache(
    output_dir: Path | str,
    *,
    model_id: str,
    splits: Sequence[str] = ("train-clean-100",),
    dataset_cache_dir: Optional[Path | str] = None,
    download: bool = True,
    shard_size: int = 256,
    max_utterances: Optional[int] = None,
    validation_size: int = 512,
    validation_mode: str = "speaker",
    validation_seed: int = 0,
    language: str = "en",
    task: str = "transcribe",
    text_case: str = "lower",
    leading_space: bool = True,
    device: str = "cpu",
    decode_workers: int = 16,
    local_files_only: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> CacheSummary:
    """Materialise log-mel features and label ids for a Whisper QAT corpus.

    Re-running against an existing directory reuses any shard whose ``.npy`` and
    ``.json`` are both present and agree on the utterance ids, so an interrupted
    build resumes instead of restarting.

    Parameters
    ----------
    output_dir:
        Cache root.  Created if missing.
    model_id:
        Whisper checkpoint whose feature extractor and tokenizer define the
        cache.  Features are only valid for models sharing its mel filterbank.
    splits:
        Names from :data:`LIBRISPEECH_TRAIN_SPLITS`, concatenated in order.
    shard_size:
        Utterances per shard.  256 keeps each shard around 117 MiB at 80 mels.
    validation_size, validation_mode, validation_seed:
        Held-out slice taken from ``splits``.  ``dev-clean`` and ``test-clean``
        are never touched here, so they stay clean for final reporting.
    device:
        Torch device for the mel STFT (``"cuda"`` is roughly an order of
        magnitude faster than ``"cpu"`` for this workload).
    decode_workers:
        Threads used for FLAC decoding; ``libsndfile`` releases the GIL.
    """
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

    extractor = WhisperFeatureExtractor.from_pretrained(
        model_id, local_files_only=local_files_only
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        model_id,
        language=language,
        task=task,
        local_files_only=local_files_only,
    )
    n_mels = int(extractor.feature_size)
    n_frames = int(extractor.nb_max_frames)
    if int(extractor.sampling_rate) != SAMPLING_RATE:
        raise ValueError(
            f"{model_id} expects {extractor.sampling_rate} Hz audio, "
            f"LibriSpeech is {SAMPLING_RATE} Hz"
        )

    shard_files: list[Path] = []
    for name in splits:
        config, split = LIBRISPEECH_TRAIN_SPLITS[name]
        shard_files.extend(
            librispeech_parquet_shards(
                config, split, cache_dir=dataset_cache_dir, download=download
            )
        )
    emit(f"source parquet shards: {len(shard_files)}")

    records: list[ShardRecord] = []
    utterances: list[Utterance] = []
    rows: list[dict[str, Any]] = []
    total_seconds = 0.0
    features_bytes = 0

    def flush(batch: list[dict[str, Any]], index: int) -> None:
        nonlocal total_seconds, features_bytes
        features_path, meta_path = _shard_paths(root, index)
        ids = [str(row["id"]) for row in batch]

        cached = _read_shard_meta(meta_path) if features_path.exists() else None
        if cached is not None and [
            item["utterance_id"] for item in cached["items"]
        ] == ids:
            record = ShardRecord.from_json(cached["record"])
            shard_utterances = [Utterance.from_json(i) for i in cached["items"]]
        else:
            with ThreadPoolExecutor(max_workers=decode_workers) as pool:
                waveforms = list(pool.map(lambda r: _decode_flac(r["audio"]), batch))
            texts = [
                _normalise_text(
                    str(row["text"]), text_case=text_case, leading_space=leading_space
                )
                for row in batch
            ]
            encoded = extractor(
                waveforms,
                sampling_rate=SAMPLING_RATE,
                padding="max_length",
                return_tensors="pt",
                device=device,
            )["input_features"]
            block = encoded.to(torch.float16).cpu().numpy()
            if block.shape[1:] != (n_mels, n_frames):
                raise ValueError(
                    f"feature extractor returned {block.shape[1:]}, "
                    f"expected {(n_mels, n_frames)}"
                )
            _write_features(features_path, block)

            label_ids = tokenizer(texts)["input_ids"]
            shard_utterances = [
                Utterance(
                    utterance_id=ids[position],
                    text=texts[position],
                    speaker_id=str(batch[position]["speaker_id"]),
                    duration_seconds=len(waveforms[position]) / SAMPLING_RATE,
                    labels=tuple(int(token) for token in label_ids[position]),
                )
                for position in range(len(batch))
            ]
            record = ShardRecord(
                index=index,
                count=len(batch),
                features_file=f"{SHARD_DIRNAME}/{features_path.name}",
                meta_file=f"{SHARD_DIRNAME}/{meta_path.name}",
                features_sha256=sha256_file(features_path),
                features_bytes=features_path.stat().st_size,
            )
            atomic_write_json(
                meta_path,
                {
                    "record": record.to_json(),
                    "items": [item.to_json() for item in shard_utterances],
                },
            )

        records.append(record)
        utterances.extend(shard_utterances)
        features_bytes += record.features_bytes
        total_seconds += sum(item.duration_seconds for item in shard_utterances)

    for row in _iter_source_rows(shard_files, limit=max_utterances):
        rows.append(row)
        if len(rows) == shard_size:
            flush(rows, len(records))
            rows = []
            if len(records) % 10 == 0:
                emit(
                    f"shard {len(records)} | {len(utterances)} utterances | "
                    f"{total_seconds / 3600:.2f} h | "
                    f"{features_bytes / 2**30:.2f} GiB"
                )
    if rows:
        flush(rows, len(records))

    if not utterances:
        raise ValueError("no utterances were materialised")

    validation = _select_validation(
        [item.speaker_id for item in utterances],
        size=validation_size,
        mode=validation_mode,
        seed=validation_seed,
    )
    validation_set = set(validation)
    train_indices = [i for i in range(len(utterances)) if i not in validation_set]

    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "model_id": model_id,
        "source": {
            "repo": LIBRISPEECH_REPO,
            "splits": list(splits),
            "parquet_root": str(
                Path(dataset_cache_dir).expanduser()
                if dataset_cache_dir is not None
                else DEFAULT_LIBRISPEECH_CACHE
            ),
            "max_utterances": max_utterances,
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
        "split": {
            "mode": validation_mode,
            "seed": validation_seed,
            "requested_validation": validation_size,
            "num_train": len(train_indices),
            "num_validation": len(validation),
            "validation_indices": list(validation),
            "validation_speakers": sorted(
                {utterances[i].speaker_id for i in validation}
            ),
        },
        "shard_size": shard_size,
        "shards": [record.to_json() for record in records],
        "num_utterances": len(utterances),
        "total_audio_seconds": round(total_seconds, 3),
    }
    manifest_path = root / MANIFEST_NAME
    atomic_write_json(manifest_path, manifest)

    summary = CacheSummary(
        root=root,
        num_utterances=len(utterances),
        num_train=len(train_indices),
        num_validation=len(validation),
        num_shards=len(records),
        total_audio_seconds=total_seconds,
        features_bytes=features_bytes,
        on_disk_bytes=_tree_bytes(root),
    )
    emit(summary.describe())
    return summary


def _tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def load_manifest(cache_dir: Path | str) -> dict[str, Any]:
    """Load ``manifest.json`` from a cache root (or an explicit manifest path)."""
    path = Path(cache_dir).expanduser().resolve()
    if path.is_dir():
        path = path / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"no feature-cache manifest at {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    version = int(manifest.get("format_version", -1))
    if version != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"{path} has format_version {version}, "
            f"this build reads {CACHE_FORMAT_VERSION}"
        )
    return manifest


def summarize_manifest(cache_dir: Path | str) -> CacheSummary:
    """Report utterance count, hours of audio and on-disk size for a cache."""
    root = Path(cache_dir).expanduser().resolve()
    if root.is_file():
        root = root.parent
    manifest = load_manifest(root)
    features_bytes = sum(int(s["features_bytes"]) for s in manifest["shards"])
    return CacheSummary(
        root=root,
        num_utterances=int(manifest["num_utterances"]),
        num_train=int(manifest["split"]["num_train"]),
        num_validation=int(manifest["split"]["num_validation"]),
        num_shards=len(manifest["shards"]),
        total_audio_seconds=float(manifest["total_audio_seconds"]),
        features_bytes=features_bytes,
        on_disk_bytes=_tree_bytes(root),
    )


@dataclass(frozen=True)
class FeatureSample:
    """One cached utterance as returned by :class:`WhisperFeatureShards`."""

    input_features: torch.Tensor  # float16 [n_mels, n_frames]
    labels: torch.Tensor  # int64 [num_tokens]
    utterance_id: str
    text: str


class WhisperFeatureShards(Dataset):
    """Memory-mapped view over a cache built by :func:`build_feature_cache`.

    ``subset`` selects ``"train"`` (everything outside the holdout),
    ``"validation"`` (the holdout), or ``"all"``.  Shard files are opened lazily
    and per process, so the dataset is safe to hand to ``DataLoader`` workers.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        subset: str = "train",
        verify_sha256: bool = False,
    ) -> None:
        if subset not in {"train", "validation", "all"}:
            raise ValueError(f"unknown subset {subset!r}")
        root = Path(cache_dir).expanduser().resolve()
        if root.is_file():
            root = root.parent
        manifest = load_manifest(root)

        self.root = root
        self.manifest = manifest
        self.subset = subset
        self.n_mels = int(manifest["features"]["n_mels"])
        self.n_frames = int(manifest["features"]["n_frames"])
        self.model_id = str(manifest["model_id"])

        self._records = [ShardRecord.from_json(s) for s in manifest["shards"]]
        if verify_sha256:
            self._verify()

        starts: list[int] = []
        offset = 0
        for record in self._records:
            starts.append(offset)
            offset += record.count
        self._shard_starts = np.asarray(starts, dtype=np.int64)
        total = offset
        if total != int(manifest["num_utterances"]):
            raise ValueError(
                f"shard counts sum to {total} but manifest declares "
                f"{manifest['num_utterances']}"
            )

        self.utterances: list[Utterance] = []
        for record in self._records:
            payload = json.loads(
                (root / record.meta_file).read_text(encoding="utf-8")
            )
            self.utterances.extend(Utterance.from_json(i) for i in payload["items"])

        held_out = set(int(i) for i in manifest["split"]["validation_indices"])
        if subset == "all":
            self._indices = np.arange(total, dtype=np.int64)
        elif subset == "validation":
            self._indices = np.asarray(sorted(held_out), dtype=np.int64)
        else:
            self._indices = np.asarray(
                [i for i in range(total) if i not in held_out], dtype=np.int64
            )

        self._memmaps: dict[int, np.ndarray] = {}
        self._memmap_pid: Optional[int] = None

    # -- introspection ----------------------------------------------------- #

    def __len__(self) -> int:
        return int(self._indices.size)

    def __repr__(self) -> str:
        return (
            f"WhisperFeatureShards(root={self.root.name!r}, subset={self.subset!r}, "
            f"n={len(self)}, hours={self.total_hours:.2f})"
        )

    @property
    def total_hours(self) -> float:
        return sum(
            self.utterances[int(i)].duration_seconds for i in self._indices
        ) / 3600.0

    def utterance(self, position: int) -> Utterance:
        """Metadata for the ``position``-th element of this subset."""
        return self.utterances[int(self._indices[position])]

    # -- data -------------------------------------------------------------- #

    def _features(self, shard_index: int) -> np.ndarray:
        pid = os.getpid()
        if self._memmap_pid != pid:
            # Never inherit a parent's mmap handles across a fork.
            self._memmaps = {}
            self._memmap_pid = pid
        block = self._memmaps.get(shard_index)
        if block is None:
            block = np.load(
                self.root / self._records[shard_index].features_file,
                mmap_mode="r",
                allow_pickle=False,
            )
            self._memmaps[shard_index] = block
        return block

    def __getitem__(self, position: int) -> FeatureSample:
        global_index = int(self._indices[position])
        shard_index = int(
            np.searchsorted(self._shard_starts, global_index, side="right") - 1
        )
        offset = global_index - int(self._shard_starts[shard_index])
        block = self._features(shard_index)
        features = torch.from_numpy(np.array(block[offset], copy=True))
        meta = self.utterances[global_index]
        return FeatureSample(
            input_features=features,
            labels=torch.tensor(meta.labels, dtype=torch.long),
            utterance_id=meta.utterance_id,
            text=meta.text,
        )

    def _verify(self) -> None:
        for record in self._records:
            digest = sha256_file(self.root / record.features_file)
            if digest != record.features_sha256:
                raise ValueError(
                    f"{record.features_file} hashes to {digest}, "
                    f"manifest records {record.features_sha256}"
                )

    # -- wiring ------------------------------------------------------------ #

    def default_collator(self, **kwargs: Any) -> "WhisperCollator":
        """Collator preconfigured with this cache's tokenizer ids."""
        labels = self.manifest["labels"]
        return WhisperCollator(
            decoder_start_token_id=int(labels["decoder_start_token_id"]),
            **kwargs,
        )


@dataclass(frozen=True)
class WhisperCollator:
    """Batch :class:`FeatureSample` objects into Whisper model kwargs.

    ``WhisperForConditionalGeneration`` builds ``decoder_input_ids`` by shifting
    ``labels`` right and prepending ``decoder_start_token_id``.  Cached labels
    already begin with ``<|startoftranscript|>``, so that token is dropped here
    to avoid emitting it twice -- the same convention the Hugging Face Whisper
    fine-tuning recipe uses.
    """

    decoder_start_token_id: int
    label_pad_id: int = -100
    feature_dtype: torch.dtype = torch.float32
    strip_decoder_start: bool = True
    include_metadata: bool = False

    def __call__(self, samples: Sequence[FeatureSample]) -> dict[str, Any]:
        if not samples:
            raise ValueError("cannot collate an empty batch")
        features = torch.stack([s.input_features for s in samples]).to(
            self.feature_dtype
        )
        sequences = [s.labels for s in samples]
        if self.strip_decoder_start and all(
            seq.numel() > 0 and int(seq[0]) == self.decoder_start_token_id
            for seq in sequences
        ):
            sequences = [seq[1:] for seq in sequences]
        width = max(int(seq.numel()) for seq in sequences)
        labels = torch.full(
            (len(sequences), width), self.label_pad_id, dtype=torch.long
        )
        for position, seq in enumerate(sequences):
            labels[position, : seq.numel()] = seq
        batch: dict[str, Any] = {"input_features": features, "labels": labels}
        if self.include_metadata:
            batch["utterance_ids"] = [s.utterance_id for s in samples]
            batch["texts"] = [s.text for s in samples]
        return batch


def build_dataloader(
    dataset: WhisperFeatureShards,
    *,
    batch_size: int = 16,
    shuffle: bool = False,
    seed: Optional[int] = None,
    num_workers: int = 4,
    collator: Optional[WhisperCollator] = None,
    drop_last: bool = False,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = 4,
    persistent_workers: Optional[bool] = None,
) -> DataLoader:
    """Wrap a cache subset in a ``DataLoader`` yielding Whisper model kwargs.

    Pass ``seed`` with ``shuffle=True`` for a reproducible epoch order; the
    generator is advanced per epoch by ``DataLoader`` itself.
    """
    generator: Optional[torch.Generator] = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collator if collator is not None else dataset.default_collator(),
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers) if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
