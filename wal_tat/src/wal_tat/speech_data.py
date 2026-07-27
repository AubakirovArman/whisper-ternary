"""Streaming speech examples with decoder-independent raw audio handling."""
from __future__ import annotations

from dataclasses import dataclass
import io
from typing import Iterator, Mapping, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class SpeechExample:
    identifier: str
    audio: np.ndarray
    sampling_rate: int
    text: str
    document_id: Optional[str] = None
    language: Optional[str] = None


def decode_audio_payload(
    payload: Mapping[str, object], *, target_sampling_rate: int = 16_000
) -> tuple[np.ndarray, int]:
    """Decode a datasets ``Audio(decode=False)`` payload via soundfile.

    This bypasses librosa/numba and keeps the dataset layer independent from
    the system NumPy version.
    """
    import soundfile as sf

    raw_bytes = payload.get("bytes")
    path = payload.get("path")
    if raw_bytes is not None:
        source = io.BytesIO(raw_bytes)  # type: ignore[arg-type]
    elif path:
        source = str(path)
    else:
        raise ValueError("audio payload has neither bytes nor path")
    audio, sampling_rate = sf.read(source, dtype="float32", always_2d=False)
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim == 2:
        value = value.mean(axis=1)
    if value.ndim != 1 or value.size == 0:
        raise ValueError("decoded audio must be a non-empty mono waveform")
    sampling_rate = int(sampling_rate)
    if sampling_rate != target_sampling_rate:
        import torchaudio.functional as audio_functional

        tensor = torch.from_numpy(value)
        value = (
            audio_functional.resample(tensor, sampling_rate, target_sampling_rate)
            .contiguous()
            .numpy()
        )
        sampling_rate = int(target_sampling_rate)
    return value, sampling_rate


def iter_hf_speech_examples(
    dataset_name: str,
    *,
    dataset_config: Optional[str],
    split: str,
    max_samples: Optional[int] = None,
    start_offset: int = 0,
    text_column: str = "text",
    id_column: str = "id",
    audio_column: str = "audio",
    document_column: Optional[str] = None,
    language: Optional[str] = None,
    streaming: bool = True,
    target_sampling_rate: int = 16_000,
) -> Iterator[SpeechExample]:
    """Yield deterministic dataset-order examples with raw-byte audio decode."""
    from datasets import Audio, load_dataset

    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    if start_offset < 0:
        raise ValueError("start_offset must be non-negative")
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )
    dataset = dataset.cast_column(audio_column, Audio(decode=False))
    if start_offset:
        # Skip at the Arrow/streaming layer so discarded utterances are never
        # decoded by soundfile. Large deterministic experiment windows otherwise
        # spend minutes decoding audio that is immediately thrown away.
        dataset = dataset.skip(start_offset)
    for index, row in enumerate(dataset):
        if max_samples is not None and index >= max_samples:
            break
        waveform, sampling_rate = decode_audio_payload(
            row[audio_column], target_sampling_rate=target_sampling_rate
        )
        identifier = str(row.get(id_column, index + start_offset))
        document_id = (
            None if document_column is None else str(row.get(document_column, identifier))
        )
        yield SpeechExample(
            identifier=identifier,
            audio=waveform,
            sampling_rate=sampling_rate,
            text=str(row[text_column]),
            document_id=document_id,
            language=language,
        )
