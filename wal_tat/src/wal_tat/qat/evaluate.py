"""Honest, reusable word-error-rate evaluation for Whisper QAT experiments.

Every quality number quoted by a QAT experiment must come from this module, so
the measurement conventions are fixed here once and recorded in the payload the
functions return:

Decoding
    Greedy only (``num_beams=1``, ``do_sample=False``).  Beam search is not an
    option: it hides quantisation damage behind extra search.

Normalisation
    :class:`~transformers.models.whisper.english_normalizer.EnglishTextNormalizer`
    seeded with the Whisper tokenizer's ``english_spelling_normalizer``.  This is
    the standard Whisper/OpenLLM English ASR convention.

WER
    Corpus ("micro") WER: ``(S + D + I) / N`` summed over the whole split, never
    the mean of per-utterance WERs.  Exact Levenshtein S/D/I counts are kept per
    utterance so a candidate can be compared against a cached baseline without
    re-decoding either system.

Sub-sampling
    ``n`` always takes the **first** ``n`` utterances in dataset order.  There is
    no shuffling and no seed: two runs with the same ``n`` see the same audio,
    and ``sample_ids_sha256`` in the payload proves it.

Uncertainty
    :func:`paired_bootstrap` resamples utterances with replacement, using the
    same indices for both systems.  The interval convention (one-sided bound vs.
    two-sided central interval, and the exact quantiles used) is written into the
    returned dictionary rather than left to the reader's imagination.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from ..speech_data import SpeechExample, decode_audio_payload
from ..speech_evaluation import (
    TextNormalizer,
    WordErrorCounts,
    corpus_word_error_counts,
    paired_bootstrap_wer,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_NEW_TOKENS",
    "LIBRISPEECH_REPO",
    "LIBRISPEECH_SPLITS",
    "SpeechDatasetSplit",
    "UtteranceResult",
    "build_english_normalizer",
    "evaluate_wer",
    "load_librispeech",
    "paired_bootstrap",
    "resolve_dataset_split",
]


DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_SAMPLING_RATE = 16_000

LIBRISPEECH_REPO = "openslr/librispeech_asr"

#: Friendly split name -> ``(dataset config, dataset split)`` in the HF repo.
LIBRISPEECH_SPLITS: Mapping[str, tuple[str, str]] = {
    "dev-clean": ("clean", "validation"),
    "dev-other": ("other", "validation"),
    "test-clean": ("clean", "test"),
    "test-other": ("other", "test"),
}

#: Default on-disk home for downloaded LibriSpeech parquet shards.
DEFAULT_LIBRISPEECH_CACHE = (
    Path(__file__).resolve().parents[3] / "cache" / "qat" / "librispeech_asr"
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def _sha256_ids(identifiers: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(identifiers).encode()).hexdigest()


@dataclass(frozen=True)
class SpeechDatasetSplit:
    """A materialised, immutably ordered evaluation split.

    ``examples`` are held in dataset order.  Slicing with :meth:`head` keeps the
    prefix, which is the only sub-sampling policy this harness supports.
    """

    name: str
    examples: tuple[SpeechExample, ...]
    source: str = ""

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("a dataset split must contain at least one example")

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    @property
    def sample_ids_sha256(self) -> str:
        """Fingerprint of the exact utterance list, in order."""
        return _sha256_ids(value.identifier for value in self.examples)

    @property
    def audio_seconds(self) -> float:
        return sum(
            len(value.audio) / value.sampling_rate for value in self.examples
        )

    def head(self, n: Optional[int]) -> "SpeechDatasetSplit":
        """Return the first ``n`` utterances (all of them when ``n`` is None)."""
        if n is None:
            return self
        if n < 1:
            raise ValueError("n must be positive")
        if n > len(self.examples):
            raise ValueError(
                f"split {self.name!r} has {len(self.examples)} utterances, "
                f"requested {n}"
            )
        if n == len(self.examples):
            return self
        return SpeechDatasetSplit(
            name=f"{self.name}[:{n}]",
            examples=self.examples[:n],
            source=self.source,
        )


def _librispeech_shards(
    config: str,
    split: str,
    *,
    cache_dir: Path,
    download: bool,
) -> list[Path]:
    local = cache_dir / config / split
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
        local_dir=str(cache_dir),
    )
    shards = sorted(local.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"download produced no parquet shards under {local}")
    return shards


def load_librispeech(
    split: str = "dev-clean",
    *,
    n: Optional[int] = None,
    cache_dir: Optional[Path | str] = None,
    download: bool = True,
    language: str = "en",
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
) -> SpeechDatasetSplit:
    """Materialise a LibriSpeech split from local parquet shards.

    The shards are the canonical ``openslr/librispeech_asr`` files, read in file
    and row order, so the utterance sequence matches the streaming
    ``datasets`` reader used elsewhere in this repository byte for byte.

    Parameters
    ----------
    split:
        One of :data:`LIBRISPEECH_SPLITS` (``"dev-clean"``, ``"test-clean"``, ...).
    n:
        Keep only the first ``n`` utterances.  ``None`` keeps the whole split.
    cache_dir:
        Where parquet shards live.  Defaults to ``wal_tat/cache/qat/librispeech_asr``.
    download:
        Fetch missing shards from the Hub.  Set to ``False`` for offline runs.
    """
    import pyarrow.parquet as pq

    if split not in LIBRISPEECH_SPLITS:
        raise ValueError(
            f"unknown split {split!r}; expected one of {sorted(LIBRISPEECH_SPLITS)}"
        )
    config, hf_split = LIBRISPEECH_SPLITS[split]
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_LIBRISPEECH_CACHE
    shards = _librispeech_shards(
        config, hf_split, cache_dir=root.expanduser(), download=download
    )

    examples: list[SpeechExample] = []
    for shard in shards:
        table = pq.read_table(shard, columns=["id", "audio", "text", "speaker_id"])
        for row in table.to_pylist():
            if n is not None and len(examples) >= n:
                break
            waveform, rate = decode_audio_payload(
                row["audio"], target_sampling_rate=sampling_rate
            )
            examples.append(
                SpeechExample(
                    identifier=str(row["id"]),
                    audio=waveform,
                    sampling_rate=rate,
                    text=str(row["text"]),
                    document_id=str(row["speaker_id"]),
                    language=language,
                )
            )
        del table
        if n is not None and len(examples) >= n:
            break
    if n is not None and len(examples) < n:
        raise ValueError(
            f"split {split!r} holds {len(examples)} utterances, requested {n}"
        )
    return SpeechDatasetSplit(
        name=f"librispeech/{config}/{hf_split}" + ("" if n is None else f"[:{n}]"),
        examples=tuple(examples),
        source=str(shards[0].parent),
    )


def resolve_dataset_split(dataset_split: Any) -> SpeechDatasetSplit:
    """Coerce the many convenient spellings of "a split" into one type.

    Accepts a :class:`SpeechDatasetSplit`, any sequence of
    :class:`~wal_tat.speech_data.SpeechExample`, or a LibriSpeech split name
    such as ``"dev-clean"`` (which is loaded from the local parquet cache).
    """
    if isinstance(dataset_split, SpeechDatasetSplit):
        return dataset_split
    if isinstance(dataset_split, str):
        return load_librispeech(dataset_split)
    examples = tuple(dataset_split)
    if not examples or not isinstance(examples[0], SpeechExample):
        raise TypeError(
            "dataset_split must be a SpeechDatasetSplit, a sequence of "
            "SpeechExample, or a LibriSpeech split name"
        )
    return SpeechDatasetSplit(name="custom", examples=examples)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def build_english_normalizer(processor) -> TextNormalizer:
    """The Whisper English normaliser, seeded from the tokenizer's spelling map."""
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    spelling = getattr(processor.tokenizer, "english_spelling_normalizer", None)
    if not spelling:
        raise ValueError(
            "the tokenizer carries no english_spelling_normalizer; the WER "
            "numbers would not be comparable with published Whisper results"
        )
    return EnglishTextNormalizer(spelling)


@dataclass(frozen=True)
class UtteranceResult:
    """One decoded utterance plus its exact edit counts."""

    index: int
    identifier: str
    document_id: Optional[str]
    duration_seconds: float
    reference: str
    hypothesis: str
    normalized_reference: str
    normalized_hypothesis: str
    S: int
    D: int
    I: int
    N: int

    @property
    def counts(self) -> WordErrorCounts:
        return WordErrorCounts(
            substitutions=self.S,
            deletions=self.D,
            insertions=self.I,
            reference_words=self.N,
        )

    @property
    def errors(self) -> int:
        return self.S + self.D + self.I

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "id": self.identifier,
            "document_id": self.document_id,
            "duration_seconds": self.duration_seconds,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
            "normalized_reference": self.normalized_reference,
            "normalized_hypothesis": self.normalized_hypothesis,
            "S": self.S,
            "D": self.D,
            "I": self.I,
            "N": self.N,
        }


def _model_device_and_dtype(model) -> tuple[torch.device, torch.dtype]:
    for parameter in model.parameters():
        return parameter.device, parameter.dtype
    for buffer in model.buffers():
        return buffer.device, buffer.dtype
    raise ValueError("model has no parameters or buffers to infer device/dtype from")


def _teacher_forced_labels(processor, texts: Sequence[str]) -> torch.Tensor:
    """Tokenise references for teacher-forced NLL, dropping the BOS marker.

    The tokenizer prepends ``<|startoftranscript|>``, which the model also
    supplies as the decoder start token; keeping it would score a token the
    model never has to predict.
    """
    encoded = processor.tokenizer(list(texts), padding=True, return_tensors="pt")
    labels = encoded.input_ids.masked_fill(encoded.attention_mask.ne(1), -100)
    start_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    if labels.shape[1] and torch.all(labels[:, 0].eq(start_id)):
        labels = labels[:, 1:]
    return labels.contiguous()


def evaluate_wer(
    model,
    processor,
    dataset_split: Any = "dev-clean",
    *,
    n: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    language: str = "en",
    task: str = "transcribe",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    normalizer: Optional[TextNormalizer] = None,
    compute_nll: bool = False,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
    progress: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Greedy-decode ``dataset_split`` and return corpus WER plus edit counts.

    The model is used exactly as handed over: it is neither moved nor cast
    unless ``device``/``dtype`` are given, which keeps a mid-training QAT model
    on its own device with its own fake-quant state intact.  ``model.eval()`` is
    applied for the duration and the previous training flag is restored.

    Returns
    -------
    dict
        ``wer`` (fraction, not percent), ``S``, ``D``, ``I``, ``N``,
        ``per_utterance`` (list of dicts, dataset order), and the decode /
        dataset metadata needed to reproduce the run.  ``nll`` and
        ``perplexity`` are present only when ``compute_nll=True``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    split = resolve_dataset_split(dataset_split).head(n)
    examples = split.examples

    processor.tokenizer.set_prefix_tokens(language=language, task=task)
    normalize = normalizer if normalizer is not None else build_english_normalizer(
        processor
    )

    model_device, model_dtype = _model_device_and_dtype(model)
    device = torch.device(device) if device is not None else model_device
    dtype = dtype if dtype is not None else model_dtype

    was_training = model.training
    model.eval()
    hypotheses: list[str] = []
    total_nll = 0.0
    total_tokens = 0
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, len(examples), batch_size):
                batch = examples[start : start + batch_size]
                features = processor.feature_extractor(
                    [value.audio for value in batch],
                    sampling_rate=DEFAULT_SAMPLING_RATE,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                input_features = features.input_features.to(
                    device=device, dtype=dtype
                )
                attention_mask = features.attention_mask.to(device=device)
                if compute_nll:
                    labels = _teacher_forced_labels(
                        processor, [value.text for value in batch]
                    ).to(device)
                    logits = model(
                        input_features=input_features,
                        attention_mask=attention_mask,
                        labels=labels,
                    ).logits
                    total_nll += float(
                        torch.nn.functional.cross_entropy(
                            logits.float().reshape(-1, logits.shape[-1]),
                            labels.reshape(-1),
                            ignore_index=-100,
                            reduction="sum",
                        ).item()
                    )
                    total_tokens += int((labels != -100).sum().item())
                generated = model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    language=language,
                    task=task,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                )
                hypotheses.extend(
                    processor.batch_decode(generated, skip_special_tokens=True)
                )
                if progress is not None:
                    progress(
                        {
                            "decoded": len(hypotheses),
                            "total": len(examples),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )
    finally:
        model.train(was_training)
    elapsed = time.perf_counter() - started

    references = [value.text for value in examples]
    total, per_counts = corpus_word_error_counts(
        references, hypotheses, normalizer=normalize
    )
    per_utterance = [
        UtteranceResult(
            index=index,
            identifier=example.identifier,
            document_id=example.document_id,
            duration_seconds=len(example.audio) / example.sampling_rate,
            reference=example.text,
            hypothesis=hypothesis,
            normalized_reference=normalize(example.text),
            normalized_hypothesis=normalize(hypothesis),
            S=counts.substitutions,
            D=counts.deletions,
            I=counts.insertions,
            N=counts.reference_words,
        ).as_dict()
        for index, (example, hypothesis, counts) in enumerate(
            zip(examples, hypotheses, per_counts)
        )
    ]

    payload: dict[str, Any] = {
        "wer": total.wer,
        "S": total.substitutions,
        "D": total.deletions,
        "I": total.insertions,
        "N": total.reference_words,
        "per_utterance": per_utterance,
        "samples": len(examples),
        "dataset": split.name,
        "dataset_source": split.source,
        "sample_ids_sha256": split.sample_ids_sha256,
        "audio_seconds": split.audio_seconds,
        "decode": {
            "num_beams": 1,
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "language": language,
            "task": task,
            "batch_size": batch_size,
            "dtype": str(dtype).removeprefix("torch."),
            "device": str(device),
        },
        "normalizer": "whisper-english",
        "wer_definition": "corpus/micro: (S + D + I) / N summed over the split",
        "elapsed_seconds": elapsed,
    }
    if compute_nll:
        if total_tokens == 0:
            raise ValueError("teacher-forced pass scored no tokens")
        mean_nll = total_nll / total_tokens
        payload["nll"] = mean_nll
        payload["perplexity"] = float(np.exp(mean_nll))
        payload["predicted_tokens"] = total_tokens
    return payload


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #

_COUNT_ALIASES = {
    "S": ("S", "substitutions"),
    "D": ("D", "deletions"),
    "I": ("I", "insertions"),
    "N": ("N", "reference_words"),
}


def _as_counts(item: Any) -> WordErrorCounts:
    if isinstance(item, WordErrorCounts):
        return item
    if isinstance(item, UtteranceResult):
        return item.counts
    if isinstance(item, Mapping):
        values = {}
        for canonical, aliases in _COUNT_ALIASES.items():
            for alias in aliases:
                if alias in item:
                    values[canonical] = int(item[alias])
                    break
            else:
                raise KeyError(
                    f"per-utterance record is missing {canonical!r} "
                    f"(tried {aliases})"
                )
        return WordErrorCounts(
            substitutions=values["S"],
            deletions=values["D"],
            insertions=values["I"],
            reference_words=values["N"],
        )
    raise TypeError(f"cannot read edit counts from {type(item).__name__}")


def _identifier(item: Any) -> Optional[str]:
    if isinstance(item, UtteranceResult):
        return item.identifier
    if isinstance(item, Mapping):
        value = item.get("id")
        return None if value is None else str(value)
    return None


def _as_per_utterance(value: Any) -> list[Any]:
    """Accept either a raw per-utterance list or a full ``evaluate_wer`` payload."""
    if isinstance(value, Mapping) and "per_utterance" in value:
        return list(value["per_utterance"])
    return list(value)


def paired_bootstrap(
    base_per_utt: Any,
    cand_per_utt: Any,
    *,
    one_sided: bool = True,
    confidence: float = 0.95,
    replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired utterance bootstrap for ``candidate WER - baseline WER``.

    Both arguments may be the ``per_utterance`` list from :func:`evaluate_wer`
    or the whole payload; the two systems must have decoded the same utterances
    in the same order.  Each replicate resamples utterance *indices* with
    replacement and applies the same indices to both systems, so the correlation
    between the systems is preserved and the interval is about the difference,
    not about either system alone.

    Interval convention (also written into the result under ``"convention"``):

    ``one_sided=True`` (default)
        ``upper`` is the ``confidence`` quantile of the replicate deltas, i.e. a
        one-sided upper bound: ``P(delta <= upper) ~= confidence``.  ``lower`` is
        the symmetric one-sided *lower* bound at the ``1 - confidence`` quantile.
        Each bound is individually valid at ``confidence``; **together they span
        only ``2 * confidence - 1``** (90% at ``confidence=0.95``), so do not
        quote the pair as a 95% interval.

    ``one_sided=False``
        Equal-tailed central interval: quantiles ``(1 - confidence) / 2`` and
        ``1 - (1 - confidence) / 2`` (0.025 and 0.975 at ``confidence=0.95``).

    Returns
    -------
    dict
        ``delta_wer`` (point estimate), ``lower``, ``upper``, ``convention``,
        the exact quantiles used, and the two corpus WERs.  All WER values are
        fractions, not percentages.
    """
    baseline_items = _as_per_utterance(base_per_utt)
    candidate_items = _as_per_utterance(cand_per_utt)
    if len(baseline_items) != len(candidate_items):
        raise ValueError(
            "paired bootstrap needs one record per system per utterance: "
            f"{len(baseline_items)} baseline vs {len(candidate_items)} candidate"
        )
    for index, (left, right) in enumerate(zip(baseline_items, candidate_items)):
        left_id, right_id = _identifier(left), _identifier(right)
        if left_id is not None and right_id is not None and left_id != right_id:
            raise ValueError(
                f"utterance {index} differs between systems: "
                f"{left_id!r} vs {right_id!r}"
            )
    baseline = [_as_counts(value) for value in baseline_items]
    candidate = [_as_counts(value) for value in candidate_items]

    interval = "one-sided" if one_sided else "two-sided"
    result = paired_bootstrap_wer(
        baseline,
        candidate,
        replicates=replicates,
        confidence=confidence,
        interval=interval,
        seed=seed,
    )
    tail = (1.0 - confidence) if one_sided else (1.0 - confidence) / 2.0
    baseline_total = sum(baseline, WordErrorCounts(0, 0, 0, 0))
    candidate_total = sum(candidate, WordErrorCounts(0, 0, 0, 0))
    if one_sided:
        convention = (
            f"one-sided: 'upper' is the {confidence:.0%} upper bound "
            f"(quantile {1.0 - tail:g}) and 'lower' is the {confidence:.0%} lower "
            f"bound (quantile {tail:g}); the pair spans only "
            f"{2 * confidence - 1:.0%}, so do not quote it as a "
            f"{confidence:.0%} interval"
        )
    else:
        convention = (
            f"two-sided equal-tailed {confidence:.0%} central interval "
            f"(quantiles {tail:g} and {1.0 - tail:g})"
        )
    return {
        "delta_wer": result.point_delta,
        "lower": result.lower,
        "upper": result.upper,
        "convention": convention,
        "interval": interval,
        "one_sided": bool(one_sided),
        "confidence": float(confidence),
        "lower_quantile": float(tail),
        "upper_quantile": float(1.0 - tail),
        "replicates": int(replicates),
        "seed": int(seed),
        "utterances": len(baseline),
        "baseline_wer": baseline_total.wer,
        "candidate_wer": candidate_total.wer,
        "relative_wer": candidate_total.wer / baseline_total.wer,
    }
