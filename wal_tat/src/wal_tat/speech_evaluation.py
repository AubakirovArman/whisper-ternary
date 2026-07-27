"""Deterministic sequence-to-sequence speech metrics for WAL-TAT.

The module deliberately has no dependency on ``jiwer`` or ``evaluate``.  It
keeps per-utterance edit counts so candidate and teacher results can be
compared with paired bootstrap resampling without decoding the audio again.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


TextNormalizer = Callable[[str], str]


@dataclass(frozen=True)
class SpeechLossMetrics:
    nll: float
    perplexity: float
    predicted_tokens: int


@dataclass(frozen=True)
class WordErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.reference_words <= 0:
            raise ValueError("WER is undefined for an empty reference")
        return self.errors / self.reference_words

    def __add__(self, other: "WordErrorCounts") -> "WordErrorCounts":
        if not isinstance(other, WordErrorCounts):
            return NotImplemented
        return WordErrorCounts(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            reference_words=self.reference_words + other.reference_words,
        )


@dataclass(frozen=True)
class PairedWERBootstrap:
    point_delta: float
    lower: float
    upper: float
    confidence: float
    replicates: int
    interval: str = "two-sided"


def _identity(value: str) -> str:
    return value


def word_error_counts(
    reference: str,
    hypothesis: str,
    *,
    normalizer: Optional[TextNormalizer] = None,
) -> WordErrorCounts:
    """Return exact Levenshtein S/D/I counts for one utterance."""
    normalize = _identity if normalizer is None else normalizer
    reference_words = normalize(reference).strip().split()
    hypothesis_words = normalize(hypothesis).strip().split()
    if not reference_words:
        raise ValueError("empty references belong in the no-speech suite, not WER")

    # Each cell stores (total edits, substitutions, deletions, insertions).
    previous = [(j, 0, 0, j) for j in range(len(hypothesis_words) + 1)]
    for row, reference_word in enumerate(reference_words, start=1):
        current = [(row, 0, row, 0)]
        for column, hypothesis_word in enumerate(hypothesis_words, start=1):
            if reference_word == hypothesis_word:
                current.append(previous[column - 1])
                continue
            diagonal = previous[column - 1]
            substitution = (
                diagonal[0] + 1,
                diagonal[1] + 1,
                diagonal[2],
                diagonal[3],
            )
            above = previous[column]
            deletion = (above[0] + 1, above[1], above[2] + 1, above[3])
            left = current[column - 1]
            insertion = (left[0] + 1, left[1], left[2], left[3] + 1)
            current.append(min(substitution, deletion, insertion))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return WordErrorCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=len(reference_words),
    )


def corpus_word_error_counts(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    normalizer: Optional[TextNormalizer] = None,
) -> tuple[WordErrorCounts, tuple[WordErrorCounts, ...]]:
    """Compute micro/corpus WER and retain paired per-utterance counts."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    if not references:
        raise ValueError("at least one utterance is required")
    per_utterance = tuple(
        word_error_counts(reference, hypothesis, normalizer=normalizer)
        for reference, hypothesis in zip(references, hypotheses)
    )
    total = WordErrorCounts(0, 0, 0, 0)
    for counts in per_utterance:
        total = total + counts
    return total, per_utterance


@torch.inference_mode()
def evaluate_speech_seq2seq_loss(
    model,
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    device: Optional[torch.device | str] = None,
) -> SpeechLossMetrics:
    """Evaluate an HF-style speech encoder-decoder on frozen labelled batches."""
    total_nll = 0.0
    total_tokens = 0
    was_training = model.training
    model.eval()
    try:
        for original in batches:
            batch = {
                key: value.to(device) if device is not None else value
                for key, value in original.items()
            }
            labels = batch.get("labels")
            if labels is None:
                raise ValueError("speech evaluation batches require labels")
            token_count = int((labels != -100).sum().item())
            if token_count == 0:
                continue
            output = model(**batch)
            if hasattr(output, "logits"):
                total_nll += float(
                    F.cross_entropy(
                        output.logits.float().reshape(-1, output.logits.shape[-1]),
                        labels.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    ).item()
                )
            else:
                loss = output.loss if hasattr(output, "loss") else output[0]
                total_nll += float(loss.detach().float().item()) * token_count
            total_tokens += token_count
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("evaluation batches contain no predicted tokens")
    mean_nll = total_nll / total_tokens
    return SpeechLossMetrics(mean_nll, math.exp(mean_nll), total_tokens)


def paired_bootstrap_wer(
    baseline: Sequence[WordErrorCounts],
    candidate: Sequence[WordErrorCounts],
    *,
    replicates: int = 10_000,
    confidence: float = 0.95,
    interval: str = "two-sided",
    seed: int = 0,
    chunk_size: int = 256,
) -> PairedWERBootstrap:
    """Paired utterance bootstrap for ``candidate WER - baseline WER``.

    For long-form evaluation callers should aggregate counts per source
    document first, then pass one item per document to preserve correlation.

    ``interval="two-sided"`` returns the equal-tailed central interval.  Use
    ``interval="one-sided"`` when ``upper`` is consumed as a one-sided
    non-inferiority bound.  At confidence 0.95 these modes use the 0.975 and
    0.95 upper quantiles respectively.
    """
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired non-empty baseline/candidate counts are required")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if interval not in {"two-sided", "one-sided"}:
        raise ValueError("interval must be 'two-sided' or 'one-sided'")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    baseline_refs = torch.tensor(
        [value.reference_words for value in baseline], dtype=torch.float64
    )
    candidate_refs = torch.tensor(
        [value.reference_words for value in candidate], dtype=torch.float64
    )
    if not torch.equal(baseline_refs, candidate_refs):
        raise ValueError("paired systems must use identical reference word counts")
    baseline_errors = torch.tensor(
        [value.errors for value in baseline], dtype=torch.float64
    )
    candidate_errors = torch.tensor(
        [value.errors for value in candidate], dtype=torch.float64
    )
    denominator = baseline_refs.sum()
    point_delta = float(
        ((candidate_errors.sum() - baseline_errors.sum()) / denominator).item()
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    samples = []
    remaining = int(replicates)
    size = len(baseline)
    while remaining:
        count = min(remaining, chunk_size)
        indices = torch.randint(size, (count, size), generator=generator)
        refs = baseline_refs[indices].sum(1).clamp_min(1)
        delta = (
            candidate_errors[indices].sum(1) - baseline_errors[indices].sum(1)
        ) / refs
        samples.append(delta)
        remaining -= count
    distribution = torch.cat(samples).sort().values
    tail = (
        (1.0 - confidence) / 2.0
        if interval == "two-sided"
        else 1.0 - confidence
    )
    lower = float(torch.quantile(distribution, tail).item())
    upper = float(torch.quantile(distribution, 1.0 - tail).item())
    return PairedWERBootstrap(
        point_delta=point_delta,
        lower=lower,
        upper=upper,
        confidence=float(confidence),
        replicates=int(replicates),
        interval=interval,
    )
