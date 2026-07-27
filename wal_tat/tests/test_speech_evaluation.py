import pytest
import torch

from wal_tat import (
    corpus_word_error_counts,
    evaluate_speech_seq2seq_loss,
    paired_bootstrap_wer,
    word_error_counts,
)


def test_word_error_counts_tracks_substitution_deletion_and_insertion():
    counts = word_error_counts("one two three", "one too extra")
    assert counts.substitutions == 2
    assert counts.deletions == 0
    assert counts.insertions == 0
    assert counts.reference_words == 3
    assert counts.wer == pytest.approx(2 / 3)

    counts = word_error_counts("one two three", "one three extra")
    assert counts.errors == 2
    assert counts.reference_words == 3


def test_corpus_wer_is_micro_aggregated():
    total, per_utterance = corpus_word_error_counts(
        ["one", "one two three"], ["bad", "one two three"]
    )
    assert len(per_utterance) == 2
    assert total.errors == 1
    assert total.reference_words == 4
    assert total.wer == pytest.approx(0.25)


def test_empty_reference_is_rejected_from_wer():
    with pytest.raises(ValueError, match="no-speech"):
        word_error_counts("", "hallucination")


def test_paired_bootstrap_uses_same_references():
    baseline = [
        word_error_counts("one two", "one two"),
        word_error_counts("three four", "three wrong"),
    ]
    candidate = [
        word_error_counts("one two", "one wrong"),
        word_error_counts("three four", "three wrong"),
    ]
    result = paired_bootstrap_wer(
        baseline, candidate, replicates=200, seed=7, chunk_size=17
    )
    assert result.point_delta == pytest.approx(0.25)
    assert result.lower <= result.point_delta <= result.upper
    assert result.interval == "two-sided"


def test_one_sided_bootstrap_uses_confidence_quantile_for_upper_bound():
    baseline = [
        word_error_counts("one two three four", "one two three four"),
        word_error_counts("five six seven eight", "five six seven wrong"),
        word_error_counts("nine ten eleven twelve", "nine ten wrong wrong"),
    ]
    candidate = [
        word_error_counts("one two three four", "one two three wrong"),
        word_error_counts("five six seven eight", "five six wrong wrong"),
        word_error_counts("nine ten eleven twelve", "nine wrong wrong wrong"),
    ]
    central = paired_bootstrap_wer(
        baseline,
        candidate,
        replicates=2_000,
        confidence=0.95,
        interval="two-sided",
        seed=19,
    )
    one_sided = paired_bootstrap_wer(
        baseline,
        candidate,
        replicates=2_000,
        confidence=0.95,
        interval="one-sided",
        seed=19,
    )
    assert one_sided.interval == "one-sided"
    assert one_sided.point_delta == central.point_delta
    assert one_sided.upper <= central.upper


def test_bootstrap_rejects_unknown_interval():
    counts = [word_error_counts("one", "one")]
    with pytest.raises(ValueError, match="interval"):
        paired_bootstrap_wer(counts, counts, interval="upper-ish")


class _SpeechOutput:
    def __init__(self, loss):
        self.loss = loss


class _SpeechModel(torch.nn.Module):
    def forward(self, input_features, labels):
        del input_features
        return _SpeechOutput((labels != -100).float().mean())


def test_seq2seq_loss_counts_unshifted_label_tokens():
    model = _SpeechModel()
    batches = [
        {
            "input_features": torch.ones(2, 80, 4),
            "labels": torch.tensor([[1, 2, -100], [3, 4, 5]]),
        }
    ]
    result = evaluate_speech_seq2seq_loss(model, batches)
    assert result.predicted_tokens == 5
    assert result.nll == pytest.approx(5 / 6)
