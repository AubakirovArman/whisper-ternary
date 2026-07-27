"""Contract tests for the QAT evaluation harness.

These cover the parts that must never drift silently: the corpus-WER
definition, the paired-bootstrap interval convention, and the coercion helpers
that let a cached JSON baseline be compared against a live candidate.
"""

from __future__ import annotations

import pytest
import torch

from wal_tat.qat.evaluate import (
    _as_counts,
    _as_per_utterance,
    paired_bootstrap,
)
from wal_tat.speech_evaluation import WordErrorCounts


def _record(identifier: str, s: int, d: int, i: int, n: int) -> dict:
    return {"id": identifier, "S": s, "D": d, "I": i, "N": n}


def _pair(count: int, *, extra_errors: int = 0):
    baseline = [_record(f"u{index}", 1, 0, 0, 10) for index in range(count)]
    candidate = [
        _record(f"u{index}", 1 + (1 if index < extra_errors else 0), 0, 0, 10)
        for index in range(count)
    ]
    return baseline, candidate


def test_as_counts_accepts_short_and_long_key_spellings() -> None:
    short = _as_counts({"S": 1, "D": 2, "I": 3, "N": 4})
    long = _as_counts(
        {
            "substitutions": 1,
            "deletions": 2,
            "insertions": 3,
            "reference_words": 4,
        }
    )
    assert short == long == WordErrorCounts(1, 2, 3, 4)


def test_as_counts_rejects_incomplete_records() -> None:
    with pytest.raises(KeyError):
        _as_counts({"S": 1, "D": 2, "I": 3})


def test_as_per_utterance_unwraps_a_full_payload() -> None:
    records = [_record("u0", 1, 0, 0, 10)]
    assert _as_per_utterance({"wer": 0.1, "per_utterance": records}) == records
    assert _as_per_utterance(records) == records


def test_paired_bootstrap_reports_micro_wer_and_point_delta() -> None:
    baseline, candidate = _pair(64, extra_errors=32)
    result = paired_bootstrap(baseline, candidate, replicates=512)
    # 64 utterances x 10 reference words; baseline 64 errors, candidate 96.
    assert result["baseline_wer"] == pytest.approx(64 / 640)
    assert result["candidate_wer"] == pytest.approx(96 / 640)
    assert result["delta_wer"] == pytest.approx(32 / 640)
    assert result["relative_wer"] == pytest.approx(96 / 64)
    assert result["utterances"] == 64


def test_one_sided_is_the_default_and_states_its_quantiles() -> None:
    baseline, candidate = _pair(32, extra_errors=8)
    result = paired_bootstrap(baseline, candidate, replicates=1024)
    assert result["one_sided"] is True
    assert result["interval"] == "one-sided"
    assert result["confidence"] == 0.95
    assert result["upper_quantile"] == pytest.approx(0.95)
    assert result["lower_quantile"] == pytest.approx(0.05)
    assert "one-sided" in result["convention"]
    # The old dossier quoted a 0.975 quantile as a "one-sided 95%" bound. The
    # convention string must make the actual quantile unambiguous.
    assert "0.95" in result["convention"] or "95%" in result["convention"]


def test_two_sided_uses_the_wider_central_interval() -> None:
    baseline, candidate = _pair(48, extra_errors=12)
    one = paired_bootstrap(baseline, candidate, replicates=4096, seed=3)
    two = paired_bootstrap(
        baseline, candidate, one_sided=False, replicates=4096, seed=3
    )
    assert two["upper_quantile"] == pytest.approx(0.975)
    assert two["lower_quantile"] == pytest.approx(0.025)
    assert two["upper"] >= one["upper"]
    assert two["lower"] <= one["lower"]
    assert "two-sided" in two["convention"]


def test_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    baseline, candidate = _pair(40, extra_errors=10)
    first = paired_bootstrap(baseline, candidate, replicates=2048, seed=11)
    second = paired_bootstrap(baseline, candidate, replicates=2048, seed=11)
    assert first["lower"] == second["lower"]
    assert first["upper"] == second["upper"]


def test_identical_systems_bracket_zero() -> None:
    baseline, _ = _pair(50)
    result = paired_bootstrap(baseline, baseline, replicates=1024)
    assert result["delta_wer"] == 0.0
    assert result["lower"] == 0.0
    assert result["upper"] == 0.0


def test_mismatched_utterance_ids_are_rejected() -> None:
    baseline, candidate = _pair(8)
    candidate[3] = dict(candidate[3], id="other")
    with pytest.raises(ValueError, match="differs between systems"):
        paired_bootstrap(baseline, candidate, replicates=16)


def test_mismatched_lengths_are_rejected() -> None:
    baseline, candidate = _pair(8)
    with pytest.raises(ValueError, match="one record per system"):
        paired_bootstrap(baseline, candidate[:-1], replicates=16)


def test_split_head_keeps_the_prefix_and_rejects_oversized_requests() -> None:
    import numpy as np

    from wal_tat.qat.evaluate import SpeechDatasetSplit
    from wal_tat.speech_data import SpeechExample

    examples = tuple(
        SpeechExample(
            identifier=f"u{index}",
            audio=np.zeros(16, dtype=np.float32),
            sampling_rate=16_000,
            text="hello",
        )
        for index in range(5)
    )
    split = SpeechDatasetSplit(name="unit", examples=examples)
    head = split.head(3)
    assert [value.identifier for value in head] == ["u0", "u1", "u2"]
    assert split.head(None) is split
    assert head.sample_ids_sha256 != split.sample_ids_sha256
    with pytest.raises(ValueError):
        split.head(9)


def test_torch_is_importable_for_the_bootstrap_backend() -> None:
    # paired_bootstrap_wer resamples with a torch generator; guard the import so
    # a missing backend fails here rather than mid-campaign.
    assert torch.randint(4, (2, 2)).shape == (2, 2)
