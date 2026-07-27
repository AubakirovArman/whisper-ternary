import hashlib
import json

import pytest

from whisper_parallel_full_audit import _comparison, _result_matches


def _entry(identifier: str, *, errors: int, words: int = 10) -> dict:
    return {
        "id": identifier,
        "substitutions": errors,
        "deletions": 0,
        "insertions": 0,
        "reference_words": words,
    }


def test_comparison_reuses_full_baseline_for_deterministic_prefix() -> None:
    baseline = {
        "metrics": {"wer": 0.1, "nll": 0.5},
        "utterances": [
            _entry("a", errors=1),
            _entry("b", errors=2),
            _entry("c", errors=0),
        ],
    }
    candidate = {
        "metrics": {"wer": 0.2, "nll": 0.7},
        "utterances": [
            _entry("a", errors=2),
            _entry("b", errors=2),
        ],
    }

    result = _comparison(
        baseline,
        candidate,
        replicates=100,
        confidence=0.95,
        seed=3,
    )

    assert result["baseline_scope"] == "superset-subset"
    assert result["baseline_wer"] == 0.15
    assert result["candidate_wer"] == 0.20
    assert result["candidate_minus_baseline"]["interval"] == "one-sided"
    assert result["nll_delta"] is None


def test_comparison_keeps_nll_delta_for_identical_scope() -> None:
    baseline = {
        "metrics": {"wer": 0.1, "nll": 0.5},
        "utterances": [_entry("a", errors=1)],
    }
    candidate = {
        "metrics": {"wer": 0.2, "nll": 0.7},
        "utterances": [_entry("a", errors=2)],
    }

    result = _comparison(
        baseline,
        candidate,
        replicates=50,
        confidence=0.95,
        seed=5,
    )

    assert result["baseline_scope"] == "exact"
    assert result["nll_delta"] == pytest.approx(0.2)


def test_audit_reuse_requires_source_and_request_sha(tmp_path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "source_checkpoint": str(checkpoint.resolve()),
                "source_checkpoint_sha256": checkpoint_sha256,
                "evaluation_request_sha256": "a" * 64,
            }
        )
    )

    assert _result_matches(
        result,
        checkpoint.resolve(),
        checkpoint_sha256=checkpoint_sha256,
        request_sha256="a" * 64,
    )
    assert not _result_matches(
        result,
        checkpoint.resolve(),
        checkpoint_sha256=checkpoint_sha256,
        request_sha256="b" * 64,
    )
