from __future__ import annotations

import pytest

from wal_tat.whisper_fastlane_policy import (
    AuthoritativeFrontiers,
    FrontierRecord,
    GateEvidence,
    GateThresholds,
    NonDominatingFrontierError,
    QualityTier,
    StaleParentError,
    WhisperFastLaneGate,
    initial_packages,
    split_failed_package,
    working_budget_envelope,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def test_working_gate_defaults_to_explicit_relative_wer_115() -> None:
    gate = WhisperFastLaneGate()

    boundary = gate.evaluate_working(GateEvidence(0.04, 0.046))
    over = gate.evaluate_working(GateEvidence(0.04, 0.0460001))

    assert boundary.tier is QualityTier.WORKING
    assert boundary.metric == "candidate_wer/baseline_wer"
    assert boundary.observed == pytest.approx(1.15)
    assert boundary.limit == 1.15
    assert boundary.passed
    assert not over.passed


def test_strict_gate_is_separate_from_working_relative_gate() -> None:
    gate = WhisperFastLaneGate()
    evidence = GateEvidence(
        baseline_wer=0.04,
        candidate_wer=0.045,
        upper_absolute_delta_wer=0.0051,
    )

    assert gate.evaluate_working(evidence).passed
    strict = gate.evaluate_strict(evidence)
    assert strict.tier is QualityTier.STRICT
    assert strict.metric == "upper_absolute_delta_wer"
    assert strict.limit == 0.005
    assert not strict.passed


def test_strict_gate_requires_paired_confidence_evidence() -> None:
    with pytest.raises(ValueError, match="requires upper_absolute_delta_wer"):
        WhisperFastLaneGate().evaluate_strict(GateEvidence(0.04, 0.041))


def test_gate_thresholds_can_be_configured_independently() -> None:
    gate = WhisperFastLaneGate(
        GateThresholds(
            working_max_relative_wer=1.10,
            strict_max_absolute_delta_wer=0.002,
        )
    )
    evidence = GateEvidence(0.04, 0.044, upper_absolute_delta_wer=0.0019)

    assert gate.evaluate_working(evidence).passed
    assert gate.evaluate_strict(evidence).passed


@pytest.mark.parametrize(
    ("coverage", "expected"),
    [(0.0, 0.0125), (0.5, 0.00875), (1.0, 0.005)],
)
def test_working_budget_envelope_tightens_with_coverage(
    coverage: float, expected: float
) -> None:
    assert working_budget_envelope(coverage) == pytest.approx(expected)


def test_initial_packages_use_4_then_2_then_1_without_trailing_three() -> None:
    matrix_ids = [f"m{index}" for index in range(11)]

    packages = initial_packages(matrix_ids)

    assert [len(package) for package in packages] == [4, 4, 2, 1]
    assert [item for package in packages for item in package] == matrix_ids


def test_failed_packages_split_4_to_2_to_1() -> None:
    first = split_failed_package(("a", "b", "c", "d"))
    assert first == (("a", "b"), ("c", "d"))

    second = tuple(child for package in first for child in split_failed_package(package))
    assert second == (("a",), ("b",), ("c",), ("d",))

    assert split_failed_package(("a",)) == ()


def _decision(
    *,
    tier: QualityTier = QualityTier.WORKING,
    candidate_wer: float = 0.044,
):
    gate = WhisperFastLaneGate()
    evidence = GateEvidence(
        baseline_wer=0.04,
        candidate_wer=candidate_wer,
        upper_absolute_delta_wer=0.004,
    )
    if tier is QualityTier.STRICT:
        return gate.evaluate_strict(evidence)
    return gate.evaluate_working(evidence)


def _record(
    *,
    checkpoint: str,
    parent: str | None,
    matrices: frozenset[str],
    weights: int,
    tier: QualityTier = QualityTier.WORKING,
) -> FrontierRecord:
    return FrontierRecord(
        model_id="openai/whisper-small",
        checkpoint_sha256=checkpoint,
        parent_sha256=parent,
        converted_matrices=matrices,
        converted_weights=weights,
        total_target_weights=1_000,
        gate=_decision(tier=tier),
    )


def test_frontier_rejects_stale_parent_even_when_coverage_is_better() -> None:
    current = _record(
        checkpoint=SHA_A,
        parent=None,
        matrices=frozenset({"m0"}),
        weights=100,
    )
    stale_candidate = _record(
        checkpoint=SHA_B,
        parent=SHA_D,
        matrices=frozenset({"m0", "m1"}),
        weights=200,
    )

    with pytest.raises(StaleParentError):
        AuthoritativeFrontiers.bootstrap(current).advance(stale_candidate)


def test_frontier_rejects_candidate_that_reverts_converted_matrix() -> None:
    current = _record(
        checkpoint=SHA_A,
        parent=None,
        matrices=frozenset({"m0", "m1"}),
        weights=200,
    )
    non_dominating = _record(
        checkpoint=SHA_B,
        parent=SHA_A,
        matrices=frozenset({"m1", "m2", "m3"}),
        weights=300,
    )

    with pytest.raises(NonDominatingFrontierError, match="reverted"):
        AuthoritativeFrontiers.bootstrap(current).advance(non_dominating)


def test_frontier_requires_strictly_more_converted_weight_coverage() -> None:
    current = _record(
        checkpoint=SHA_A,
        parent=None,
        matrices=frozenset({"m0"}),
        weights=100,
    )
    no_progress = _record(
        checkpoint=SHA_B,
        parent=SHA_A,
        matrices=frozenset({"m0", "m1"}),
        weights=100,
    )

    with pytest.raises(NonDominatingFrontierError, match="strictly increase"):
        AuthoritativeFrontiers.bootstrap(current).advance(no_progress)


def test_working_advance_does_not_overwrite_last_strict_milestone() -> None:
    strict = _record(
        checkpoint=SHA_A,
        parent=None,
        matrices=frozenset({"m0"}),
        weights=100,
        tier=QualityTier.STRICT,
    )
    working = _record(
        checkpoint=SHA_B,
        parent=SHA_A,
        matrices=frozenset({"m0", "m1"}),
        weights=200,
        tier=QualityTier.WORKING,
    )

    state = AuthoritativeFrontiers.bootstrap(strict).advance(working)

    assert state.head is working
    assert state.strict_milestone is strict


def test_new_strict_milestone_is_recorded_without_losing_head_lineage() -> None:
    initial_strict = _record(
        checkpoint=SHA_A,
        parent=None,
        matrices=frozenset({"m0"}),
        weights=100,
        tier=QualityTier.STRICT,
    )
    working = _record(
        checkpoint=SHA_B,
        parent=SHA_A,
        matrices=frozenset({"m0", "m1"}),
        weights=200,
    )
    new_strict = _record(
        checkpoint=SHA_C,
        parent=SHA_B,
        matrices=frozenset({"m0", "m1", "m2"}),
        weights=300,
        tier=QualityTier.STRICT,
    )

    state = (
        AuthoritativeFrontiers.bootstrap(initial_strict)
        .advance(working)
        .advance(new_strict)
    )

    assert state.head is new_strict
    assert state.strict_milestone is new_strict
    assert state.head.parent_sha256 == working.checkpoint_sha256


def test_frontier_record_serializes_quality_tier_and_coverage() -> None:
    record = _record(
        checkpoint=SHA_A,
        parent=None,
        matrices=frozenset({"m1", "m0"}),
        weights=250,
    )

    payload = record.to_dict()

    assert payload["schema"] == "wal-tat.whisper-fastlane-frontier.v1"
    assert payload["converted_matrices"] == ["m0", "m1"]
    assert payload["converted_matrix_count"] == 2
    assert payload["coverage"] == pytest.approx(0.25)
    assert payload["quality_tier"] == "working"

