from pathlib import Path
import sys

import pytest


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from commit_partial_initializer_artifact import (  # noqa: E402
    configured_validation_suites,
    validate_audits,
)


def audit(*, suite_hash: str, checkpoint_hash: str, artifact_hash: str) -> dict:
    domain = {
        "observed_ratio": 1.0,
        "ratio_ci": {"lower": 0.999, "upper": 1.001},
    }
    return {
        "schema": "wal-tat-partial-initializer-artifact-audit-v1",
        "checkpoint_sha256": checkpoint_hash,
        "artifact_sha256": artifact_hash,
        "suite_sha256": suite_hash,
        "candidate_groups": 8,
        "cumulative_candidate_vs_bf16": {"test": domain},
        "incremental_candidate_vs_frontier": {"test": domain},
    }


def acceptance(**extra: object) -> dict:
    return {
        "cumulative_point_ratio_max_each_domain": 1.01,
        "incremental_paired_upper_ratio_max_each_domain": 1.002,
        **extra,
    }


def test_normalizes_recurring_validation_policy() -> None:
    policy = {
        "recurring_validation": [{"name": "r1", "sha256": "suite-r1"}],
    }
    assert configured_validation_suites(policy) == [
        {"name": "r1", "sha256": "suite-r1"}
    ]


def test_validates_predeclared_one_shot_audit_without_required_name_list() -> None:
    policy = {
        "one_shot_audit": {"sha256": "suite-v13"},
        "candidate": {"groups": 8},
        "acceptance": acceptance(),
    }
    result = validate_audits(
        [
            audit(
                suite_hash="suite-v13",
                checkpoint_hash="frontier",
                artifact_hash="candidate",
            )
        ],
        policy,
        checkpoint_hash="frontier",
        artifact_hash="candidate",
    )
    assert set(result) == {"one_shot_audit"}


def test_validates_split_one_shot_audit_policy() -> None:
    policy = {
        "sealed_audit": {"sha256": "suite-v24"},
        "frozen_candidate": {"groups": 8},
        "acceptance": acceptance(),
    }
    result = validate_audits(
        [
            audit(
                suite_hash="suite-v24",
                checkpoint_hash="frontier",
                artifact_hash="candidate",
            )
        ],
        policy,
        checkpoint_hash="frontier",
        artifact_hash="candidate",
    )
    assert set(result) == {"sealed_audit"}


def test_rejects_policy_without_any_validation_suite() -> None:
    with pytest.raises(ValueError, match="no validation suites"):
        configured_validation_suites({})
