import json

import pytest

from wal_tat import (
    common_campaign_frontier,
    accepted_weight_counts,
    atomic_write_json,
    coverage_proportional_nll_gate,
    sha256_file,
    synchronize_campaign_frontiers,
    validate_checkpoint_deletion_target,
    worst_ratio,
)


def test_worst_ratio_requires_and_combines_independent_audits():
    assert worst_ratio(
        [
            {"ratios": {"c4": 0.99, "squad": 1.001}},
            {"ratios": {"c4": 0.995, "code": 0.98}},
        ]
    ) == pytest.approx(1.001)
    with pytest.raises(ValueError, match="at least one"):
        worst_ratio([])


def test_checkpoint_cleanup_is_restricted_to_exact_wal_tat_file(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    valid = checkpoint_dir / "wal-tat-frontier.pt"
    valid.write_bytes(b"checkpoint")
    assert validate_checkpoint_deletion_target(valid, checkpoint_dir) == valid.resolve()

    invalid = checkpoint_dir / "other.pt"
    invalid.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="wal-tat"):
        validate_checkpoint_deletion_target(invalid, checkpoint_dir)

    outside = tmp_path / "wal-tat-outside.pt"
    outside.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="outside"):
        validate_checkpoint_deletion_target(outside, checkpoint_dir)

    link = checkpoint_dir / "wal-tat-link.pt"
    link.symlink_to(valid)
    with pytest.raises(ValueError, match="symlinks"):
        validate_checkpoint_deletion_target(link, checkpoint_dir)


def test_atomic_campaign_state_replaces_valid_json(tmp_path):
    path = tmp_path / "campaign.json"
    atomic_write_json(path, {"step": 1})
    assert json.loads(path.read_text()) == {"step": 1}
    atomic_write_json(path, {"step": 2, "passed": True})
    assert json.loads(path.read_text()) == {"step": 2, "passed": True}
    assert list(tmp_path.glob("*.tmp")) == []


def test_exact_accepted_weight_count_handles_padded_last_group():
    checkpoint = {
        "matrices": {
            "matrix": {
                "shape": (2, 5),
                "group_size": 4,
                "committed_mask": [[True, True], [False, True]],
            }
        }
    }
    assert accepted_weight_counts(checkpoint) == {"matrix": 6}


def test_coverage_proportional_gate_reaches_declared_full_budget():
    assert coverage_proportional_nll_gate(25, 100, 0.05) == pytest.approx(1.0125)
    assert coverage_proportional_nll_gate(100, 100, 0.05) == pytest.approx(1.05)
    with pytest.raises(ValueError):
        coverage_proportional_nll_gate(101, 100, 0.05)


def test_round_robin_frontier_sync_preserves_per_campaign_coverage(tmp_path):
    first_checkpoint = tmp_path / "wal-tat-first.pt"
    first_checkpoint.write_bytes(b"first")
    first_digest = sha256_file(first_checkpoint)
    states = []
    for index, coverage in enumerate((0.25, 0.75)):
        path = tmp_path / f"campaign-{index}.json"
        atomic_write_json(
            path,
            {
                "frontier": {
                    "checkpoint": str(first_checkpoint),
                    "sha256": first_digest,
                    "candidate_coverage": coverage,
                }
            },
        )
        states.append(path)
    assert common_campaign_frontier(states) == (
        first_checkpoint.resolve(),
        first_digest,
    )

    second_checkpoint = tmp_path / "wal-tat-second.pt"
    second_checkpoint.write_bytes(b"second")
    second_digest = synchronize_campaign_frontiers(states, second_checkpoint)
    assert common_campaign_frontier(states) == (
        second_checkpoint.resolve(),
        second_digest,
    )
    assert [
        json.loads(path.read_text())["frontier"]["candidate_coverage"]
        for path in states
    ] == [0.25, 0.75]


def test_round_robin_frontier_rejects_divergent_campaigns(tmp_path):
    states = []
    for index in range(2):
        checkpoint = tmp_path / f"wal-tat-{index}.pt"
        checkpoint.write_bytes(str(index).encode())
        path = tmp_path / f"campaign-{index}.json"
        atomic_write_json(
            path,
            {
                "frontier": {
                    "checkpoint": str(checkpoint),
                    "sha256": sha256_file(checkpoint),
                    "candidate_coverage": 0.0,
                }
            },
        )
        states.append(path)
    with pytest.raises(ValueError, match="not synchronized"):
        common_campaign_frontier(states)
