from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from whisper_fastlane_campaign import (  # noqa: E402
    RankedGroup,
    apply_package_outcomes,
    assert_authoritative_parent,
    build_dry_run_plan,
    build_recovery_spec,
    filter_remaining_ranked_groups,
    load_or_initialize_state,
    make_package,
    needs_full_audit,
    partition_ranked_groups,
    plan_recovery_batch,
    rank_local_shortlist,
    select_working_candidate,
    singleton_group_matrix_name,
)


def _groups(count: int) -> list[RankedGroup]:
    return [RankedGroup(f"encoder.{index}.self_q") for index in range(count)]


def test_fixed_ranking_partitions_once_as_4_4_2_1() -> None:
    packages = partition_ranked_groups(_groups(11))

    assert [value["size"] for value in packages] == [4, 4, 2, 1]
    assert [
        group["group_id"]
        for package in packages
        for group in package["groups"]
    ] == [value.group_id for value in _groups(11)]


def test_recovery_batch_is_bounded_and_structurally_disjoint() -> None:
    packages = partition_ranked_groups(_groups(20))

    batch = plan_recovery_batch(packages, maximum_candidates=4)

    assert len(batch) == 4
    identifiers = [
        group["group_id"] for package in batch for group in package["groups"]
    ]
    assert len(identifiers) == len(set(identifiers)) == 16
    # Planning never mutates or reranks the fixed queue.
    assert [value["package_id"] for value in packages[:4]] == [
        value["package_id"] for value in batch
    ]


def test_only_real_working_failures_split_4_to_2_to_1_and_defer_others() -> None:
    original = partition_ranked_groups(_groups(9))
    first, unselected, tail = original

    pending, terminal = apply_package_outcomes(
        original,
        accepted_package_id=None,
        failed_working_package_ids={first["package_id"]},
        deferred_package_ids={unselected["package_id"]},
    )
    assert not terminal
    assert [value["size"] for value in pending] == [2, 2, 1, 4]
    assert pending[2]["package_id"] == tail["package_id"]
    assert pending[3]["package_id"] == unselected["package_id"]
    assert pending[3]["attempts"] == 1

    failed_pair = pending[0]
    pending, terminal = apply_package_outcomes(
        pending,
        accepted_package_id=None,
        failed_working_package_ids={failed_pair["package_id"]},
    )
    assert not terminal
    assert [value["size"] for value in pending[:3]] == [1, 1, 2]

    failed_singleton = pending[0]
    pending, terminal = apply_package_outcomes(
        pending,
        accepted_package_id=None,
        failed_working_package_ids={failed_singleton["package_id"]},
    )
    assert len(terminal) == 1
    assert terminal[0]["terminal_reason"] == "failed_working_gate_at_singleton"


def test_fastlane_v1_rejects_multi_matrix_ranked_entries() -> None:
    with pytest.raises(ValueError, match="exactly one matrix"):
        RankedGroup("encoder.0.self_qk", matrix_count=2)


def test_singleton_preflight_resolves_and_filters_committed_matrices() -> None:
    groups = [
        RankedGroup("encoder.0.self_q"),
        RankedGroup("decoder.7.cross_k"),
        RankedGroup("decoder.7.mlp_out"),
    ]

    assert singleton_group_matrix_name("decoder.7.cross_k") == (
        "model.decoder.layers.7.encoder_attn.k_proj"
    )
    remaining = filter_remaining_ranked_groups(
        groups,
        committed_matrices={
            "model.decoder.layers.7.encoder_attn.k_proj",
        },
    )
    assert [value.group_id for value in remaining] == [
        "encoder.0.self_q",
        "decoder.7.mlp_out",
    ]


def test_singleton_preflight_rejects_structural_aliases() -> None:
    with pytest.raises(ValueError, match="multi-matrix/unknown"):
        singleton_group_matrix_name("decoder.7.cross_qk")


def test_local_evaluation_only_ranks_top_two() -> None:
    ranking = [
        {"label": "c", "local_wer_ucb": 0.03, "local_wer": 0.04, "local_nll": 2.0},
        {"label": "a", "local_wer_ucb": 0.01, "local_wer": 0.05, "local_nll": 3.0},
        {"label": "b", "local_wer_ucb": 0.01, "local_wer": 0.04, "local_nll": 4.0},
        {"label": "d", "local_wer_ucb": 0.04, "local_wer": 0.03, "local_nll": 1.0},
    ]

    shortlisted = rank_local_shortlist(ranking)

    assert [value["label"] for value in shortlisted] == ["b", "a"]
    assert all("passed" not in value for value in shortlisted)


def test_working_gate_is_explicit_and_separate_from_strict() -> None:
    selected, failed = select_working_candidate(
        [
            {
                "label": "good",
                "wer": 0.045,
                "wer_ucb": 0.010,
                "nll": 1.0,
            },
            {
                "label": "relative-fail",
                "wer": 0.0461,
                "wer_ucb": 0.010,
                "nll": 0.9,
            },
            {
                "label": "catastrophic-fail",
                "wer": 0.044,
                "wer_ucb": 0.021,
                "nll": 0.8,
            },
        ],
        baseline_wer=0.04,
    )

    assert selected is not None
    assert selected["label"] == "good"
    assert selected["quality_tier"] == "working"
    assert selected["working_gate"]["maximum_relative_wer"] == 1.15
    assert failed == {"relative-fail", "catastrophic-fail"}


def test_working_gate_uses_medium_subset_baseline_from_ranking() -> None:
    selected, failed = select_working_candidate(
        [
            {
                "label": "subset-fail",
                "baseline_wer": 0.035,
                "wer": 0.041,
                "wer_ucb": 0.010,
                "nll": 1.0,
            }
        ],
        # A full-suite fallback with a different WER must not be mixed with
        # the candidate's deterministic medium subset.
        baseline_wer=0.040,
    )

    assert selected is None
    assert failed == {"subset-fail"}


@pytest.mark.parametrize(
    ("candidate", "last_full", "strict", "final", "expected"),
    [
        (105, 98, False, False, False),
        (106, 98, False, False, True),
        (99, 98, True, False, True),
        (99, 98, False, True, True),
    ],
)
def test_full_2703_audit_only_at_milestone_or_strict_boundary(
    candidate: int,
    last_full: int,
    strict: bool,
    final: bool,
    expected: bool,
) -> None:
    assert (
        needs_full_audit(
            candidate_matrix_count=candidate,
            last_full_matrix_count=last_full,
            strict_check_requested=strict,
            final_candidate=final,
        )
        is expected
    )


def test_recovery_arms_differ_only_by_structural_package(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pt"
    parent.touch()
    packages = [
        make_package(_groups(4)),
        make_package(
            [RankedGroup(f"decoder.{index}.cross_q") for index in range(4)]
        ),
    ]

    spec = build_recovery_spec(
        common={"model": "openai/whisper-small", "steps": 256, "train_codes": True},
        parent_checkpoint=parent,
        packages=packages,
    )

    assert spec["common"]["steps"] == 256
    assert len(spec["arms"]) == 2
    assert {
        frozenset(value["overrides"])
        for value in spec["arms"]
    } == {frozenset({"candidate_groups", "candidate_precision"})}
    assert {
        value["overrides"]["candidate_precision"] for value in spec["arms"]
    } == {"t3"}
    assert (
        spec["arms"][0]["overrides"]["candidate_groups"]
        != spec["arms"][1]["overrides"]["candidate_groups"]
    )


def test_recovery_common_cannot_smuggle_hyperparameter_arm_or_parent(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.pt"
    parent.touch()
    with pytest.raises(ValueError, match="controller-owned"):
        build_recovery_spec(
            common={
                "model": "openai/whisper-small",
                "parent_checkpoint": "stale.pt",
            },
            parent_checkpoint=parent,
            packages=[make_package(_groups(4))],
        )


def test_stale_authoritative_parent_sha_aborts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "frontier.pt"
    torch.save(
        {
            "accepted": True,
            "provisional": False,
            "quality_tier": "working",
            "matrices": {
                "m": {
                    "precision": "t3",
                    "codes": torch.zeros(1, dtype=torch.int8),
                }
            },
        },
        checkpoint,
    )
    import whisper_fastlane_campaign as campaign

    digest = campaign.sha256_file(checkpoint)
    state = {
        "head": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
        }
    }
    assert_authoritative_parent(state)
    torch.save({"different": True}, checkpoint)

    with pytest.raises(RuntimeError, match="stale authoritative parent"):
        assert_authoritative_parent(state)


def test_resume_state_is_pinned_and_not_reinitialized(tmp_path: Path) -> None:
    import whisper_fastlane_campaign as campaign

    parent = tmp_path / "parent.pt"
    torch.save(
        {
            "accepted": True,
            "provisional": False,
            "quality_tier": "strict-validation",
            "groups": [],
            "matrices": {
                "old": {
                    "precision": "t3",
                    "codes": torch.zeros(1, dtype=torch.int8),
                }
            },
        },
        parent,
    )
    ranked = tmp_path / "ranked.json"
    ranked.write_text(
        json.dumps(
            {
                "groups": [
                    "encoder.0.self_q",
                    "encoder.0.self_k",
                    "encoder.0.self_v",
                    "encoder.0.self_o",
                ]
            }
        )
    )
    common = tmp_path / "common.json"
    common.write_text(json.dumps({"model": "openai/whisper-small", "steps": 8}))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"metrics": {"wer": 0.04}, "utterances": []}))
    state_path = tmp_path / "state.json"

    class Args:
        model = "openai/whisper-small"
        parent_checkpoint = parent
        parent_sha256 = campaign.sha256_file(parent)
        initial_quality_tier = "strict-validation"
        ranked_groups = ranked
        recovery_common = common
        state = state_path
        output_root = tmp_path / "out"
        devices = ("2", "3")
        maximum_concurrent = 4
        candidate_precision = "t3"
        medium_top_k = 2
        medium_baseline_result = baseline
        medium_feature_cache_manifest = None
        medium_split = "validation"
        medium_samples = 1024
        medium_offset = 0
        full_baseline_result = baseline
        full_feature_cache_manifest = None
        full_split = "validation"
        full_samples = 2703
        full_offset = 0
        full_milestone_matrices = 8
        dataset = "openslr/librispeech_asr"
        dataset_config = "clean"
        batch_size = 16
        working_max_relative_wer = 1.15
        working_max_wer_ucb = 0.020
        strict_max_wer_ucb = 0.005
        medium_bootstrap_replicates = 10_000
        full_bootstrap_replicates = 50_000
        confidence = 0.95
        seed = 17301
        strict_check = False
        resume = False

    initial = load_or_initialize_state(Args, persist=True)
    initial["round_index"] = 7
    campaign.atomic_write_json(state_path, initial)
    Args.resume = True

    resumed = load_or_initialize_state(Args, persist=True)

    assert resumed["round_index"] == 7
    assert resumed["head"]["checkpoint_sha256"] == Args.parent_sha256


def test_dry_run_builds_commands_without_launching_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import whisper_fastlane_campaign as campaign

    parent = tmp_path / "parent.pt"
    torch.save(
        {
            "accepted": True,
            "provisional": False,
            "quality_tier": "working",
            "groups": [],
            "matrices": {
                "old": {
                    "precision": "t3",
                    "codes": torch.zeros(1, dtype=torch.int8),
                }
            },
        },
        parent,
    )
    common = tmp_path / "common.json"
    common.write_text(
        json.dumps({"model": "openai/whisper-small", "steps": 256})
    )
    digest = campaign.sha256_file(parent)
    state = {
        "head": {
            "checkpoint": str(parent),
            "checkpoint_sha256": digest,
            "matrix_count": 1,
            "quality_tier": "working",
        },
        "pending_packages": partition_ranked_groups(_groups(8)),
        "round_index": 0,
        "active_round": None,
    }
    args = SimpleNamespace(
        output_root=tmp_path / "out",
        maximum_concurrent=4,
        candidate_precision="t3",
        recovery_common=common,
        devices=("2", "3"),
        working_max_relative_wer=1.15,
        working_max_wer_ucb=0.020,
        strict_max_wer_ucb=0.005,
        full_samples=2703,
        full_milestone_matrices=8,
        medium_top_k=2,
        medium_samples=1024,
    )
    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("dry run launched a subprocess"),
    )

    plan = build_dry_run_plan(args, state)

    assert plan["status"] == "dry-run"
    assert len(plan["batch"]) == 2
    assert plan["recovery_command"][-1] == "--resume"
    assert not (tmp_path / "out").exists()
