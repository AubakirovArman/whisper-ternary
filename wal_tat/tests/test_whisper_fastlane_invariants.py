from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import promote_whisper_frontier as promotion  # noqa: E402
import whisper_fastlane_campaign as campaign  # noqa: E402
import whisper_parallel_recovery as parallel_recovery  # noqa: E402


PARENT_Q = "model.encoder.layers.0.self_attn.q_proj"
PARENT_K = "model.encoder.layers.0.self_attn.k_proj"
NEW_V = "model.encoder.layers.0.self_attn.v_proj"
EXTRA_O = "model.encoder.layers.0.self_attn.out_proj"


def _matrix() -> dict[str, object]:
    return {
        "precision": "t3",
        "codes": torch.zeros(1, dtype=torch.int8),
    }


def _save_parent(path: Path, matrix_names: set[str]) -> str:
    torch.save(
        {
            "accepted": True,
            "provisional": False,
            "quality_tier": "working",
            "groups": [],
            "matrices": {name: _matrix() for name in sorted(matrix_names)},
        },
        path,
    )
    return campaign.sha256_file(path)


def _save_candidate(
    path: Path,
    *,
    parent: Path,
    parent_sha256: str,
    matrix_names: set[str],
    candidate_groups: list[str],
) -> None:
    torch.save(
        {
            "accepted": False,
            "provisional": True,
            "quality_tier": "working",
            "parent_checkpoint": str(parent.resolve()),
            "parent_checkpoint_sha256": parent_sha256,
            "candidate_groups": candidate_groups,
            "groups": candidate_groups,
            "matrices": {name: _matrix() for name in sorted(matrix_names)},
        },
        path,
    )


def _wer_entry(identifier: str, *, errors: int = 4, words: int = 100) -> dict:
    return {
        "id": identifier,
        "substitutions": errors,
        "deletions": 0,
        "insertions": 0,
        "reference_words": words,
    }


def _write_wer_result(
    path: Path,
    *,
    source_checkpoint: Path | None = None,
    source_checkpoint_sha256: str | None = None,
    errors: int = 4,
) -> None:
    payload = {
        "metrics": {"wer": errors / 100, "nll": 0.5},
        "utterances": [_wer_entry("u0", errors=errors)],
    }
    if source_checkpoint is not None:
        payload["source_checkpoint"] = str(source_checkpoint.resolve())
    if source_checkpoint_sha256 is not None:
        payload["source_checkpoint_sha256"] = source_checkpoint_sha256
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_candidate_cannot_drop_a_parent_matrix_even_if_count_increases(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.pt"
    parent_sha = _save_parent(parent, {PARENT_Q, PARENT_K})
    candidate = tmp_path / "candidate.pt"
    _save_candidate(
        candidate,
        parent=parent,
        parent_sha256=parent_sha,
        # The count grows from two to three, but PARENT_Q was silently lost.
        matrix_names={PARENT_K, NEW_V, EXTRA_O},
        candidate_groups=["encoder.0.self_v", "encoder.0.self_o"],
    )

    with pytest.raises((ValueError, RuntimeError), match="parent|revert|missing"):
        campaign._checkpoint_metadata_for_candidate(
            candidate,
            parent=parent,
            parent_sha256=parent_sha,
        )


def test_candidate_cannot_add_a_matrix_outside_its_declared_groups(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.pt"
    parent_sha = _save_parent(parent, {PARENT_Q})
    candidate = tmp_path / "candidate.pt"
    _save_candidate(
        candidate,
        parent=parent,
        parent_sha256=parent_sha,
        # Only self_v is declared, but self_o was added as well.
        matrix_names={PARENT_Q, NEW_V, EXTRA_O},
        candidate_groups=["encoder.0.self_v"],
    )

    with pytest.raises((ValueError, RuntimeError), match="unexpected|declared|delta"):
        campaign._checkpoint_metadata_for_candidate(
            candidate,
            parent=parent,
            parent_sha256=parent_sha,
        )


def test_candidate_rejects_a_recorded_stale_parent_sha(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pt"
    parent_sha = _save_parent(parent, {PARENT_Q})
    candidate = tmp_path / "candidate.pt"
    _save_candidate(
        candidate,
        parent=parent,
        parent_sha256="0" * 64,
        matrix_names={PARENT_Q, NEW_V},
        candidate_groups=["encoder.0.self_v"],
    )

    with pytest.raises(RuntimeError, match="stale parent"):
        campaign._checkpoint_metadata_for_candidate(
            candidate,
            parent=parent,
            parent_sha256=parent_sha,
        )


def test_recovery_result_is_not_reusable_after_same_parent_path_changes(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.pt"
    old_sha = _save_parent(parent, {PARENT_Q})
    result = tmp_path / "recovery.json"
    result.write_text(
        json.dumps(
            {
                "parent_checkpoint": str(parent.resolve()),
                "parent_checkpoint_sha256": old_sha,
            }
        ),
        encoding="utf-8",
    )

    # Mutate the content without changing the filesystem path.
    new_sha = _save_parent(parent, {PARENT_Q, PARENT_K})
    assert new_sha != old_sha

    reusable = parallel_recovery._result_is_reusable
    if "parent_sha256" in inspect.signature(reusable).parameters:
        observed = reusable(result, parent.resolve(), parent_sha256=new_sha)
    else:
        observed = reusable(result, parent.resolve())
    assert observed is False


def test_promotion_rejects_audit_result_for_another_source_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent.pt"
    parent_sha = _save_parent(parent, {PARENT_Q})
    source = tmp_path / "candidate.pt"
    _save_candidate(
        source,
        parent=parent,
        parent_sha256=parent_sha,
        matrix_names={PARENT_Q, NEW_V},
        candidate_groups=["encoder.0.self_v"],
    )
    baseline = tmp_path / "baseline.json"
    _write_wer_result(baseline)
    candidate_result = tmp_path / "candidate-result.json"
    _write_wer_result(
        candidate_result,
        source_checkpoint=source,
        source_checkpoint_sha256="f" * 64,
    )
    output = tmp_path / "promoted.pt"
    comparison = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_whisper_frontier.py",
            "--source-checkpoint",
            str(source),
            "--parent-checkpoint",
            str(parent),
            "--baseline-result",
            str(baseline),
            "--candidate-result",
            str(candidate_result),
            "--output-checkpoint",
            str(output),
            "--comparison-output",
            str(comparison),
            "--quality-tier",
            "working",
            "--max-wer-ucb",
            "1",
            "--max-relative-wer",
            "2",
            "--replicates",
            "20",
        ],
    )

    with pytest.raises((ValueError, RuntimeError), match="SHA|sha|source"):
        promotion.main()


def _package(group_id: str) -> dict[str, object]:
    return campaign.make_package([campaign.RankedGroup(group_id)])


def _active_round(
    root: Path,
    *,
    parent: Path,
    parent_sha256: str,
    packages: list[dict[str, object]],
) -> dict[str, object]:
    labels = {
        str(package["package_id"]): str(package["package_id"])
        for package in packages
    }
    return {
        "round_index": 1,
        "parent_checkpoint": str(parent.resolve()),
        "parent_checkpoint_sha256": parent_sha256,
        "stage": "planned",
        "packages": packages,
        "package_by_label": labels,
        "root": str(root),
        "recovery_spec": str(root / "recovery-spec.json"),
        "recovery_summary": str(root / "recovery-summary.json"),
        "recovery_output_dir": str(root / "recovery"),
        "medium_summary": str(root / "medium-summary.json"),
        "medium_output_dir": str(root / "medium"),
        "full_summary": str(root / "full-summary.json"),
        "full_output_dir": str(root / "full"),
        "promotion_checkpoint": str(root / "promoted.pt"),
        "promotion_comparison": str(root / "comparison.json"),
    }


def _state(
    *,
    parent: Path,
    parent_sha256: str,
    active: dict[str, object],
    pending: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema": campaign.STATE_SCHEMA,
        "status": "running",
        "head": {
            "checkpoint": str(parent.resolve()),
            "checkpoint_sha256": parent_sha256,
            "matrix_count": 1,
            "quality_tier": "working",
        },
        "strict_milestone": None,
        "last_full_matrix_count": 1,
        "strict_check_requested": False,
        "pending_packages": pending,
        "terminal_failed_packages": [],
        "active_round": active,
        "round_index": 0,
        "history": [],
    }


def _args(tmp_path: Path, baseline: Path) -> SimpleNamespace:
    return SimpleNamespace(
        state=tmp_path / "state.json",
        recovery_common=tmp_path / "unused-common.json",
        candidate_precision="t3",
        devices=("2", "3"),
        maximum_concurrent=4,
        medium_top_k=2,
        model="openai/whisper-small",
        medium_baseline_result=baseline,
        medium_feature_cache_manifest=None,
        medium_split="validation",
        medium_samples=1024,
        medium_offset=0,
        dataset="openslr/librispeech_asr",
        dataset_config="clean",
        batch_size=16,
        working_max_relative_wer=1.15,
        working_max_wer_ucb=0.020,
        medium_bootstrap_replicates=20,
        confidence=0.95,
        seed=17000,
        local_files_only=True,
        full_milestone_matrices=8,
        full_baseline_result=baseline,
        full_feature_cache_manifest=None,
        full_split="validation",
        full_samples=2703,
        full_offset=0,
        strict_max_wer_ucb=0.005,
        full_bootstrap_replicates=20,
        sealed_final=False,
    )


def _write_recovery_summary(
    active: dict[str, object],
    *,
    parent_sha256: str,
    candidates: list[tuple[dict[str, object], Path, float]],
) -> None:
    spec_path = Path(str(active["recovery_spec"]))
    spec_path.write_text("{}", encoding="utf-8")
    ranking = []
    arms = []
    for package, checkpoint, local_wer_ucb in candidates:
        label = str(package["package_id"])
        checkpoint_sha256 = campaign.sha256_file(checkpoint)
        diagnostic = checkpoint.with_suffix(".diagnostic.pt")
        torch.save(
            torch.load(checkpoint, map_location="cpu", weights_only=True),
            diagnostic,
        )
        ranking.append(
            {
                "label": label,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "selected_checkpoint": str(checkpoint),
                "selected_checkpoint_sha256": checkpoint_sha256,
                "recovered_diagnostic_checkpoint": str(diagnostic),
                "recovered_diagnostic_checkpoint_sha256": campaign.sha256_file(
                    diagnostic
                ),
                "selected": "recovered",
                "local_wer": 0.04,
                "local_nll": 0.5,
                "local_wer_ucb": local_wer_ucb,
            }
        )
        arms.append({"label": label, "recovery_request_sha256": "a" * 64})
    Path(str(active["recovery_summary"])).write_text(
        json.dumps(
            {
                "status": "completed",
                "spec_sha256": campaign.sha256_file(spec_path),
                "parent_checkpoint": active["parent_checkpoint"],
                "parent_checkpoint_sha256": parent_sha256,
                "arms": arms,
                "ranking": ranking,
            }
        ),
        encoding="utf-8",
    )


def _write_audit_summary(
    path: Path,
    *,
    parent_sha256: str,
    rows: list[dict[str, object]],
) -> None:
    baseline = path.parent.parent / "baseline.json"
    if not baseline.is_file():
        _write_wer_result(baseline)
    request_sha256 = "b" * 64
    for row in rows:
        result_path = Path(str(row["result"]))
        try:
            result_payload = json.loads(result_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            result_payload = {}
        checkpoint = Path(str(row["checkpoint"])).resolve()
        result_payload.update(
            {
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_sha256": campaign.sha256_file(checkpoint),
                "evaluation_request_sha256": request_sha256,
            }
        )
        result_path.write_text(json.dumps(result_payload), encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "baseline_result": str(baseline.resolve()),
                "baseline_result_sha256": campaign.sha256_file(baseline),
                "candidates": [
                    {
                        "label": row["label"],
                        "checkpoint": row["checkpoint"],
                        "checkpoint_sha256": campaign.sha256_file(
                            Path(str(row["checkpoint"]))
                        ),
                        "parent_checkpoint_sha256": parent_sha256,
                        "evaluation_request_sha256": request_sha256,
                        "output": row["result"],
                    }
                    for row in rows
                ],
                "ranking": rows,
            }
        ),
        encoding="utf-8",
    )


def test_full_failure_preserves_union_of_medium_and_full_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "round"
    root.mkdir()
    parent = tmp_path / "parent.pt"
    parent_sha = _save_parent(parent, {PARENT_Q})
    selected_package = _package("encoder.0.self_v")
    medium_failed_package = _package("encoder.0.self_o")
    packages = [selected_package, medium_failed_package]
    active = _active_round(
        root,
        parent=parent,
        parent_sha256=parent_sha,
        packages=packages,
    )
    selected_checkpoint = root / "selected.pt"
    _save_candidate(
        selected_checkpoint,
        parent=parent,
        parent_sha256=parent_sha,
        matrix_names={PARENT_Q, NEW_V},
        candidate_groups=["encoder.0.self_v"],
    )
    failed_checkpoint = root / "failed.pt"
    _save_candidate(
        failed_checkpoint,
        parent=parent,
        parent_sha256=parent_sha,
        matrix_names={PARENT_Q, EXTRA_O},
        candidate_groups=["encoder.0.self_o"],
    )
    _write_recovery_summary(
        active,
        parent_sha256=parent_sha,
        candidates=[
            (selected_package, selected_checkpoint, 0.001),
            (medium_failed_package, failed_checkpoint, 0.002),
        ],
    )
    selected_result = root / "selected-result.json"
    failed_result = root / "failed-result.json"
    selected_result.write_text("{}", encoding="utf-8")
    failed_result.write_text("{}", encoding="utf-8")
    _write_audit_summary(
        Path(str(active["medium_summary"])),
        parent_sha256=parent_sha,
        rows=[
            {
                "label": selected_package["package_id"],
                "passed": True,
                "baseline_wer": 0.04,
                "wer": 0.044,
                "wer_ucb": 0.010,
                "nll": 0.5,
                "checkpoint": str(selected_checkpoint),
                "result": str(selected_result),
            },
            {
                "label": medium_failed_package["package_id"],
                "passed": False,
                "baseline_wer": 0.04,
                "wer": 0.050,
                "wer_ucb": 0.030,
                "nll": 0.6,
                "checkpoint": str(failed_checkpoint),
                "result": str(failed_result),
            },
        ],
    )
    full_result = root / "full-result.json"
    full_result.write_text("{}", encoding="utf-8")
    _write_audit_summary(
        Path(str(active["full_summary"])),
        parent_sha256=parent_sha,
        rows=[
            {
                "label": selected_package["package_id"],
                "passed": False,
                "baseline_wer": 0.04,
                "wer": 0.050,
                "wer_ucb": 0.030,
                "nll": 0.6,
                "checkpoint": str(selected_checkpoint),
                "result": str(full_result),
            }
        ],
    )
    state = _state(
        parent=parent,
        parent_sha256=parent_sha,
        active=active,
        pending=packages,
    )
    state["strict_check_requested"] = True
    baseline = tmp_path / "baseline.json"
    _write_wer_result(baseline)
    observed: dict[str, set[str]] = {}

    def capture_close(
        args: object,
        state: dict[str, object],
        *,
        active: dict[str, object],
        failed_package_ids: set[str],
        medium_ranking: object,
    ) -> None:
        observed["failed"] = set(failed_package_ids)

    monkeypatch.setattr(campaign, "_close_failed_round", capture_close)

    assert campaign._execute_round(_args(tmp_path, baseline), state)
    assert observed["failed"] == {
        str(selected_package["package_id"]),
        str(medium_failed_package["package_id"]),
    }


def test_terminal_failure_cannot_produce_completed_or_strict_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "round"
    root.mkdir()
    parent = tmp_path / "parent.pt"
    parent_sha = _save_parent(parent, {PARENT_Q})
    selected_package = _package("encoder.0.self_v")
    failed_package = _package("encoder.0.self_o")
    packages = [selected_package, failed_package]
    active = _active_round(
        root,
        parent=parent,
        parent_sha256=parent_sha,
        packages=packages,
    )
    selected_checkpoint = root / "selected.pt"
    _save_candidate(
        selected_checkpoint,
        parent=parent,
        parent_sha256=parent_sha,
        matrix_names={PARENT_Q, NEW_V},
        candidate_groups=["encoder.0.self_v"],
    )
    failed_checkpoint = root / "failed.pt"
    _save_candidate(
        failed_checkpoint,
        parent=parent,
        parent_sha256=parent_sha,
        matrix_names={PARENT_Q, EXTRA_O},
        candidate_groups=["encoder.0.self_o"],
    )
    _write_recovery_summary(
        active,
        parent_sha256=parent_sha,
        candidates=[
            (selected_package, selected_checkpoint, 0.001),
            (failed_package, failed_checkpoint, 0.002),
        ],
    )
    selected_result = root / "selected-result.json"
    failed_result = root / "failed-result.json"
    selected_result.write_text("{}", encoding="utf-8")
    failed_result.write_text("{}", encoding="utf-8")
    _write_audit_summary(
        Path(str(active["medium_summary"])),
        parent_sha256=parent_sha,
        rows=[
            {
                "label": selected_package["package_id"],
                "passed": True,
                "baseline_wer": 0.04,
                "wer": 0.044,
                "wer_ucb": 0.010,
                "nll": 0.5,
                "checkpoint": str(selected_checkpoint),
                "result": str(selected_result),
            },
            {
                "label": failed_package["package_id"],
                "passed": False,
                "baseline_wer": 0.04,
                "wer": 0.050,
                "wer_ucb": 0.030,
                "nll": 0.6,
                "checkpoint": str(failed_checkpoint),
                "result": str(failed_result),
            },
        ],
    )
    full_result = root / "full-result.json"
    full_result.write_text("{}", encoding="utf-8")
    _write_audit_summary(
        Path(str(active["full_summary"])),
        parent_sha256=parent_sha,
        rows=[
            {
                "label": selected_package["package_id"],
                "passed": True,
                "baseline_wer": 0.04,
                "wer": 0.041,
                "wer_ucb": 0.004,
                "nll": 0.5,
                "checkpoint": str(selected_checkpoint),
                "result": str(full_result),
            }
        ],
    )
    state = _state(
        parent=parent,
        parent_sha256=parent_sha,
        active=active,
        pending=packages,
    )
    baseline = tmp_path / "baseline.json"
    _write_wer_result(baseline)
    args = _args(tmp_path, baseline)
    args.sealed_final = True

    def fake_run(command: list[str]) -> None:
        assert str(campaign.PROMOTION_SCRIPT) in command
        source = Path(command[command.index("--source-checkpoint") + 1])
        output = Path(command[command.index("--output-checkpoint") + 1])
        tier = command[command.index("--quality-tier") + 1]
        promoted = torch.load(source, map_location="cpu", weights_only=True)
        promoted.update(
            {
                "accepted": True,
                "provisional": False,
                "quality_tier": tier,
                "strict_accepted": tier.startswith("strict-"),
                "promoted_from": str(source.resolve()),
                "promoted_from_sha256": campaign.sha256_file(source),
                    "full_audit": {
                        "parent_checkpoint_sha256": parent_sha,
                        "source_checkpoint_sha256": campaign.sha256_file(source),
                        "baseline_result_sha256": campaign.sha256_file(baseline),
                        "candidate_result_sha256": campaign.sha256_file(
                            full_result
                        ),
                    },
            }
        )
        torch.save(promoted, output)

    monkeypatch.setattr(campaign, "_run", fake_run)

    assert campaign._execute_round(args, state)
    assert state["status"] == "exhausted"
    assert state["head"]["quality_tier"] != "strict-final"
    assert state["terminal_failed_packages"]


def test_existing_promoted_checkpoint_must_match_selected_source_sha(
    tmp_path: Path,
) -> None:
    root = tmp_path / "round"
    root.mkdir()
    parent = tmp_path / "parent.pt"
    parent_sha = _save_parent(parent, {PARENT_Q})
    selected_package = _package("encoder.0.self_v")
    tail_package = _package("encoder.0.self_o")
    active = _active_round(
        root,
        parent=parent,
        parent_sha256=parent_sha,
        packages=[selected_package],
    )
    selected_checkpoint = root / "selected.pt"
    _save_candidate(
        selected_checkpoint,
        parent=parent,
        parent_sha256=parent_sha,
        matrix_names={PARENT_Q, NEW_V},
        candidate_groups=["encoder.0.self_v"],
    )
    _write_recovery_summary(
        active,
        parent_sha256=parent_sha,
        candidates=[(selected_package, selected_checkpoint, 0.001)],
    )
    selected_result = root / "selected-result.json"
    selected_result.write_text("{}", encoding="utf-8")
    _write_audit_summary(
        Path(str(active["medium_summary"])),
        parent_sha256=parent_sha,
        rows=[
            {
                "label": selected_package["package_id"],
                "passed": True,
                "baseline_wer": 0.04,
                "wer": 0.044,
                "wer_ucb": 0.010,
                "nll": 0.5,
                "checkpoint": str(selected_checkpoint),
                "result": str(selected_result),
            }
        ],
    )
    promoted = torch.load(
        selected_checkpoint, map_location="cpu", weights_only=True
    )
    promoted.update(
        {
            "accepted": True,
            "provisional": False,
            "quality_tier": "working",
            "promoted_from": str(selected_checkpoint.resolve()),
            "promoted_from_sha256": "e" * 64,
            "full_audit": {"parent_checkpoint_sha256": parent_sha},
        }
    )
    torch.save(promoted, Path(str(active["promotion_checkpoint"])))
    state = _state(
        parent=parent,
        parent_sha256=parent_sha,
        active=active,
        pending=[selected_package, tail_package],
    )
    baseline = tmp_path / "baseline.json"
    _write_wer_result(baseline)

    with pytest.raises(RuntimeError, match="promoted|source|SHA|sha"):
        campaign._execute_round(_args(tmp_path, baseline), state)
