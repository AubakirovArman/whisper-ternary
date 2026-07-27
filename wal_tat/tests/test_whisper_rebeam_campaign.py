from pathlib import Path

import torch

from whisper_rebeam_campaign import (
    _positive_ints_csv,
    _is_completed_json,
    _merge_branch_queue,
    _pop_backtrack_branch,
    compose_macro_checkpoint,
    select_recovery_promotion,
    select_promotion,
)


def test_is_completed_json_rejects_partial_resume_artifact(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text('{"status": "running"}\n', encoding="utf-8")
    assert not _is_completed_json(summary)

    summary.write_text('{"status": "completed", "ranking": []}\n', encoding="utf-8")
    assert _is_completed_json(summary)


def test_positive_ints_csv() -> None:
    assert _positive_ints_csv("8,2,4,2") == [2, 4, 8]
    assert _positive_ints_csv("") == []


def test_compose_macro_checkpoint_combines_disjoint_strict_edits(
    tmp_path,
) -> None:
    parent_path = tmp_path / "parent.pt"
    parent = {
        "accepted": True,
        "provisional": False,
        "model": "whisper-small",
        "group_size": 128,
        "groups": ["old"],
        "matrices": {
            "old.weight": {
                "precision": "t3",
                "codes": torch.zeros((1, 1, 128), dtype=torch.int8),
                "scales": torch.ones((1, 1), dtype=torch.float16),
            }
        },
        "history": [],
        "full_audit": {"old": True},
    }
    torch.save(parent, parent_path)

    sources = []
    for index in range(2):
        source_path = tmp_path / f"source-{index}.pt"
        source = {
            **parent,
            "accepted": False,
            "provisional": True,
            "parent_checkpoint": str(parent_path.resolve()),
            "candidate_group": f"group-{index}",
            "candidate_groups": [f"group-{index}"],
            "matrices": {
                **parent["matrices"],
                f"new-{index}.weight": {
                    "precision": "t3",
                    "codes": torch.full(
                        (1, 1, 128), index, dtype=torch.int8
                    ),
                    "scales": torch.ones((1, 1), dtype=torch.float16),
                },
            },
        }
        torch.save(source, source_path)
        sources.append(source_path)

    output = tmp_path / "macro.pt"
    metadata = compose_macro_checkpoint(parent_path, sources, output)
    macro = torch.load(output, map_location="cpu", weights_only=True)
    assert not macro["accepted"]
    assert macro["provisional"]
    assert macro["candidate_groups"] == ["group-0", "group-1"]
    assert set(macro["matrices"]) == {
        "old.weight",
        "new-0.weight",
        "new-1.weight",
    }
    assert "full_audit" not in macro
    assert metadata["added_weights"] == 256


def test_backtrack_prefers_deepest_safe_branch(tmp_path) -> None:
    parent = tmp_path / "parent.pt"
    result = tmp_path / "result.json"
    sources = [tmp_path / f"source-{index}.pt" for index in range(3)]
    for path in [parent, result, *sources]:
        path.touch()
    state: dict = {}
    _merge_branch_queue(
        state,
        [
            {
                "source_checkpoint": str(sources[0]),
                "parent_checkpoint": str(parent),
                "result": str(result),
                "matrix_count": 75,
                "lowbit_weights": 100,
                "wer_ucb": 0.0040,
                "wer": 0.039,
            },
            {
                "source_checkpoint": str(sources[1]),
                "parent_checkpoint": str(parent),
                "result": str(result),
                "matrix_count": 76,
                "lowbit_weights": 200,
                "wer_ucb": 0.0047,
                "wer": 0.039,
            },
            {
                "source_checkpoint": str(sources[2]),
                "parent_checkpoint": str(parent),
                "result": str(result),
                "matrix_count": 76,
                "lowbit_weights": 300,
                "wer_ucb": 0.0049,
                "wer": 0.039,
            },
        ],
    )
    selected = _pop_backtrack_branch(state, preferred_wer_ucb=0.0048)
    assert selected is not None
    assert selected["source_checkpoint"] == str(sources[1])
    assert str(sources[1]) in state["explored_branch_sources"]
    assert len(state["branch_queue"]) == 2


def test_select_promotion_coverage_prefers_more_weights(monkeypatch) -> None:
    weights = {
        "small.pt": 10,
        "large.pt": 100,
    }
    monkeypatch.setattr(
        "whisper_rebeam_campaign._added_weight_count",
        lambda checkpoint, parent: weights[checkpoint.name],
    )
    selected = select_promotion(
        [
            {
                "label": "small",
                "passed": True,
                "wer_ucb": 0.0041,
                "wer": 0.039,
                "checkpoint": "small.pt",
            },
            {
                "label": "large",
                "passed": True,
                "wer_ucb": 0.0049,
                "wer": 0.040,
                "checkpoint": "large.pt",
            },
        ],
        Path("parent.pt"),
        policy="coverage",
        preferred_wer_ucb=0.0048,
    )
    assert selected is not None
    assert selected["label"] == "large"


def test_select_promotion_hybrid_respects_preferred_ucb(monkeypatch) -> None:
    weights = {
        "small.pt": 10,
        "large.pt": 100,
        "medium.pt": 60,
    }
    monkeypatch.setattr(
        "whisper_rebeam_campaign._added_weight_count",
        lambda checkpoint, parent: weights[checkpoint.name],
    )
    selected = select_promotion(
        [
            {
                "label": "small",
                "passed": True,
                "wer_ucb": 0.0042,
                "wer": 0.039,
                "checkpoint": "small.pt",
            },
            {
                "label": "large",
                "passed": True,
                "wer_ucb": 0.00495,
                "wer": 0.040,
                "checkpoint": "large.pt",
            },
            {
                "label": "medium",
                "passed": True,
                "wer_ucb": 0.0047,
                "wer": 0.0395,
                "checkpoint": "medium.pt",
            },
        ],
        Path("parent.pt"),
        policy="hybrid",
        preferred_wer_ucb=0.0048,
    )
    assert selected is not None
    assert selected["label"] == "medium"


def test_select_promotion_returns_none_without_passes(monkeypatch) -> None:
    monkeypatch.setattr(
        "whisper_rebeam_campaign._added_weight_count",
        lambda checkpoint, parent: 1,
    )
    assert (
        select_promotion(
            [
                {
                    "label": "failed",
                    "passed": False,
                    "wer_ucb": 0.0051,
                    "wer": 0.04,
                    "checkpoint": "failed.pt",
                }
            ],
            Path("parent.pt"),
            policy="hybrid",
            preferred_wer_ucb=0.0048,
        )
        is None
    )


def test_select_recovery_promotion_requires_real_ucb_gain(monkeypatch) -> None:
    monkeypatch.setattr(
        "whisper_rebeam_campaign._checkpoint_full_ucb",
        lambda checkpoint: 0.0049,
    )
    selected = select_recovery_promotion(
        [
            {
                "label": "too-small",
                "passed": True,
                "wer_ucb": 0.00475,
                "wer": 0.039,
                "nll": 0.6,
            },
            {
                "label": "useful",
                "passed": True,
                "wer_ucb": 0.0046,
                "wer": 0.0391,
                "nll": 0.61,
            },
            {
                "label": "failed",
                "passed": False,
                "wer_ucb": 0.0040,
                "wer": 0.038,
                "nll": 0.59,
            },
        ],
        Path("parent.pt"),
        minimum_ucb_gain=0.0002,
    )
    assert selected is not None
    assert selected["label"] == "useful"


def test_select_recovery_promotion_returns_none_without_gain(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "whisper_rebeam_campaign._checkpoint_full_ucb",
        lambda checkpoint: 0.0049,
    )
    assert (
        select_recovery_promotion(
            [
                {
                    "label": "flat",
                    "passed": True,
                    "wer_ucb": 0.0048,
                    "wer": 0.039,
                    "nll": 0.6,
                }
            ],
            Path("parent.pt"),
            minimum_ucb_gain=0.0002,
        )
        is None
    )
