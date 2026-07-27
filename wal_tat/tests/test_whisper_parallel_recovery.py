import json
import hashlib

import pytest
import torch

from whisper_parallel_recovery import (
    _argument_vector,
    _load_spec,
    _ranking,
    _result_is_reusable,
    _validate_parent,
)


def test_argument_vector_handles_repeatable_and_boolean_values() -> None:
    assert _argument_vector(
        {
            "model": "model",
            "train_source": ["clean:train.100:0:8", "other:train.500:0:8"],
            "train_feature_cache_manifest": ["a.json", "b.json"],
            "local_files_only": True,
            "train_codes": False,
            "scale_lr": 1e-5,
            "candidate_group": None,
        }
    ) == [
        "--model",
        "model",
        "--train-source",
        "clean:train.100:0:8",
        "--train-source",
        "other:train.500:0:8",
        "--train-feature-cache-manifest",
        "a.json",
        "--train-feature-cache-manifest",
        "b.json",
        "--local-files-only",
        "--scale-lr",
        "1e-05",
    ]


def test_load_spec_merges_overrides_and_rejects_controller_paths(tmp_path) -> None:
    path = tmp_path / "spec.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "common": {"model": "model", "steps": 8},
                "arms": [
                    {"label": "small lr", "overrides": {"scale_lr": 1e-6}},
                    {"label": "large", "overrides": {"steps": 16}},
                ],
            }
        )
    )
    _, arms = _load_spec(path)
    assert [value["label"] for value in arms] == ["small_lr", "large"]
    assert arms[0]["arguments"]["steps"] == 8
    assert arms[1]["arguments"]["steps"] == 16

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "common": {"model": "model"},
                "arms": [{"label": "bad", "overrides": {"output": "owned"}}],
            }
        )
    )
    with pytest.raises(ValueError, match="controller-owned"):
        _load_spec(path)


def test_argument_vector_forwards_controller_request_sha256() -> None:
    request_sha256 = "a" * 64
    assert _argument_vector({"request_sha256": request_sha256}) == [
        "--request-sha256",
        request_sha256,
    ]


def test_validate_parent_requires_strict_accepted_b1_t3(tmp_path) -> None:
    path = tmp_path / "parent.pt"
    torch.save(
        {
            "accepted": True,
            "provisional": False,
            "matrices": {"layer": {"precision": "t3"}},
        },
        path,
    )
    resolved, digest = _validate_parent({"parent_checkpoint": str(path)})
    assert resolved == path.resolve()
    assert len(digest) == 64

    torch.save(
        {
            "accepted": True,
            "provisional": False,
            "matrices": {"layer": {"precision": "q4"}},
        },
        path,
    )
    with pytest.raises(ValueError, match="non-B1/T3"):
        _validate_parent({"parent_checkpoint": str(path)})


def test_ranking_uses_worker_selected_state_not_always_recovered(tmp_path) -> None:
    output = tmp_path / "result.json"
    output.write_text(
        json.dumps(
                {
                    "selected": "initial",
                    "checkpoint_sha256": "selected-sha",
                    "recovered_checkpoint_sha256": "recovered-sha",
                    "gate": {"max_absolute_wer_ucb": 0.01},
                "initial": {"wer": 0.041, "nll": 0.52},
                "initial_bootstrap": {"upper": 0.008},
                "recovered": {"wer": 0.043, "nll": 0.54},
                "recovered_bootstrap": {"upper": 0.011},
            }
        )
    )
    ranking = _ranking(
        [
            {
                "label": "rollback-winner",
                "status": "completed",
                "output": str(output),
                "selected_checkpoint": str(tmp_path / "selected.pt"),
                "provisional_checkpoint": str(tmp_path / "recovered.pt"),
            }
        ]
    )

    assert ranking == [
        {
            "label": "rollback-winner",
            "checkpoint": str(tmp_path / "selected.pt"),
            "checkpoint_sha256": "selected-sha",
            "selected_checkpoint": str(tmp_path / "selected.pt"),
            "selected_checkpoint_sha256": "selected-sha",
            "recovered_diagnostic_checkpoint": str(tmp_path / "recovered.pt"),
            "recovered_diagnostic_checkpoint_sha256": "recovered-sha",
            "selected": "initial",
            "local_wer": 0.041,
            "local_nll": 0.52,
            "local_wer_ucb": 0.008,
            "local_gate_passed": True,
        }
    ]


def test_recovery_reuse_is_pinned_to_parent_and_request_sha(tmp_path) -> None:
    parent = tmp_path / "parent.pt"
    torch.save({"accepted": True, "provisional": False, "matrices": {}}, parent)
    parent_sha256 = hashlib.sha256(parent.read_bytes()).hexdigest()
    output = tmp_path / "result.json"
    output.write_text(
        json.dumps(
            {
                "parent_checkpoint": str(parent.resolve()),
                "parent_checkpoint_sha256": parent_sha256,
                "recovery_request_sha256": "a" * 64,
            }
        )
    )

    assert _result_is_reusable(
        output,
        parent.resolve(),
        parent_sha256=parent_sha256,
        request_sha256="a" * 64,
    )
    assert not _result_is_reusable(
        output,
        parent.resolve(),
        parent_sha256=parent_sha256,
        request_sha256="b" * 64,
    )
