import json
from pathlib import Path

import torch

from build_whisper_hard_cache import (
    _forbidden_identity_sets,
    _selection_score,
)
from whisper_global_lowbit_recovery import (
    TrainingSource,
    _load_training_feature_cache,
)
from wal_tat import sha256_file


def test_selection_score_prioritizes_positive_student_error_delta() -> None:
    teacher = {
        "id": "a",
        "hypothesis": "teacher",
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
    }
    worse_student = {
        "id": "a",
        "hypothesis": "student",
        "substitutions": 1,
        "deletions": 0,
        "insertions": 0,
    }
    differing_but_not_worse = {
        "id": "b",
        "hypothesis": "different",
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
    }

    assert _selection_score(teacher, worse_student) > _selection_score(
        teacher, differing_but_not_worse
    )


def test_forbidden_sets_include_example_and_document_ids(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "batches": [
                    {
                        "examples": [
                            {"id": "sample-1", "document_id": "document-1"},
                            {"id": "sample-2", "document_id": None},
                        ]
                    }
                ]
            }
        )
    )

    identifiers, documents, metadata = _forbidden_identity_sets([path])

    assert identifiers == {"sample-1", "sample-2"}
    assert documents == {"document-1"}
    assert metadata[0]["samples"] == 2
    assert metadata[0]["manifest_sha256"] == sha256_file(path)


def test_recovery_loader_accepts_selected_feature_cache(tmp_path: Path) -> None:
    tensors = {
        "input_features": torch.zeros((2, 80, 10), dtype=torch.bfloat16),
        "attention_mask": torch.ones((2, 10), dtype=torch.long),
        "labels": torch.tensor([[1, 2, -100], [3, 4, 5]], dtype=torch.long),
    }
    batch_path = tmp_path / "batch_0000.pt"
    torch.save(tensors, batch_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "whisper-selected-feature-cache",
                "model_revision": "whisper-small",
                "dataset": "openslr/librispeech_asr",
                "dataset_config": "selected",
                "split": "hard-v1",
                "offset": 0,
                "samples": 2,
                "language": "en",
                "task": "transcribe",
                "feature_dtype": "bfloat16",
                "payload_bytes": batch_path.stat().st_size,
                "batches": [
                    {
                        "offset": 0,
                        "samples": 2,
                        "file": batch_path.name,
                        "bytes": batch_path.stat().st_size,
                        "sha256": sha256_file(batch_path),
                        "examples": [
                            {"id": "a", "document_id": "doc-a", "text": "one"},
                            {"id": "b", "document_id": "doc-b", "text": "two"},
                        ],
                    }
                ],
            }
        )
    )

    rows, metadata = _load_training_feature_cache(
        manifest_path,
        model="whisper-small",
        dataset="openslr/librispeech_asr",
        source=TrainingSource("selected", "hard-v1", 0, 2),
    )

    assert [row.identifier for row in rows] == ["a", "b"]
    assert rows[0].labels.tolist() == [1, 2]
    assert rows[1].labels.tolist() == [3, 4, 5]
    assert metadata["samples"] == 2
