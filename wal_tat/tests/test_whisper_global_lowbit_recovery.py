import pytest
from argparse import Namespace
import hashlib
import json
from types import SimpleNamespace

import torch
from torch import nn

from whisper_global_lowbit_recovery import (
    CachedTrainingExample,
    _batch,
    _checkpoint_with_candidates,
    _load_training_feature_cache,
    _minimum_word_error_risk,
    _parse_group_names,
    _parse_training_source,
    _reference_sequence,
    _resolve_compensation_names,
    _training_sources,
    TrainingSource,
)
from whisper_expand_frontier import _batch as _expand_frontier_batch
from wal_tat import WhisperFamilySpec


def test_macro_candidate_adds_multiple_groups_atomically() -> None:
    teacher = nn.Module()
    teacher.first = nn.Linear(4, 2, bias=False)
    teacher.second = nn.Linear(4, 3, bias=True)
    parent = {
        "schema_version": 3,
        "precision": "t3",
        "group_size": 2,
        "accepted": True,
        "groups": ["existing"],
        "group": "existing",
        "matrices": {},
    }
    groups = [
        SimpleNamespace(name="first_group", module_names=("first",)),
        SimpleNamespace(name="second_group", module_names=("second",)),
    ]
    moments = {
        "first": torch.ones(4),
        "second": torch.arange(1, 5, dtype=torch.float32),
    }
    checkpoint = _checkpoint_with_candidates(
        parent,
        teacher,
        groups,
        moments,
        precision="t3",
        group_size=2,
    )
    assert set(checkpoint["matrices"]) == {"first", "second"}
    assert checkpoint["groups"] == ["existing", "first_group", "second_group"]
    assert checkpoint["group"] == "existing+first_group+second_group"
    assert checkpoint["matrices"]["first"]["precision"] == "t3"
    assert checkpoint["matrices"]["second"]["codes"].shape == (3, 2, 2)
    assert checkpoint["matrices"]["second"]["bias"] is not None
    assert parent["matrices"] == {}


def test_parse_compensation_groups_is_ordered_and_unique():
    assert _parse_group_names("decoder.2.mlp_out, encoder.1.self_k") == (
        "decoder.2.mlp_out",
        "encoder.1.self_k",
    )
    assert _parse_group_names("  ") == ()
    with pytest.raises(ValueError, match="unique"):
        _parse_group_names("decoder.2.mlp_out,decoder.2.mlp_out")


def test_minimum_word_error_risk_pushes_probability_to_better_hypothesis():
    scores = torch.tensor([[0.0, 0.0]], requires_grad=True)
    risks = torch.tensor([[0.0, 1.0]])
    loss, expected, oracle = _minimum_word_error_risk(
        scores,
        risks,
        temperature=1.0,
    )
    loss.backward()
    assert expected.item() == pytest.approx(0.5)
    assert oracle.item() == pytest.approx(0.0)
    assert scores.grad is not None
    assert scores.grad[0, 0].item() < 0
    assert scores.grad[0, 1].item() > 0


def test_minimum_word_error_risk_validates_shapes_and_temperature():
    scores = torch.zeros((1, 2))
    with pytest.raises(ValueError, match="equal shapes"):
        _minimum_word_error_risk(
            scores,
            torch.zeros((2, 1)),
            temperature=1.0,
        )
    with pytest.raises(ValueError, match="at least two"):
        _minimum_word_error_risk(
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            temperature=1.0,
        )
    with pytest.raises(ValueError, match="positive"):
        _minimum_word_error_risk(
            scores,
            torch.zeros_like(scores),
            temperature=0.0,
        )


def test_reference_sequence_removes_padding_and_prepends_decoder_start():
    assert _reference_sequence(
        torch.tensor([5, 6, -100, -100]),
        decoder_start_token_id=2,
    ).tolist() == [2, 5, 6]
    assert _reference_sequence(
        torch.tensor([2, 5, 6]),
        decoder_start_token_id=2,
    ).tolist() == [2, 5, 6]
    with pytest.raises(ValueError, match="no target"):
        _reference_sequence(
            torch.tensor([-100, -100]),
            decoder_start_token_id=2,
        )


def test_compensation_matrices_must_be_committed_and_disjoint():
    family = WhisperFamilySpec(encoder_layers=2, decoder_layers=3)
    fc1 = "model.decoder.layers.2.fc1"
    fc2 = "model.decoder.layers.2.fc2"
    assert _resolve_compensation_names(
        family,
        ("decoder.2.mlp_out",),
        parent_names={fc2},
        candidate_names={fc1},
    ) == {fc2}

    with pytest.raises(ValueError, match="not committed"):
        _resolve_compensation_names(
            family,
            ("decoder.2.mlp_out",),
            parent_names=set(),
            candidate_names={fc1},
        )
    with pytest.raises(ValueError, match="overlap candidate"):
        _resolve_compensation_names(
            family,
            ("decoder.2.mlp_in",),
            parent_names={fc1},
            candidate_names={fc1},
        )


def test_parse_multi_domain_training_sources():
    assert _parse_training_source("clean:train.100:4096:256") == TrainingSource(
        dataset_config="clean",
        split="train.100",
        offset=4096,
        samples=256,
    )
    with pytest.raises(ValueError, match="CONFIG:SPLIT"):
        _parse_training_source("clean:train.100:256")
    with pytest.raises(ValueError, match="non-negative"):
        _parse_training_source("clean:train.100:-1:256")
    with pytest.raises(ValueError, match="positive"):
        _parse_training_source("clean:train.100:0:0")


def test_training_sources_fall_back_and_reject_duplicates():
    fallback = Namespace(
        train_source=[],
        dataset_config="clean",
        train_split="train.100",
        train_offset=32,
        train_samples=64,
    )
    assert _training_sources(fallback) == (
        TrainingSource("clean", "train.100", 32, 64),
    )
    explicit = Namespace(
        train_source=[
            "clean:train.100:32:64",
            "other:train.500:96:64",
        ]
    )
    assert _training_sources(explicit) == (
        TrainingSource("clean", "train.100", 32, 64),
        TrainingSource("other", "train.500", 96, 64),
    )
    explicit.train_source.append("clean:train.100:32:64")
    with pytest.raises(ValueError, match="unique"):
        _training_sources(explicit)


def test_cached_training_examples_are_randomly_rebatchable() -> None:
    examples = [
        CachedTrainingExample(
            identifier="a",
            document_id=None,
            text="one",
            input_features=torch.full((2, 3), 1.0, dtype=torch.bfloat16),
            attention_mask=torch.tensor([1, 1, 0]),
            labels=torch.tensor([4, 5]),
        ),
        CachedTrainingExample(
            identifier="b",
            document_id="doc",
            text="two",
            input_features=torch.full((2, 3), 2.0, dtype=torch.bfloat16),
            attention_mask=torch.tensor([1, 1, 1]),
            labels=torch.tensor([6]),
        ),
    ]
    batch = _batch(
        processor=None,
        examples=[examples[1], examples[0]],
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert batch["input_features"][:, 0, 0].tolist() == [2.0, 1.0]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[6, -100], [4, 5]]
    expand_batch = _expand_frontier_batch(
        processor=None,
        examples=[examples[1], examples[0]],
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    for key in batch:
        assert torch.equal(batch[key], expand_batch[key])


def test_load_training_feature_cache_validates_and_loads_rows(tmp_path) -> None:
    payload = {
        "input_features": torch.arange(24, dtype=torch.float32)
        .reshape(2, 2, 6)
        .bfloat16(),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        "labels": torch.tensor([[4, 5, -100], [6, 7, 8]]),
    }
    batch_path = tmp_path / "batch_0000.pt"
    torch.save(payload, batch_path)
    digest = hashlib.sha256(batch_path.read_bytes()).hexdigest()
    model_path = tmp_path / "revision"
    model_path.mkdir()
    manifest = {
        "schema_version": 1,
        "kind": "whisper-batch-exact-feature-cache",
        "model_revision": "revision",
        "dataset": "dataset",
        "dataset_config": "clean",
        "split": "train.100",
        "offset": 32,
        "samples": 2,
        "language": "en",
        "task": "transcribe",
        "feature_dtype": "bfloat16",
        "payload_bytes": batch_path.stat().st_size,
        "batches": [
            {
                "offset": 32,
                "samples": 2,
                "file": batch_path.name,
                "bytes": batch_path.stat().st_size,
                "sha256": digest,
                "examples": [
                    {"id": "a", "document_id": None, "text": "one"},
                    {"id": "b", "document_id": "doc", "text": "two"},
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    examples, metadata = _load_training_feature_cache(
        manifest_path,
        model=str(model_path),
        dataset="dataset",
        source=TrainingSource("clean", "train.100", 32, 2),
    )
    assert [value.identifier for value in examples] == ["a", "b"]
    assert examples[0].labels.tolist() == [4, 5]
    assert examples[1].labels.tolist() == [6, 7, 8]
    assert metadata["samples"] == 2
    assert metadata["payload_bytes"] == batch_path.stat().st_size


def test_load_training_feature_cache_rejects_source_mismatch(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "whisper-batch-exact-feature-cache",
                "model_revision": "model",
                "dataset": "dataset",
                "dataset_config": "other",
                "split": "train.500",
                "offset": 0,
                "samples": 1,
                "language": "en",
                "task": "transcribe",
                "feature_dtype": "bfloat16",
                "batches": [],
            }
        )
    )
    with pytest.raises(ValueError, match="dataset_config differs"):
        _load_training_feature_cache(
            manifest_path,
            model="model",
            dataset="dataset",
            source=TrainingSource("clean", "train.100", 0, 1),
        )
