"""Unit tests for model surgery, coverage accounting and QAT persistence."""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from wal_tat.binary_packing import read_packed_binary_matrix, unpack_binary_matrix
from wal_tat.packing import read_packed_matrix, unpack_partial_matrix
from wal_tat.qat.convert import (
    QATManifest,
    collect_input_second_moments,
    convert_model_to_qat,
    export_packed_artifact,
    iter_target_linears,
    load_qat_checkpoint,
    qat_modules,
    save_qat_checkpoint,
)
from wal_tat.qat.quant import QuantLinear


class Block(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, value):
        value = self.out_proj(self.q_proj(value))
        return self.norm(value + self.fc2(torch.relu(self.fc1(value))))


class Toy(nn.Module):
    """Two blocks plus an untouched embedding and output head."""

    def __init__(self, width: int = 256, hidden: int = 512, vocabulary: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocabulary, width)
        self.layers = nn.ModuleList(Block(width, hidden) for _ in range(2))
        self.proj_out = nn.Linear(width, vocabulary, bias=False)

    def forward(self, tokens):
        value = self.embed(tokens)
        for layer in self.layers:
            value = layer(value)
        return self.proj_out(value)


def toy_tokens() -> torch.Tensor:
    return torch.randint(0, 64, (3, 5))


def build() -> Toy:
    torch.manual_seed(0)
    return Toy()


def test_conversion_covers_every_target_and_nothing_else():
    model = build()
    dense_parameters = sum(p.numel() for p in model.parameters())
    manifest = convert_model_to_qat(model, precision="t3", group_size=128)

    assert manifest.num_matrices == 8
    assert manifest.quantized_weights == 2 * (
        256 * 256 * 2 + 256 * 512 * 2
    )
    assert manifest.model_parameters == dense_parameters
    assert manifest.untouched_linear == ("proj_out",)
    assert set(qat_modules(model)) == {
        f"layers.{index}.{leaf}"
        for index in range(2)
        for leaf in ("q_proj", "out_proj", "fc1", "fc2")
    }
    assert not [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name != "proj_out"
    ]
    assert isinstance(model.proj_out, nn.Linear)
    assert isinstance(model.embed, nn.Embedding)
    assert model(toy_tokens()).shape == (3, 5, 64)


def test_manifest_accounting_is_self_consistent():
    model = build()
    manifest = convert_model_to_qat(model, group_size=128)
    assert manifest.matrix_bpw == pytest.approx(2.125)
    assert manifest.quantized_payload_bits == (
        manifest.quantized_weights * 2 + manifest.quantized_groups * 16
    )
    assert manifest.total_bits == (
        manifest.quantized_payload_bits
        + (manifest.model_parameters - manifest.quantized_weights) * 16
    )
    assert manifest.effective_bpw == pytest.approx(
        manifest.total_bits / manifest.model_parameters
    )
    assert manifest.compression_ratio == pytest.approx(
        16 / manifest.effective_bpw
    )
    assert 0 < manifest.parameter_coverage < 1
    assert manifest.linear_coverage > manifest.parameter_coverage
    assert "matrices" in manifest.summary()


def test_manifest_round_trips_through_json():
    model = build()
    manifest = convert_model_to_qat(model, group_size=128)
    restored = QATManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    assert restored == manifest


def test_targets_and_exclusions_are_honoured():
    model = build()
    manifest = convert_model_to_qat(
        model, targets=("fc1", "fc2"), exclude=("layers.1.",), group_size=128
    )
    assert [entry.name for entry in manifest.entries] == ["layers.0.fc1", "layers.0.fc2"]
    assert isinstance(model.layers[0].q_proj, nn.Linear)
    assert isinstance(model.layers[1].fc1, nn.Linear)


def test_predicate_can_select_individual_modules():
    model = build()
    manifest = convert_model_to_qat(
        model, predicate=lambda name, _module: name.endswith("fc1"), group_size=128
    )
    assert [entry.name for entry in manifest.entries] == ["layers.0.fc1", "layers.1.fc1"]


def test_iter_target_linears_also_sees_converted_layers():
    model = build()
    convert_model_to_qat(model, group_size=128)
    found = dict(iter_target_linears(model))
    assert len(found) == 8
    assert all(isinstance(module, QuantLinear) for module in found.values())
    assert "proj_out" not in found


def test_freeze_non_quantized_leaves_only_quant_parameters_trainable():
    model = build()
    convert_model_to_qat(model, group_size=128, freeze_non_quantized=True)
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert all(name.startswith("layers.") for name in trainable)
    assert not model.embed.weight.requires_grad
    assert not model.proj_out.weight.requires_grad
    assert model.layers[0].fc1.weight.requires_grad
    assert model.layers[0].fc1.group_scale.requires_grad


def test_calibration_moments_initialize_the_scales():
    model = build()
    tokens = toy_tokens()
    moments = collect_input_second_moments(model, lambda: model(tokens))
    assert len(moments) == 8
    assert moments["layers.0.fc1"].shape == (256,)
    assert torch.all(moments["layers.0.fc1"] >= 0)

    plain = build()
    weighted = build()
    convert_model_to_qat(plain, group_size=128)
    convert_model_to_qat(weighted, group_size=128, input_second_moments=moments)
    moment = moments["layers.0.fc1"]
    assert (
        qat_modules(weighted)["layers.0.fc1"].quant_error(moment)[
            "weighted_relative_error"
        ]
        <= qat_modules(plain)["layers.0.fc1"].quant_error(moment)[
            "weighted_relative_error"
        ]
    )


def test_collect_moments_requires_targets():
    with pytest.raises(ValueError):
        collect_input_second_moments(build(), lambda: None, targets=("missing",))


def test_unknown_precision_is_rejected():
    with pytest.raises(ValueError):
        convert_model_to_qat(build(), precision="int4")


def test_checkpoint_round_trip_is_exact(tmp_path):
    model = build()
    manifest = convert_model_to_qat(model, group_size=128)
    with torch.no_grad():
        model.layers[0].fc1.weight.add_(0.01)
    tokens = toy_tokens()
    with torch.no_grad():
        expected = model(tokens)

    path = save_qat_checkpoint(
        model, tmp_path / "state.pt", manifest=manifest, extra={"step": 7}
    )
    fresh = build()
    restored, extra = load_qat_checkpoint(fresh, path)
    assert extra == {"step": 7}
    assert restored.to_dict() == manifest.to_dict()
    assert set(qat_modules(fresh)) == set(qat_modules(model))
    with torch.no_grad():
        assert torch.equal(fresh(tokens), expected)


def test_checkpoint_rejects_a_foreign_format(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"format": "something-else"}, path)
    with pytest.raises(ValueError):
        load_qat_checkpoint(build(), path)


def test_ternary_export_matches_the_deployed_matrix(tmp_path):
    model = build()
    manifest = convert_model_to_qat(model, group_size=128)
    summary = export_packed_artifact(model, tmp_path / "artifact", manifest=manifest)
    assert summary["matrices"] == 8
    assert (tmp_path / "artifact" / "manifest.json").exists()
    for record in summary["files"]:
        packed = read_packed_matrix(tmp_path / "artifact" / record["file"])
        module = qat_modules(model)[record["name"]]
        assert torch.equal(unpack_partial_matrix(packed), module.dequantized_weight())
        assert record["true_bpw"] < 2.2
    assert summary["artifact_bytes"] * 8 / manifest.quantized_weights < 2.2


def test_binary_export_matches_the_deployed_matrix(tmp_path):
    model = build()
    manifest = convert_model_to_qat(model, precision="b1", group_size=128)
    summary = export_packed_artifact(model, tmp_path / "artifact", manifest=manifest)
    assert summary["precision"] == "b1"
    for record in summary["files"]:
        packed = read_packed_binary_matrix(tmp_path / "artifact" / record["file"])
        module = qat_modules(model)[record["name"]]
        assert torch.equal(unpack_binary_matrix(packed), module.dequantized_weight())
        assert record["true_bpw"] < 1.2


def test_export_requires_converted_modules(tmp_path):
    with pytest.raises(ValueError):
        export_packed_artifact(build(), tmp_path / "artifact")


def test_export_overwrites_a_previous_artifact(tmp_path):
    model = build()
    convert_model_to_qat(model, group_size=128)
    first = export_packed_artifact(model, tmp_path / "artifact")
    second = export_packed_artifact(model, tmp_path / "artifact")
    assert first["artifact_bytes"] == second["artifact_bytes"]
