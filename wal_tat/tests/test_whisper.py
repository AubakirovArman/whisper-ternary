from types import SimpleNamespace

import pytest
import torch

from wal_tat import (
    TransactionalTernaryLinear,
    WhisperFamilySpec,
    freeze_except_transactional,
    get_module,
    install_transactional_group,
)


class _Attention(torch.nn.Module):
    def __init__(self, features=8):
        super().__init__()
        self.q_proj = torch.nn.Linear(features, features)
        self.k_proj = torch.nn.Linear(features, features, bias=False)
        self.v_proj = torch.nn.Linear(features, features)
        self.out_proj = torch.nn.Linear(features, features)


class _EncoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.fc1 = torch.nn.Linear(8, 16)
        self.fc2 = torch.nn.Linear(16, 8)


class _DecoderLayer(_EncoderLayer):
    def __init__(self):
        super().__init__()
        self.encoder_attn = _Attention()


class _ToyWhisper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.encoder = torch.nn.Module()
        self.model.decoder = torch.nn.Module()
        self.model.encoder.layers = torch.nn.ModuleList([_EncoderLayer(), _EncoderLayer()])
        self.model.decoder.layers = torch.nn.ModuleList([_DecoderLayer(), _DecoderLayer()])


def test_whisper_family_enumerates_structural_transactions():
    model = _ToyWhisper()
    spec = WhisperFamilySpec.from_model(model)
    assert len(spec.linear_specs()) == 2 * 6 + 2 * 10
    assert len(spec.transaction_groups()) == 2 * 9 + 2 * 15
    assert spec.group("decoder.1.cross_qk").module_names == (
        "model.decoder.layers.1.encoder_attn.q_proj",
        "model.decoder.layers.1.encoder_attn.k_proj",
    )
    assert spec.group("decoder.1.mlp_in").module_names == (
        "model.decoder.layers.1.fc1",
    )
    assert spec.group("encoder.0.mlp_out").module_names == (
        "model.encoder.layers.0.fc2",
    )
    assert spec.group("decoder.1.self_o").module_names == (
        "model.decoder.layers.1.self_attn.out_proj",
    )
    assert spec.group("decoder.0.cross_k").module_names == (
        "model.decoder.layers.0.encoder_attn.k_proj",
    )


def test_install_transactional_group_preserves_bias_and_freezes_others():
    model = _ToyWhisper()
    spec = WhisperFamilySpec.from_model(model)
    group = spec.group("decoder.1.mlp")
    original_bias = get_module(model, group.module_names[0]).bias.detach().clone()
    installed = install_transactional_group(model, group, group_size=4)
    freeze_except_transactional(model, installed.values())
    first = get_module(model, group.module_names[0])
    assert isinstance(first, TransactionalTernaryLinear)
    assert torch.equal(first.bias, original_bias)
    assert first.matrix.master_weight.requires_grad
    assert first.matrix.group_scale.requires_grad
    selected = {
        id(parameter)
        for linear in installed.values()
        for parameter in (linear.matrix.master_weight, linear.matrix.group_scale)
    }
    assert all(parameter.requires_grad == (id(parameter) in selected) for parameter in model.parameters())


def test_whisper_family_rejects_wrong_layout():
    with pytest.raises(TypeError, match="Whisper layout"):
        WhisperFamilySpec.from_model(SimpleNamespace())
