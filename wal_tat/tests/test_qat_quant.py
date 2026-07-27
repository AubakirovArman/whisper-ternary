"""Unit tests for the QAT quantizers and the QuantLinear layer."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from wal_tat.qat.quant import (
    PRECISIONS,
    QuantLinear,
    get_quantizer,
    grouped_view,
    moment_grouped,
)


def make_layer(**kwargs) -> QuantLinear:
    torch.manual_seed(0)
    linear = nn.Linear(kwargs.pop("in_features", 256), kwargs.pop("out_features", 8))
    return QuantLinear.from_linear(linear, **kwargs)


def test_grouped_view_pads_only_the_last_group():
    weight = torch.arange(2 * 300, dtype=torch.float32).reshape(2, 300)
    grouped = grouped_view(weight, 128)
    assert grouped.shape == (2, 3, 128)
    assert torch.equal(grouped[0, 0], weight[0, :128])
    assert torch.equal(grouped[1, 2, :44], weight[1, 256:])
    assert torch.equal(grouped[1, 2, 44:], torch.zeros(84))


def test_grouped_view_rejects_non_matrix():
    with pytest.raises(ValueError):
        grouped_view(torch.zeros(4), 128)


def test_moment_grouped_rejects_oversized_moment():
    grouped = torch.zeros(2, 2, 4)
    with pytest.raises(ValueError):
        moment_grouped(torch.ones(9), grouped)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_export_codes_uses_only_the_declared_alphabet(precision):
    layer = make_layer(precision=precision, group_size=128)
    codes, scales = layer.export_codes()
    assert codes.shape == (8, 2, 128)
    assert codes.dtype == torch.int8
    assert scales.shape == (8, 2)
    assert scales.dtype == torch.float16
    assert set(codes.unique().tolist()) <= set(get_quantizer(precision).levels)
    assert torch.all(scales > 0)


def test_ternary_group_has_at_most_three_distinct_values():
    layer = make_layer(precision="t3", group_size=64)
    codes, scales = layer.export_codes()
    approx = codes.float() * scales.float().unsqueeze(-1)
    for group in approx.reshape(-1, 64):
        assert group.unique().numel() <= 3


def test_forward_executes_the_exported_codes_exactly():
    layer = make_layer(group_size=128)
    value = torch.randn(3, 256)
    expected = torch.nn.functional.linear(
        value, layer.dequantized_weight(), layer.bias
    )
    assert torch.equal(layer(value), expected)
    with torch.no_grad():
        assert torch.equal(layer.quantized_weight(), layer.dequantized_weight())


def test_straight_through_gradient_is_the_identity():
    layer = make_layer(group_size=128)
    value = torch.randn(4, 256)
    layer(value).sum().backward()
    # d(loss)/d(w_q)[o, i] = sum_b value[b, i]; the STE passes it through
    # unchanged to the latent weight.
    expected = value.sum(0).expand(8, 256)
    assert torch.allclose(layer.weight.grad, expected, atol=1e-5)
    assert (layer.weight.grad != 0).all()


def test_group_scale_receives_its_exact_gradient():
    layer = make_layer(group_size=128)
    value = torch.randn(4, 256)
    codes, _ = layer.export_codes()
    layer(value).sum().backward()
    upstream = grouped_view(value.sum(0).expand(8, 256), 128)
    expected = (upstream * codes.float()).sum(-1)
    assert torch.allclose(layer.group_scale.grad, expected, atol=1e-4)


def test_gradient_reaches_weights_whose_code_is_zero():
    layer = make_layer(group_size=128)
    codes, _ = layer.export_codes()
    assert (codes == 0).any()
    value = torch.randn(4, 256)
    layer(value).sum().backward()
    grouped_grad = grouped_view(layer.weight.grad, 128)
    assert (grouped_grad[codes == 0] != 0).all()


def test_optimizer_step_flips_codes():
    layer = make_layer(group_size=128)
    before, _ = layer.export_codes()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-2)
    value = torch.randn(4, 256)
    layer(value).sum().backward()
    optimizer.step()
    layer.constrain_()
    assert layer.code_flips(before) > 0


def test_code_flips_rejects_mismatched_reference():
    layer = make_layer(group_size=128)
    with pytest.raises(ValueError):
        layer.code_flips(torch.zeros(1, 1, 1, dtype=torch.int8))


def test_derived_scales_ignore_padding_lanes():
    torch.manual_seed(1)
    linear = nn.Linear(300, 6)
    padded = QuantLinear.from_linear(linear, group_size=128, learnable_scales=False)
    grouped = grouped_view(padded.weight.detach(), 128)
    expected_last = grouped[:, -1, :44].abs().sum(-1) / 44
    assert torch.allclose(padded.effective_scales()[:, -1], expected_last, atol=1e-3)


def test_padding_does_not_leak_into_the_forward():
    torch.manual_seed(1)
    linear = nn.Linear(300, 6)
    layer = QuantLinear.from_linear(linear, group_size=128)
    assert layer.groups == 3 and layer.padding == 84
    value = torch.randn(2, 300)
    assert layer(value).shape == (2, 6)
    codes, scales = layer.export_codes()
    dense = (codes.float() * scales.float().unsqueeze(-1)).reshape(6, -1)
    assert torch.equal(layer.dequantized_weight(), dense[:, :300])


def test_projection_init_is_a_fixed_point_of_the_encoder():
    torch.manual_seed(2)
    linear = nn.Linear(256, 16)
    moment = torch.rand(256) * 4 + 0.1
    layer = QuantLinear.from_linear(
        linear, group_size=128, input_second_moment=moment
    )
    codes, _, _ = get_quantizer("t3").project(
        layer.weight.detach(), moment, group_size=128
    )
    exported, _ = layer.export_codes()
    disagreement = (exported != codes).float().mean().item()
    assert disagreement < 1e-3


def test_projection_init_beats_absmean_under_its_own_metric():
    torch.manual_seed(3)
    linear = nn.Linear(256, 16)
    moment = torch.rand(256) * 4 + 0.1
    derived = QuantLinear.from_linear(linear, group_size=128, learnable_scales=False)
    projected = QuantLinear.from_linear(
        linear, group_size=128, input_second_moment=moment
    )
    assert (
        projected.quant_error(moment)["weighted_relative_error"]
        < derived.quant_error(moment)["weighted_relative_error"]
    )


def test_quant_error_reports_a_complete_code_histogram():
    layer = make_layer(group_size=128)
    stats = layer.quant_error()
    fractions = [stats[f"code_fraction_{level}"] for level in (-1, 0, 1)]
    assert sum(fractions) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < stats["relative_frobenius"] < 1.0
    assert stats["scale_min"] > 0


def test_constrain_restores_positive_scales():
    layer = make_layer(group_size=128)
    with torch.no_grad():
        layer.group_scale.fill_(-3.0)
    layer.constrain_()
    assert torch.all(layer.group_scale > 0)


def test_physical_bpw_matches_the_packing_budget():
    ternary = make_layer(group_size=128, precision="t3")
    binary = make_layer(group_size=128, precision="b1")
    assert ternary.physical_bpw() == pytest.approx(2.125)
    assert binary.physical_bpw() == pytest.approx(1.125)
    assert get_quantizer("t3").entropy_bits == pytest.approx(1.5849625007)
    assert get_quantizer("b1").entropy_bits == pytest.approx(1.0)


def test_state_dict_only_holds_trainable_tensors():
    learned = make_layer(group_size=128)
    derived = make_layer(group_size=128, learnable_scales=False)
    assert set(learned.state_dict()) == {"weight", "group_scale", "bias"}
    assert set(derived.state_dict()) == {"weight", "bias"}
    assert learned.state_dict()["weight"].dtype == torch.float32


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        QuantLinear(16, 4, precision="q4")
    with pytest.raises(ValueError):
        QuantLinear(16, 4, group_size=0)
    with pytest.raises(ValueError):
        QuantLinear(0, 4)
    with pytest.raises(ValueError):
        QuantLinear(16, 4, scale_eps=0.0)
    with pytest.raises(TypeError):
        QuantLinear.from_linear(nn.Conv1d(2, 2, 1))  # type: ignore[arg-type]
    layer = make_layer(group_size=128)
    with pytest.raises(ValueError):
        layer.reset_scales_from_projection(torch.ones(3))
