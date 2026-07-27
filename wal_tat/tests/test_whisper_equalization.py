import torch

from wal_tat import value_output_channel_equalization


def test_value_output_equalization_preserves_linear_composition():
    generator = torch.Generator().manual_seed(7)
    value_weight = torch.randn(6, 5, generator=generator)
    value_bias = torch.randn(6, generator=generator)
    output_weight = torch.randn(4, 6, generator=generator)
    inputs = torch.randn(3, 5, generator=generator)
    value, bias, output, scale = value_output_channel_equalization(
        value_weight, value_bias, output_weight, alpha=0.75
    )
    original = (inputs @ value_weight.T + value_bias) @ output_weight.T
    transformed = (inputs @ value.T + bias) @ output.T
    torch.testing.assert_close(transformed, original, rtol=1e-5, atol=1e-5)
    assert torch.all(scale > 0)


def test_zero_alpha_is_identity():
    value_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    output_weight = torch.tensor([[2.0, 8.0], [1.0, 4.0]])
    value, bias, output, scale = value_output_channel_equalization(
        value_weight, None, output_weight, alpha=0.0
    )
    torch.testing.assert_close(value, value_weight)
    torch.testing.assert_close(output, output_weight)
    torch.testing.assert_close(scale, torch.ones_like(scale))
    assert bias is None
