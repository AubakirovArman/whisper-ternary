import torch
import torch.nn.functional as F

from wal_tat import (
    FixedTernaryLinear,
    ProxyTernaryMatrix,
    TransformedProxyTernaryLinear,
    blockwise_randomized_hadamard,
    inverse_blockwise_randomized_hadamard,
    normalized_hadamard,
)


def test_normalized_hadamard_is_self_inverse():
    value = torch.randn(3, 16)
    restored = normalized_hadamard(normalized_hadamard(value))
    assert torch.allclose(restored, value, atol=1e-6, rtol=1e-6)


def test_randomized_hadamard_round_trip():
    value = torch.randn(2, 3, 32)
    transformed = blockwise_randomized_hadamard(value, group_size=8, seed=17)
    restored = inverse_blockwise_randomized_hadamard(
        transformed, group_size=8, seed=17
    )
    assert torch.allclose(restored, value, atol=1e-6, rtol=1e-6)


def test_dense_linear_equivalence_in_rht_basis():
    weight = torch.randn(7, 16)
    value = torch.randn(2, 5, 16)
    transformed_weight = blockwise_randomized_hadamard(
        weight, group_size=8, seed=23
    )
    transformed_value = blockwise_randomized_hadamard(
        value, group_size=8, seed=23
    )
    expected = F.linear(value, weight)
    actual = F.linear(transformed_value, transformed_weight)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_fixed_rht_ternary_linear_executes_hard_codes():
    weight = torch.randn(6, 16)
    value = torch.randn(4, 16)
    layer = FixedTernaryLinear.from_weight(
        weight, group_size=8, transform="rht", transform_seed=31
    )
    assert set(layer.ternary_codes.unique().tolist()) <= {-1, 0, 1}
    expected = F.linear(value, layer.effective_weight())
    actual = layer(value)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_transform_seed_is_deterministic_and_changes_basis():
    value = torch.randn(2, 32)
    first = blockwise_randomized_hadamard(value, group_size=8, seed=41)
    repeated = blockwise_randomized_hadamard(value, group_size=8, seed=41)
    different = blockwise_randomized_hadamard(value, group_size=8, seed=43)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)


def test_transform_rejects_padding_and_non_power_of_two():
    value = torch.randn(2, 15)
    try:
        blockwise_randomized_hadamard(value, group_size=8)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("expected non-divisible input to fail")

    try:
        normalized_hadamard(torch.randn(2, 12))
    except ValueError as error:
        assert "power of two" in str(error)
    else:
        raise AssertionError("expected non-power-of-two transform to fail")


def test_transformed_proxy_executes_hard_forward_and_supplies_gradients():
    weight = torch.randn(6, 16)
    transformed = blockwise_randomized_hadamard(weight, group_size=8, seed=47)
    fixed = FixedTernaryLinear.from_weight(
        weight, group_size=8, transform="rht", transform_seed=47
    )
    proxy = ProxyTernaryMatrix(
        fixed.ternary_codes,
        fixed.group_scales,
        compute_dtype=weight.dtype,
        temperature=0.25,
    )
    layer = TransformedProxyTernaryLinear(
        proxy, transform="rht", transform_seed=47
    )
    value = torch.randn(4, 16)
    actual = layer(value)
    hard_weight = (
        proxy.hard_codes().float() * proxy.group_scale.detach().unsqueeze(-1)
    ).reshape_as(transformed)
    expected = F.linear(
        blockwise_randomized_hadamard(value, group_size=8, seed=47),
        hard_weight,
    )
    assert torch.allclose(actual.detach(), expected, atol=2e-6, rtol=2e-6)
    actual.square().mean().backward()
    assert proxy.proxy_code.grad is not None
    assert proxy.group_scale.grad is not None
