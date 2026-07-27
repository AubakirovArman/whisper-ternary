import pytest
import torch

from wal_tat import (
    PackedB1Matrix,
    ProxyBinaryMatrix,
    pack_binary_codes,
    pack_binary_matrix,
    q1_g128_physical_bpw,
    read_packed_binary_matrix,
    unpack_binary_codes,
    unpack_binary_matrix,
    weighted_binary_project,
    write_packed_binary_matrix,
)


def test_binary_projection_is_strict_and_weighted():
    weight = torch.tensor([[-1.0, -3.0, 1.0, 3.0]])
    codes, scales, error = weighted_binary_project(
        weight, torch.tensor([1.0, 3.0, 1.0, 3.0]), group_size=4
    )
    assert set(codes.unique().tolist()) == {-1, 1}
    assert scales.item() == pytest.approx(2.5)
    assert error.item() >= 0


def test_binary_proxy_has_exact_hard_forward_and_gradient():
    codes = torch.tensor([[[-1, 1, -1, 1]]], dtype=torch.int8)
    matrix = ProxyBinaryMatrix(codes, torch.tensor([[2.0]]), compute_dtype=torch.float32)
    weight = matrix.effective_weight()
    assert torch.equal(weight.detach(), torch.tensor([[-2.0, 2.0, -2.0, 2.0]]))
    weight.sum().backward()
    assert torch.count_nonzero(matrix.proxy_code.grad) > 0


def test_binary_pack_roundtrip_and_bpw():
    codes = torch.tensor([-1, 1, 1, -1, 1, -1, -1, 1, 1], dtype=torch.int8)
    packed = pack_binary_codes(codes)
    assert packed.numel() == 2
    assert torch.equal(unpack_binary_codes(packed, codes.numel()), codes)
    assert q1_g128_physical_bpw() == pytest.approx(1.125)


def test_binary_pack_rejects_zero():
    with pytest.raises(ValueError, match="binary"):
        pack_binary_codes(torch.tensor([-1, 0, 1], dtype=torch.int8))


def test_binary_matrix_pack_file_roundtrip(tmp_path):
    codes = torch.tensor(
        [[[-1, 1, -1, 1, 1, -1, 1, -1]], [[1, 1, -1, -1, 1, -1, -1, 1]]],
        dtype=torch.int8,
    )
    scales = torch.tensor([[2.0], [3.0]], dtype=torch.float16)
    packed = pack_binary_matrix(codes, scales, shape=(2, 8), group_size=8)
    assert isinstance(packed, PackedB1Matrix)
    assert packed.true_bpw(include_header=False) == pytest.approx(3.0)
    expected = (codes.float() * scales.float().unsqueeze(-1)).reshape(2, 8)
    assert torch.equal(unpack_binary_matrix(packed), expected)

    path = tmp_path / "weights.walb1"
    assert write_packed_binary_matrix(path, packed) == packed.serialized_nbytes
    restored = read_packed_binary_matrix(path)
    assert torch.equal(unpack_binary_matrix(restored), expected)


def test_binary_matrix_pack_rejects_nonbinary_code():
    with pytest.raises(ValueError, match="binary codes"):
        pack_binary_matrix(
            torch.tensor([[[-1, 1, 0, 1, 1, -1, 1, -1]]]),
            torch.ones(1, 1),
            shape=(1, 8),
            group_size=8,
        )
