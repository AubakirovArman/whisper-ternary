from pathlib import Path

import pytest
import torch

from wal_tat import (
    pack_bool_mask,
    pack_partial_matrix,
    pack_ternary_codes,
    read_packed_matrix,
    true_artifact_bpw,
    unpack_bool_mask,
    unpack_partial_matrix,
    unpack_ternary_codes,
    write_packed_matrix,
)


def test_ternary_code_roundtrip_and_slot_order():
    codes = torch.tensor([-1, 0, 1, -1, 1, 0, -1], dtype=torch.int8)
    packed = pack_ternary_codes(codes)
    assert packed.tolist()[0] == 0b00100100
    assert torch.equal(unpack_ternary_codes(packed, codes.numel()), codes)


def test_reserved_ternary_slot_is_rejected():
    with pytest.raises(ValueError, match="reserved"):
        unpack_ternary_codes(torch.tensor([0b00000011], dtype=torch.uint8), 1)


def test_boolean_mask_roundtrip():
    mask = torch.tensor([True, False, True, True, False, False, True, False, True])
    packed = pack_bool_mask(mask)
    assert packed.tolist()[0] == 0b01001101
    assert torch.equal(unpack_bool_mask(packed, mask.numel()), mask)


def partial_fixture():
    master = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
        ],
        dtype=torch.bfloat16,
    )
    codes = torch.tensor(
        [
            [1, 0, -1, 1, -1, -1, 0, 1],
            [0, 1, 1, -1, 1, 0, 0, -1],
        ],
        dtype=torch.int8,
    )
    scales = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float16)
    mask = torch.tensor([[True, False], [False, True]])
    return master, codes, scales, mask


def test_partial_matrix_roundtrip_preserves_q2_and_bf16_fallback():
    master, codes, scales, mask = partial_fixture()
    packed = pack_partial_matrix(master, codes, scales, mask, group_size=4)
    reconstructed = unpack_partial_matrix(packed, dtype=torch.float32)
    grouped_master = master.float().view(2, 2, 4)
    grouped_codes = codes.float().view(2, 2, 4)
    expected = torch.where(
        mask.unsqueeze(-1), grouped_codes * scales.float().unsqueeze(-1), grouped_master
    ).view(2, 8)
    assert torch.equal(reconstructed, expected)
    assert packed.committed_groups == 2
    assert packed.total_groups == 4


def test_binary_file_roundtrip_and_true_size(tmp_path: Path):
    master, codes, scales, mask = partial_fixture()
    packed = pack_partial_matrix(master, codes, scales, mask, group_size=4)
    path = tmp_path / "matrix.q2g"
    size = write_packed_matrix(path, packed)
    loaded = read_packed_matrix(path)
    assert size == path.stat().st_size == packed.serialized_nbytes
    assert loaded.true_bpw() == packed.true_bpw()
    assert torch.equal(unpack_partial_matrix(loaded), unpack_partial_matrix(packed))


def test_fully_committed_q2_payload_is_2_125_bpw_at_g128():
    groups = 32
    master = torch.zeros((1, groups * 128), dtype=torch.bfloat16)
    codes = torch.zeros_like(master, dtype=torch.int8)
    scales = torch.ones((1, groups), dtype=torch.float16)
    mask = torch.ones((1, groups), dtype=torch.bool)
    packed = pack_partial_matrix(master, codes, scales, mask, group_size=128)
    assert packed.mask_packed.numel() == 0
    assert packed.true_bpw(include_header=False) == 2.125
    assert packed.true_bpw(include_header=True) > 2.125


def test_true_artifact_bpw_counts_headers_and_extra_bytes():
    master, codes, scales, mask = partial_fixture()
    packed = pack_partial_matrix(master, codes, scales, mask, group_size=4)
    expected = (packed.serialized_nbytes + 17) * 8 / master.numel()
    assert true_artifact_bpw([packed], extra_bytes=17) == expected


def test_reader_rejects_reserved_code_in_file(tmp_path: Path):
    from wal_tat.packing import HEADER

    master = torch.zeros((1, 4), dtype=torch.bfloat16)
    codes = torch.zeros_like(master, dtype=torch.int8)
    scales = torch.ones((1, 1), dtype=torch.float16)
    mask = torch.ones((1, 1), dtype=torch.bool)
    packed = pack_partial_matrix(master, codes, scales, mask, group_size=4)
    path = tmp_path / "reserved.q2g"
    write_packed_matrix(path, packed)
    data = bytearray(path.read_bytes())
    data[HEADER.size] = 0xFF
    path.write_bytes(data)
    with pytest.raises(ValueError, match="reserved"):
        read_packed_matrix(path)
