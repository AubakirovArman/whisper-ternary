import pytest
import torch
import torch.nn as nn

from wal_tat import (
    FixedMixedQ2Q4Linear,
    install_mixed_q2_q4_artifact,
    q2_g128_physical_bpw,
    q4_g128_physical_bpw,
    q8_g128_physical_bpw,
    valid_group_weight_count,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.old = nn.Linear(130, 2, bias=False)
        self.new = nn.Linear(130, 2, bias=False)
        self.norm = nn.LayerNorm(130, elementwise_affine=True, bias=False)


class TinyTiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(2, 130)
        self.head = nn.Linear(130, 2, bias=False)
        self.head.weight = self.embed.weight


def source_entry():
    codes = torch.zeros((2, 130), dtype=torch.int8)
    codes[0, :4] = torch.tensor([1, 0, -1, 1], dtype=torch.int8)
    return {
        "shape": (2, 130),
        "group_size": 128,
        "committed_mask": torch.tensor([[True, False], [False, False]]),
        "ternary_codes_int8": codes,
        "scales_fp16": torch.tensor([[0.5, 1.0], [1.0, 1.0]], dtype=torch.float16),
    }


def artifact_entry(source_mask):
    q2_codes = torch.zeros((2, 2, 128), dtype=torch.int8)
    q2_codes[0, 0, :4] = torch.tensor([1, 0, -1, 1], dtype=torch.int8)
    q2_codes[1, 0, :4] = torch.tensor([1, 1, 0, -1], dtype=torch.int8)
    q4_codes = torch.zeros((2, 2, 128), dtype=torch.int8)
    q4_codes[0, 1, :2] = torch.tensor([2, -2], dtype=torch.int8)
    q4_codes[1, 1, :2] = torch.tensor([3, -1], dtype=torch.int8)
    return {
        "shape": (2, 130),
        "source_committed_mask": source_mask,
        "q4_mask": torch.tensor([[False, True], [False, True]]),
        "q2_codes_int8": q2_codes,
        "q2_scales_fp16": torch.tensor(
            [[0.5, 1.0], [0.25, 1.0]], dtype=torch.float16
        ),
        "q4_codes_int8": q4_codes,
        "q4_scales_fp16": torch.tensor(
            [[1.0, 0.125], [1.0, 0.25]], dtype=torch.float16
        ),
    }


def artifact(matrices):
    return {
        "format": "wal-tat-mixed-q2-q4-v1",
        "source_checkpoint_sha256": "source-sha",
        "group_size": 128,
        "q2_physical_bpw": q2_g128_physical_bpw(),
        "q4_physical_bpw": q4_g128_physical_bpw(),
        "matrices": matrices,
    }


def test_valid_group_weight_count_handles_partial_last_group():
    mask = torch.tensor([[True, False], [True, True]])
    assert valid_group_weight_count(mask, columns=6, group_size=4) == 10


def test_installer_accepts_new_matrix_and_preserves_source_codes():
    model = TinyModel()
    source = {"matrices": {"old": source_entry()}}
    old = artifact_entry(source["matrices"]["old"]["committed_mask"].clone())
    new = artifact_entry(torch.zeros((2, 2), dtype=torch.bool))
    payload = artifact({"old": old, "new": new})
    result = install_mixed_q2_q4_artifact(
        model,
        payload,
        source,
        device="cpu",
        expected_source_sha256="source-sha",
    )

    assert isinstance(model.old, FixedMixedQ2Q4Linear)
    assert isinstance(model.new, FixedMixedQ2Q4Linear)
    assert result.new_q2_weights == 384
    assert result.q4_weights == 8
    assert torch.equal(
        model.new.weight[:, :4],
        torch.tensor([[0.5, 0.0, -0.5, 0.5], [0.25, 0.25, 0.0, -0.25]]),
    )
    assert torch.equal(
        model.new.weight[:, 128:],
        torch.tensor([[0.25, -0.25], [0.75, -0.25]]),
    )


def test_installer_rejects_source_code_mutation():
    model = TinyModel()
    source = {"matrices": {"old": source_entry()}}
    entry = artifact_entry(source["matrices"]["old"]["committed_mask"].clone())
    entry["q2_codes_int8"][0, 0, 0] = -1
    payload = artifact({"old": entry})
    with pytest.raises(ValueError, match="accepted source codes changed"):
        install_mixed_q2_q4_artifact(
            model,
            payload,
            source,
            device="cpu",
            expected_source_sha256="source-sha",
        )


def test_installer_rejects_source_scale_mutation_by_default():
    model = TinyModel()
    source = {"matrices": {"old": source_entry()}}
    entry = artifact_entry(source["matrices"]["old"]["committed_mask"].clone())
    entry["q2_scales_fp16"][0, 0] = 0.625
    payload = artifact({"old": entry})
    with pytest.raises(ValueError, match="accepted source scales changed"):
        install_mixed_q2_q4_artifact(
            model,
            payload,
            source,
            device="cpu",
            expected_source_sha256="source-sha",
        )


def test_installer_allows_explicit_source_q2_scale_recovery_only():
    model = TinyModel()
    source = {"matrices": {"old": source_entry()}}
    entry = artifact_entry(source["matrices"]["old"]["committed_mask"].clone())
    entry["q2_scales_fp16"][0, 0] = 0.625
    entry["source_q2_scale_recovery"] = True
    payload = artifact({"old": entry})
    payload["allow_source_q2_scale_recovery"] = True

    install_mixed_q2_q4_artifact(
        model,
        payload,
        source,
        device="cpu",
        expected_source_sha256="source-sha",
    )

    assert torch.equal(
        model.old.weight[0, :4],
        torch.tensor([0.625, 0.0, -0.625, 0.625]),
    )


def test_installer_rejects_source_commitment_on_new_matrix():
    model = TinyModel()
    entry = artifact_entry(torch.tensor([[True, False], [False, False]]))
    payload = artifact({"new": entry})
    with pytest.raises(ValueError, match="new artifact matrix has source commitments"):
        install_mixed_q2_q4_artifact(
            model,
            payload,
            {"matrices": {}},
            device="cpu",
            expected_source_sha256="source-sha",
        )


def test_installer_supports_disjoint_q8_rescue():
    model = TinyModel()
    entry = artifact_entry(torch.zeros((2, 2), dtype=torch.bool))
    entry["q4_mask"] = torch.tensor([[False, False], [False, True]])
    entry["q8_mask"] = torch.tensor([[False, True], [False, False]])
    entry["q8_codes_int8"] = torch.zeros((2, 2, 128), dtype=torch.int8)
    entry["q8_codes_int8"][0, 1, :2] = torch.tensor([10, -10], dtype=torch.int8)
    entry["q8_scales_fp16"] = torch.ones((2, 2), dtype=torch.float16)
    entry["q8_scales_fp16"][0, 1] = 0.125
    payload = artifact({"new": entry})
    payload["format"] = "wal-tat-mixed-q2-q4-q8-v1"
    payload["q8_physical_bpw"] = q8_g128_physical_bpw()

    result = install_mixed_q2_q4_artifact(
        model,
        payload,
        {"matrices": {}},
        device="cpu",
        expected_source_sha256="source-sha",
    )

    assert result.q8_weights == 2
    assert result.q4_weights == 2
    assert torch.equal(model.new.weight[0, 128:], torch.tensor([1.25, -1.25]))


def test_installer_supports_explicit_no_zero_q2_groups():
    model = TinyModel()
    entry = artifact_entry(torch.zeros((2, 2), dtype=torch.bool))
    entry["q4_mask"] = torch.zeros((2, 2), dtype=torch.bool)
    entry["nz4_mask"] = torch.tensor([[True, False], [False, False]])
    entry["q2_codes_int8"][0, 0] = torch.tensor([-3, -1, 1, 3]).repeat(32)
    entry["q2_scales_fp16"][0, 0] = 0.5
    payload = artifact({"new": entry})

    result = install_mixed_q2_q4_artifact(
        model,
        payload,
        {"matrices": {}},
        device="cpu",
        expected_source_sha256="source-sha",
    )

    assert result.new_q2_weights == 260
    assert result.nz4_weights == 128
    assert torch.equal(model.new.weight[0, :4], torch.tensor([-1.5, -0.5, 0.5, 1.5]))


def test_installer_rejects_zero_inside_no_zero_q2_group():
    model = TinyModel()
    entry = artifact_entry(torch.zeros((2, 2), dtype=torch.bool))
    entry["q4_mask"] = torch.zeros((2, 2), dtype=torch.bool)
    entry["nz4_mask"] = torch.tensor([[True, False], [False, False]])
    with pytest.raises(ValueError, match="invalid no-zero Q2 code"):
        install_mixed_q2_q4_artifact(
            model,
            artifact({"new": entry}),
            {"matrices": {}},
            device="cpu",
            expected_source_sha256="source-sha",
        )


def test_installer_preserves_one_shared_tied_embedding_head_weight():
    model = TinyTiedModel()
    entry = artifact_entry(torch.zeros((2, 2), dtype=torch.bool))
    entry["kind"] = "tied_embedding_head"
    entry["tied_linear_name"] = "head"
    payload = artifact({"embed": entry})

    result = install_mixed_q2_q4_artifact(
        model,
        payload,
        {"matrices": {}},
        device="cpu",
        expected_source_sha256="source-sha",
    )

    assert result.new_q2_weights == 256
    assert result.q4_weights == 4
    assert model.embed.weight is model.head.weight
    tokens = torch.tensor([[0, 1]])
    hidden = model.embed(tokens)
    assert torch.equal(hidden, model.head.weight.index_select(0, tokens.reshape(-1)).view_as(hidden))


def test_installer_rejects_duplicate_tied_head_entry():
    model = TinyTiedModel()
    entry = artifact_entry(torch.zeros((2, 2), dtype=torch.bool))
    entry["kind"] = "tied_embedding_head"
    entry["tied_linear_name"] = "head"
    payload = artifact({"embed": entry, "head": artifact_entry(torch.zeros((2, 2), dtype=torch.bool))})
    with pytest.raises(ValueError, match="duplicate matrix entry"):
        install_mixed_q2_q4_artifact(
            model,
            payload,
            {"matrices": {}},
            device="cpu",
            expected_source_sha256="source-sha",
        )


def test_installer_requires_explicit_norm_recovery_flag():
    model = TinyModel()
    payload = artifact({})
    payload["norm_extras"] = {"norm.weight": torch.full((130,), 0.75)}
    with pytest.raises(ValueError, match="not explicitly enabled"):
        install_mixed_q2_q4_artifact(
            model,
            payload,
            {"matrices": {}},
            device="cpu",
            expected_source_sha256="source-sha",
        )


def test_installer_applies_explicit_finite_norm_recovery():
    model = TinyModel()
    payload = artifact({})
    payload["allow_norm_recovery"] = True
    payload["norm_extras"] = {"norm.weight": torch.full((130,), 0.75)}

    install_mixed_q2_q4_artifact(
        model,
        payload,
        {"matrices": {}},
        device="cpu",
        expected_source_sha256="source-sha",
    )

    assert torch.equal(model.norm.weight, torch.full((130,), 0.75))
