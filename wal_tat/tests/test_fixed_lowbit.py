import hashlib
import json

import pytest
import torch

from wal_tat import (
    FixedGroupwiseLinear,
    FixedPartialGroupwiseLinear,
    install_fixed_checkpoint,
    install_packed_lowbit_manifest,
    install_trainable_lowbit_checkpoint,
    install_packed_t3_manifest,
    pack_binary_matrix,
    pack_partial_matrix,
    project_linear_module,
    write_packed_matrix,
    write_packed_binary_matrix,
)


@pytest.mark.parametrize(
    ("precision", "allowed"),
    [
        ("b1", {-1, 1}),
        ("t3", {-1, 0, 1}),
        ("nz4", {-3, -1, 1, 3}),
        ("q4", set(range(-8, 8))),
    ],
)
def test_project_linear_module_uses_declared_codebook(precision, allowed):
    linear = torch.nn.Linear(8, 4, bias=True)
    fixed, error, histogram = project_linear_module(
        linear, precision=precision, group_size=4
    )
    assert isinstance(fixed, FixedGroupwiseLinear)
    assert set(fixed.codes.unique().tolist()) <= allowed
    assert error >= 0
    assert sum(histogram.values()) == linear.weight.numel()
    assert fixed(torch.randn(2, 8)).shape == (2, 4)


def test_install_fixed_checkpoint_replaces_fresh_linear_exactly():
    model = torch.nn.Module()
    model.target = torch.nn.Linear(4, 2, bias=True)
    checkpoint = {
        "precision": "t3",
        "matrices": {
            "target": {
                "codes": torch.tensor(
                    [[[-1, 0, 1, 1]], [[1, -1, 0, 1]]], dtype=torch.int8
                ),
                "scales": torch.tensor([[2.0], [3.0]], dtype=torch.float16),
                "bias": torch.tensor([0.5, -0.5]),
            }
        },
    }
    assert install_fixed_checkpoint(model, checkpoint) == ("target",)
    expected = torch.tensor([[-2.0, 0.0, 2.0, 2.0], [3.0, -3.0, 0.0, 3.0]])
    assert torch.equal(model.target.effective_weight(), expected)


def test_install_fixed_checkpoint_supports_per_matrix_mixed_precision():
    model = torch.nn.Module()
    model.first = torch.nn.Linear(4, 1, bias=False)
    model.second = torch.nn.Linear(4, 1, bias=False)
    checkpoint = {
        "precision": "mixed",
        "matrices": {
            "first": {
                "precision": "b1",
                "codes": torch.tensor([[[-1, 1, -1, 1]]], dtype=torch.int8),
                "scales": torch.ones(1, 1),
            },
            "second": {
                "precision": "q4",
                "codes": torch.tensor([[[-8, -1, 0, 7]]], dtype=torch.int8),
                "scales": torch.ones(1, 1),
            },
        },
    }
    assert install_fixed_checkpoint(model, checkpoint) == ("first", "second")
    assert set(model.first.codes.unique().tolist()) == {-1, 1}
    assert set(model.second.codes.unique().tolist()) == {-8, -1, 0, 7}


def test_install_fixed_checkpoint_supports_partial_t3_with_exact_fallback():
    model = torch.nn.Module()
    model.target = torch.nn.Linear(4, 1, bias=False)
    checkpoint = {
        "precision": "t3",
        "matrices": {
            "target": {
                "precision": "t3",
                "codes": torch.tensor([[[-1, 1], [1, -1]]], dtype=torch.int8),
                "scales": torch.tensor([[2.0, 9.0]], dtype=torch.float16),
                "committed_mask": torch.tensor([[True, False]]),
                "base_weight": torch.tensor([[7.0, 8.0, 3.0, 4.0]]),
            }
        },
    }
    assert install_fixed_checkpoint(model, checkpoint) == ("target",)
    assert isinstance(model.target, FixedPartialGroupwiseLinear)
    assert torch.equal(
        model.target.effective_weight(), torch.tensor([[-2.0, 2.0, 3.0, 4.0]])
    )


def test_install_trainable_checkpoint_preserves_partial_t3_fallback():
    model = torch.nn.Module()
    model.target = torch.nn.Linear(4, 1, bias=False)
    checkpoint = {
        "precision": "t3",
        "matrices": {
            "target": {
                "precision": "t3",
                "codes": torch.tensor([[[-1, 1], [1, -1]]], dtype=torch.int8),
                "scales": torch.tensor([[2.0, 9.0]], dtype=torch.float16),
                "committed_mask": torch.tensor([[True, False]]),
                "base_weight": torch.tensor([[7.0, 8.0, 3.0, 4.0]]),
            }
        },
    }
    proxies = install_trainable_lowbit_checkpoint(
        model, checkpoint, initial_proxy_magnitude=0.75, fake_fp16_scale=False
    )
    assert torch.equal(
        proxies["target"].effective_weight(),
        torch.tensor([[-2.0, 2.0, 3.0, 4.0]]),
    )


def test_install_trainable_lowbit_checkpoint_keeps_exact_hard_forward():
    model = torch.nn.Module()
    model.first = torch.nn.Linear(4, 2, bias=False)
    model.second = torch.nn.Linear(4, 2, bias=False)
    checkpoint = {
        "precision": "mixed",
        "matrices": {
            "first": {
                "precision": "b1",
                "codes": torch.tensor([[[-1, 1, -1, 1]], [[1, 1, -1, -1]]]),
                "scales": torch.tensor([[2.0], [3.0]]),
                "bias": None,
            },
            "second": {
                "precision": "t3",
                "codes": torch.tensor([[[-1, 0, 1, 0]], [[1, 0, -1, 1]]]),
                "scales": torch.tensor([[1.5], [0.5]]),
                "bias": None,
            },
        },
    }
    proxies = install_trainable_lowbit_checkpoint(
        model, checkpoint, initial_proxy_magnitude=0.75, fake_fp16_scale=False
    )
    expected_first = torch.tensor([[-2.0, 2.0, -2.0, 2.0], [3.0, 3.0, -3.0, -3.0]])
    expected_second = torch.tensor([[-1.5, 0.0, 1.5, 0.0], [0.5, 0.0, -0.5, 0.5]])
    assert torch.equal(model.first.matrix.hard_codes(), checkpoint["matrices"]["first"]["codes"])
    assert torch.equal(model.second.matrix.hard_codes(), checkpoint["matrices"]["second"]["codes"])
    assert torch.equal(model.first.matrix.effective_weight(), expected_first)
    assert torch.equal(model.second.matrix.effective_weight(), expected_second)
    assert set(proxies) == {"first", "second"}


def test_trainable_lowbit_checkpoint_supports_binary_boundary_override():
    model = torch.nn.Module()
    model.binary = torch.nn.Linear(4, 1, bias=False)
    model.ternary = torch.nn.Linear(4, 1, bias=False)
    checkpoint = {
        "precision": "mixed",
        "matrices": {
            "binary": {
                "precision": "b1",
                "codes": torch.tensor([[[-1, 1, -1, 1]]]),
                "scales": torch.ones(1, 1),
            },
            "ternary": {
                "precision": "t3",
                "codes": torch.tensor([[[-1, 0, 1, 0]]]),
                "scales": torch.ones(1, 1),
            },
        },
    }
    proxies = install_trainable_lowbit_checkpoint(
        model,
        checkpoint,
        initial_proxy_magnitude=0.75,
        initial_proxy_magnitudes={"binary": 0.05},
        fake_fp16_scale=False,
    )
    assert torch.equal(proxies["binary"].hard_codes(), checkpoint["matrices"]["binary"]["codes"])
    assert torch.equal(proxies["ternary"].hard_codes(), checkpoint["matrices"]["ternary"]["codes"])
    assert torch.allclose(proxies["binary"].proxy_code.abs(), torch.full((1, 1, 4), 0.05))
    assert torch.allclose(proxies["ternary"].proxy_code.abs().amax(), torch.tensor(0.75))


def test_trainable_lowbit_checkpoint_supports_ternary_zero_boundary_override():
    model = torch.nn.Module()
    model.target = torch.nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.target.weight.copy_(torch.tensor([[-2.0, -0.2, 0.3, 2.0]]))
    checkpoint = {
        "precision": "t3",
        "matrices": {
            "target": {
                "codes": torch.tensor([[[-1, 0, 0, 1]]]),
                "scales": torch.ones(1, 1),
            }
        },
    }
    proxies = install_trainable_lowbit_checkpoint(
        model,
        checkpoint,
        initial_proxy_magnitudes={"target": 0.52},
        initial_zero_proxy_boundaries={"target": 0.48},
        fake_fp16_scale=False,
    )
    matrix = proxies["target"]
    assert torch.equal(matrix.hard_codes(), checkpoint["matrices"]["target"]["codes"])
    assert torch.allclose(
        matrix.proxy_code.detach(),
        torch.tensor([[[-0.52, -0.48, 0.48, 0.52]]]),
    )


def test_trainable_lowbit_checkpoint_rejects_small_ternary_override():
    model = torch.nn.Module()
    model.target = torch.nn.Linear(4, 1, bias=False)
    checkpoint = {
        "precision": "t3",
        "matrices": {
            "target": {
                "codes": torch.tensor([[[-1, 0, 1, 0]]]),
                "scales": torch.ones(1, 1),
            }
        },
    }
    with pytest.raises(ValueError, match="at least 0.5 for T3"):
        install_trainable_lowbit_checkpoint(
            model,
            checkpoint,
            initial_proxy_magnitudes={"target": 0.05},
        )


def test_trainable_lowbit_checkpoint_rejects_q4_parent():
    model = torch.nn.Module()
    model.target = torch.nn.Linear(4, 1, bias=False)
    checkpoint = {
        "precision": "q4",
        "matrices": {
            "target": {
                "codes": torch.tensor([[[1, 2, 3, 4]]]),
                "scales": torch.tensor([[1.0]]),
                "bias": None,
            }
        },
    }
    with pytest.raises(ValueError, match="only b1/t3"):
        install_trainable_lowbit_checkpoint(model, checkpoint)


def test_install_packed_t3_manifest_uses_serialized_codes_and_scales(tmp_path):
    model = torch.nn.Module()
    model.target = torch.nn.Linear(4, 2, bias=True)
    codes = torch.tensor(
        [[[-1, 0, 1, 1]], [[1, -1, 0, 1]]], dtype=torch.int8
    )
    scales = torch.tensor([[2.0], [3.0]], dtype=torch.float16)
    packed = pack_partial_matrix(
        torch.empty(2, 4, dtype=torch.bfloat16),
        codes,
        scales,
        torch.ones(2, 1, dtype=torch.bool),
        group_size=4,
    )
    matrix_path = tmp_path / "target.waltq2"
    write_packed_matrix(matrix_path, packed)
    bias_path = tmp_path / "target.bias.bf16"
    bias = torch.tensor([0.5, -0.5], dtype=torch.bfloat16)
    bias_path.write_bytes(bias.view(torch.uint8).numpy().tobytes())

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "schema_version": 1,
        "precision": "t3",
        "entries": [
            {
                "name": "target",
                "matrix_file": matrix_path.name,
                "matrix_bytes": matrix_path.stat().st_size,
                "matrix_sha256": sha(matrix_path),
                "bias_file": bias_path.name,
                "bias_bytes": bias_path.stat().st_size,
                "bias_sha256": sha(bias_path),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assert install_packed_t3_manifest(model, manifest_path) == ("target",)
    expected = torch.tensor([[-2.0, 0.0, 2.0, 2.0], [3.0, -3.0, 0.0, 3.0]])
    assert torch.equal(model.target.effective_weight(), expected)
    assert torch.equal(model.target.bias, bias)


def test_install_packed_lowbit_manifest_supports_mixed_b1_t3(tmp_path):
    model = torch.nn.Module()
    model.binary = torch.nn.Linear(8, 1, bias=False)
    model.ternary = torch.nn.Linear(8, 1, bias=False)

    binary_codes = torch.tensor([[[-1, 1, -1, 1, 1, -1, 1, -1]]])
    binary_scales = torch.tensor([[2.0]], dtype=torch.float16)
    binary = pack_binary_matrix(
        binary_codes, binary_scales, shape=(1, 8), group_size=8
    )
    binary_path = tmp_path / "binary.walb1"
    write_packed_binary_matrix(binary_path, binary)

    ternary_codes = torch.tensor([[[-1, 0, 1, 0, 1, -1, 0, 1]]])
    ternary_scales = torch.tensor([[3.0]], dtype=torch.float16)
    ternary = pack_partial_matrix(
        torch.empty(1, 8, dtype=torch.bfloat16),
        ternary_codes,
        ternary_scales,
        torch.ones(1, 1, dtype=torch.bool),
        group_size=8,
    )
    ternary_path = tmp_path / "ternary.waltq2"
    write_packed_matrix(ternary_path, ternary)

    def entry(name, precision, path):
        return {
            "name": name,
            "precision": precision,
            "matrix_file": path.name,
            "matrix_bytes": path.stat().st_size,
            "matrix_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bias_file": None,
            "bias_bytes": 0,
            "bias_sha256": None,
        }

    manifest = {
        "schema_version": 2,
        "precision": "mixed",
        "precision_policy": "strict_b1_t3_only",
        "entries": [
            entry("binary", "b1", binary_path),
            entry("ternary", "t3", ternary_path),
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assert install_packed_lowbit_manifest(model, manifest_path) == (
        "binary",
        "ternary",
    )
    assert torch.equal(
        model.binary.effective_weight(),
        (binary_codes.float() * binary_scales.float().unsqueeze(-1)).reshape(1, 8),
    )
    assert torch.equal(
        model.ternary.effective_weight(),
        (ternary_codes.float() * ternary_scales.float().unsqueeze(-1)).reshape(1, 8),
    )


def test_install_packed_lowbit_manifest_supports_partial_t3_fallback(tmp_path):
    model = torch.nn.Module()
    model.target = torch.nn.Linear(8, 1, bias=False, dtype=torch.bfloat16)
    master = torch.tensor(
        [[10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0]],
        dtype=torch.bfloat16,
    )
    codes = torch.tensor([[[1, 0, -1, 1], [0, 0, 0, 0]]], dtype=torch.int8)
    scales = torch.tensor([[2.0, 1.0]], dtype=torch.float16)
    packed = pack_partial_matrix(
        master,
        codes,
        scales,
        torch.tensor([[True, False]]),
        group_size=4,
    )
    matrix_path = tmp_path / "target.waltq2"
    write_packed_matrix(matrix_path, packed)
    manifest = {
        "schema_version": 3,
        "precision": "mixed",
        "precision_policy": "strict_b1_t3_partial_bf16_fallback",
        "entries": [
            {
                "name": "target",
                "precision": "t3",
                "matrix_file": matrix_path.name,
                "matrix_bytes": matrix_path.stat().st_size,
                "matrix_sha256": hashlib.sha256(
                    matrix_path.read_bytes()
                ).hexdigest(),
                "bias_file": None,
                "bias_bytes": 0,
                "bias_sha256": None,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    assert install_packed_lowbit_manifest(model, manifest_path) == ("target",)
    assert torch.equal(
        model.target.effective_weight(),
        torch.tensor(
            [[2.0, 0.0, -2.0, 2.0, 20.0, 21.0, 22.0, 23.0]],
            dtype=torch.bfloat16,
        ),
    )
