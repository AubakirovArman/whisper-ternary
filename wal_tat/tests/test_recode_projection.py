import sys
from pathlib import Path

import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from commit_ternary_recode_artifact import grouped  # noqa: E402
from counterfactual_scale_recovery import FixedCodeScaleLinear  # noqa: E402
from distill_residual_to_ternary import make_candidate  # noqa: E402
from wal_tat import TransactionalTernaryMatrix  # noqa: E402


def test_fixed_code_ls_preserves_codes_and_recovers_shared_scale():
    source_codes = torch.tensor([[[-1, 0, 1, 1]]], dtype=torch.int8)
    source_scales = torch.ones((1, 1))
    target = source_codes.float() * 2.0
    codes, scales = make_candidate(
        "fixed_code_ls", target, source_codes, source_scales
    )
    assert torch.equal(codes, source_codes)
    assert torch.equal(scales, torch.tensor([[2.0]]))


def test_all_recode_recipes_emit_strict_ternary_codes_and_positive_scales():
    source_codes = torch.tensor([[[-1, 0, 1, 1]]], dtype=torch.int8)
    source_scales = torch.ones((1, 1))
    target = torch.tensor([[[-1.8, -0.1, 2.2, 1.7]]])
    for recipe in (
        "fixed_code_ls",
        "fixed_scale_recode",
        "absmean_recode",
        "lloyd_recode",
        "threshold_search",
    ):
        codes, scales = make_candidate(
            recipe, target, source_codes, source_scales
        )
        assert set(codes.flatten().tolist()) <= {-1, 0, 1}
        assert torch.all(scales > 0)


def test_grouped_pads_only_the_input_axis():
    codes = torch.tensor([[1, 0, -1, 1, 0]], dtype=torch.int8)
    result = grouped(codes, (1, 5), group_size=4)
    assert result.shape == (1, 2, 4)
    assert torch.equal(
        result.reshape(-1), torch.tensor([1, 0, -1, 1, 0, 0, 0, 0])
    )


def test_fixed_code_scale_layer_uses_fp16_forward_with_ste_gradient():
    matrix = TransactionalTernaryMatrix(
        torch.tensor([[1.0, -1.0, 0.0, 1.0]]), group_size=4
    )
    with torch.no_grad():
        matrix.committed_mask.fill_(True)
        matrix.committed_codes.copy_(
            torch.tensor([[[1, -1, 0, 1]]], dtype=torch.int8)
        )
        matrix.group_scale.fill_(1.0)
    layer = FixedCodeScaleLinear(
        matrix, max_abs_log_scale_delta=0.25, fake_fp16_scale=True
    )
    with torch.no_grad():
        layer.log_scale_delta.fill_(0.00123)
    continuous = layer.base_scale * layer.log_scale_delta.exp()
    effective = layer.effective_scales()
    assert torch.equal(effective.detach(), continuous.half().float())
    effective.sum().backward()
    assert layer.log_scale_delta.grad is not None
    assert torch.count_nonzero(layer.log_scale_delta.grad) == 1
