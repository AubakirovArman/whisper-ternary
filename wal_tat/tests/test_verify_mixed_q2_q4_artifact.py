import sys
import math
from pathlib import Path

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from wal_tat import valid_group_weight_count
from verify_mixed_q2_q4_artifact import perplexity_ratios, ratio_gate_passed


def test_valid_weight_count_handles_partial_last_group():
    mask = torch.tensor([[True, False], [True, True]])
    assert valid_group_weight_count(mask, columns=6, group_size=4) == 10


def test_perplexity_ratios_use_additive_nll_not_nll_ratio():
    source = {"code": {"nll": 1.7}, "prose": {"nll": 3.5}}
    candidate = {"code": {"nll": 1.71}, "prose": {"nll": 3.51}}

    result = perplexity_ratios(candidate, source)

    assert result["code"] == pytest.approx(result["prose"])
    assert result["code"] == pytest.approx(math.exp(0.01))


def test_perplexity_gate_separates_cumulative_and_parent_budgets():
    assert ratio_gate_passed({"code": 1.00929, "prose": 1.004}, 1.01)
    assert not ratio_gate_passed({"code": 1.01001}, 1.01)
    assert ratio_gate_passed({"code": 1.00015}, 1.001)


def test_ratio_gate_rejects_improvement_limit_below_one():
    with pytest.raises(ValueError, match="at least 1.0"):
        ratio_gate_passed({"code": 0.99}, 0.999)
